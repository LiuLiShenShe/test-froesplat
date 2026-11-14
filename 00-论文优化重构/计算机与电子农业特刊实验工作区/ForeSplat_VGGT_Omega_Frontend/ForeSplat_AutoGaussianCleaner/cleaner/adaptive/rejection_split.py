"""Native derivatives subdivide rejected groups; derivatives never certify edits."""
import math
import numpy as np
import torch
from ..foreground_protection import regions
from ...evaluation.metrics import arrays
from ..io import write_json


def native_alpha(camera, model):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height, image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx/2), tanfovy=math.tan(camera.FoVy/2),
        kernel_size=0., bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=model.active_sh_degree, sg_degree=model.active_sg_degree,
        campos=camera.camera_center, prefiltered=False, require_depth=True, debug=False)
    scales, opacity = model.get_scaling_n_opacity_with_3D_filter
    output = GaussianRasterizer(raster_settings=settings)(
        means3D=model.get_xyz.detach(), means2D=torch.zeros_like(model.get_xyz),
        shs=None, sg_axis=model.get_sg_axis.detach(),
        sg_sharpness=model.get_sg_sharpness.detach(), sg_color=model.get_sg_color.detach(),
        colors_precomp=torch.ones((len(model.get_xyz), 3), device='cuda'),
        opacities=opacity, scales=scales.detach(), rotations=model.get_rotation.detach(),
        cov3Ds_precomp=None)
    return output[0].mean(0)


def select_with_budgets(benefit, risk, budgets, allowed):
    """Stable independent-risk packing, without cancellation by negative losses."""
    benefit = np.asarray(benefit)
    risk = np.asarray(risk)
    if risk.size:
        ratio = risk/np.maximum(np.asarray(budgets)[:, None], 1e-10)
        order = np.argsort(-benefit/(1+ratio.max(0)), kind='stable')
        used = np.zeros(len(budgets))
    else:
        order = np.argsort(-benefit, kind='stable'); used = np.empty(0)
    chosen = []
    for position in order:
        if not allowed[position]: continue
        if not risk.size or np.all(used+risk[:, position] <= np.asarray(budgets)+1e-10):
            chosen.append(position)
            if risk.size: used += risk[:, position]
    return np.asarray(chosen, np.int64), used


def split_rejected(scene, model, rejected, local, ids, benefits, reasons, current, initial, config, folder):
    """Return disjoint original-ID groups proposed from actual rejecting cameras.

    Photo gradients at full and half local opacity reject harmful members.
    Alpha gradients at full/half/5% opacity bound remaining dark/foreground
    mass and newly exposed boundary/thin holes. All proposed children must
    still pass the ordinary complete nonlinear multiview evaluator.
    """
    local = np.asarray(local, np.int64)
    if len(local) < 2 or len(local) > config.get('gradient_split_max_gaussians', 512):
        return []
    names = {v['name']: v['index'] for v in current['views']}
    flags = {}
    for reason in reasons:
        if ':' not in reason: continue
        name, rule = reason.split(':', 1)
        if name in names: flags.setdefault(names[name], set()).update(rule.split(','))
    # Rejection cameras derive entirely from this trial. Never use plant names,
    # manual ROIs, hand selected views or historical primitive IDs.
    selected_views = sorted(flags, key=lambda i: (-len(flags[i]), i))[:config.get('gradient_split_max_views', 8)]
    if not selected_views: return []
    original_opacity = model._opacity
    base = original_opacity.detach(); raw = torch.sigmoid(base)
    tensor_ids = torch.as_tensor(local, device='cuda')
    photo_risks = []; structural_risks = []; dimensions = []; observations = []
    previous = {v['index']: v for v in current['views']}
    original = {v['index']: v for v in initial['views']}
    q = config['quality']
    try:
        for index in selected_views:
            camera, target, mask, *_ = scene.get_view(index)
            r = regions(mask, config['boundary_relative_width'])
            dark = mask & ((target @ np.array([.2126, .7152, .0722])) < .12)
            rules = flags[index]
            photo_supports = []
            if 'dark_leaf_mse' in rules and dark.any(): photo_supports.append(('dark_photo', dark))
            if 'foreground_mse' in rules or 'foreground_mse_relative' in rules or 'fg_mse' in rules:
                photo_supports.append(('foreground_photo', mask))
            if 'unsupported_local_photo' in rules:
                unsupported=mask&~scene.original_foreground_support[index][0]
                if unsupported.any():photo_supports.append(('excluded_photo',unsupported))
            # Depth/normal changes may expose another valid surface; photo
            # risk is a useful subdivision hint, never a geometric certificate.
            if not photo_supports and any('depth' in key or 'normal' in key for key in rules):
                photo_supports.append(('geometry_photo_hint', mask))
            target_tensor = torch.as_tensor(target.transpose(2, 0, 1).copy(), device='cuda')
            for label, support in photo_supports:
                device_support = torch.as_tensor(support, device='cuda'); estimates = []
                for attenuation in [1., .5]:
                    value = base.clone(); sample = (raw[tensor_ids]*attenuation).clamp(1e-7, 1-1e-7)
                    value[tensor_ids] = torch.logit(sample); model._opacity = value.requires_grad_(True)
                    image = scene.render(camera, model)['render']
                    loss = ((image-target_tensor)**2)[:, device_support].mean()
                    gradient = torch.autograd.grad(loss, model._opacity)[0][tensor_ids]
                    estimate = (-gradient*raw[tensor_ids]/(sample*(1-sample)).clamp_min(1e-8)).flatten().detach().cpu().numpy()
                    if not np.isfinite(estimate).all(): raise ValueError('Nonfinite native photo grouping gradient')
                    estimates.append(estimate); model._opacity = original_opacity
                risk = np.maximum.reduce(estimates); photo_risks.append(risk)
                observations.append(dict(view=camera.image_name, proxy=label, potentially_harmful=int((risk>1e-10).sum())))
            supports = []
            if any('alpha' in key or 'coverage' in key for key in rules):
                with torch.no_grad():
                    before = arrays(scene.render(camera, model)); after = arrays(scene.render(camera, rejected))
                def mass_budget(support, field, tolerance):
                    values = [record.get(field) for record in [original[index], previous[index]]]
                    old = max(values) if all(value is not None for value in values) else None
                    if old is None or not support.any(): return 0.
                    return max(0., (float(before['alpha'][support].mean())-old+tolerance)*support.sum())
                if 'dark_leaf_alpha' in rules and dark.any():
                    supports.append(('dark_alpha', dark, mass_budget(dark, 'dark_foreground_alpha', q['view_foreground_alpha_drop'])))
                if any('foreground_alpha' in key or key == 'fg_alpha' for key in rules):
                    supports.append(('foreground_alpha', mask, mass_budget(mask, 'fg_alpha', q['view_foreground_alpha_drop'])))
                for label, support in [('boundary', r['boundary']), ('thin', r['thin']), ('skeleton', r['skeleton']), ('fg', mask)]:
                    if label+'_coverage' not in rules: continue
                    lost = support & (before['alpha']>.5) & (after['alpha']<=.5)
                    if lost.any():
                        # Keep a margin above the actual binary coverage threshold
                        # on all newly affected protected pixels.
                        supports.append((label+'_holes', lost, float((before['alpha'][lost]-.5).sum())*.5))
                for label, support, budget in supports:
                    estimates = []; device_support = torch.as_tensor(support, device='cuda')
                    for attenuation in [1., .5, .05]:
                        value = base.clone(); sample = (raw[tensor_ids]*attenuation).clamp(1e-7, 1-1e-7)
                        value[tensor_ids] = torch.logit(sample); model._opacity = value.requires_grad_(True)
                        image = native_alpha(camera, model); loss = image[device_support].sum()
                        gradient = torch.autograd.grad(loss, model._opacity)[0][tensor_ids]
                        estimate = (gradient*raw[tensor_ids]/(sample*(1-sample)).clamp_min(1e-8)).flatten().detach().cpu().numpy()
                        if not np.isfinite(estimate).all(): raise ValueError('Nonfinite native alpha grouping gradient')
                        estimates.append(np.maximum(estimate, 0)); model._opacity = original_opacity
                    structural_risks.append(np.maximum.reduce(estimates))
                    dimensions.append(dict(view=camera.image_name, proxy=label, pixels=int(support.sum()), budget=float(budget)))
    finally:
        model._opacity = original_opacity
    allowed = np.maximum.reduce(photo_risks)<=1e-10 if photo_risks else np.ones(len(local), bool)
    risk = np.stack(structural_risks) if structural_risks else np.empty((0, len(local)))
    budgets = np.asarray([d['budget'] for d in dimensions])
    chosen, used = select_with_budgets(benefits, risk, budgets, allowed)
    write_json(folder/'native_gradient_subdivision.json', dict(
        candidates=len(local), photo_nonpositive=int(allowed.sum()), proposed=len(chosen),
        proposed_original_ids=ids[local][chosen], predicted_bg=float(np.asarray(benefits)[chosen].sum()),
        observations=observations, structural_dimensions=dimensions, estimated_used=used,
        certificate=False, quality_thresholds_changed=False,
        scope='Native derivatives only propose a partition; every child requires complete nonlinear quality and measured background improvement'))
    if not len(chosen) or len(chosen)==len(local): return []
    remainder=np.ones(len(local),bool);remainder[chosen]=False
    return [ids[local][chosen].copy(), ids[local][remainder].copy()]
