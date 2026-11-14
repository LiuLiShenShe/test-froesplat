"""Bounded anisotropic covariance optimization of automatically identified mixed Gaussians.

Effective (filtered) opacity is constant by default. Adaptive callers may
enable a bounded per-Gaussian opacity reduction for the identified local
mixed group; all unselected rows stay exact. XYZ and SH stay unchanged.
Every checkpoint is subject to full multi-view
photometry, boundary, skeleton, depth and normal constraints by the caller.
"""
import numpy as np
import torch
from .foreground_protection import regions


def refine(scene, model, ids, indices, config, checkpoint, auxiliary_loss=None,
           optimization_indices=None, observed_view_loss=None):
    ids = np.asarray(ids, np.int64)
    if not len(ids):
        return []
    index = torch.as_tensor(ids, device='cuda')
    base_scale = model._scaling.detach().clone()
    base_rotation = model._rotation.detach().clone()
    base_opacity = model._opacity.detach().clone()
    effective_opacity = model.get_opacity_with_3D_filter.detach().clone()
    delta = torch.nn.Parameter(torch.zeros((len(ids), 3), device='cuda'))
    rotation_delta = torch.nn.Parameter(torch.zeros((len(ids), 4), device='cuda'))
    opacity_delta = torch.nn.Parameter(torch.zeros((len(ids), 1), device='cuda'))
    opacity_trust = config.get('refine_opacity_trust',0.)
    parameters = [delta,rotation_delta]+([opacity_delta] if opacity_trust>0 else [])
    optimizer = torch.optim.Adam(parameters, lr=config['refine_lr'])
    # Optimization cycles through a deterministic, diverse subset; acceptance uses ALL input cameras.
    opt_indices = indices[::max(1, len(indices)//16)] if optimization_indices is None else list(optimization_indices)
    targets = {}
    for vi in opt_indices:
        c, target, mask, depth, conf = scene.get_view(vi)
        with torch.no_grad():
            out = scene.render(c, model)
            targets[vi] = {key: value.detach().cpu() for key, value in out.items() if key in ['mask', 'median_depth', 'normal']}
    history = []
    try:
        for step in range(config['refine_steps']):
            optimizer.zero_grad(set_to_none=True)
            model._scaling = base_scale.index_copy(0, index, base_scale[index] + delta)
            model._rotation = base_rotation.index_copy(0, index, base_rotation[index] + rotation_delta)
            sigma = torch.exp(model._scaling)
            coef = (sigma/torch.hypot(sigma, model.filter_3D)).prod(1, keepdim=True)
            desired_opacity=effective_opacity
            if opacity_trust>0:
                desired_opacity=effective_opacity.index_copy(0,index,effective_opacity[index]*torch.exp(opacity_delta))
            raw_opacity = desired_opacity/coef.clamp_min(1e-12)
            # Scaling candidates that cannot preserve effective alpha are penalized and not accepted.
            saturation = torch.relu(raw_opacity[index] - .999999).square().mean()
            model._opacity = torch.logit(raw_opacity.clamp(1e-8, .999999))
            vi = opt_indices[step % len(opt_indices)]
            c, target, mask, depth, conf = scene.get_view(vi)
            r = regions(mask, config['boundary_relative_width'])
            out = scene.render(c, model)
            alpha, rgb = out['mask'].squeeze(), out['render'].permute(1, 2, 0)
            device_mask = torch.as_tensor(mask, device='cuda')
            coverage_mask=device_mask
            sensitive_mask=r['boundary']|r['thin']
            geometry_mask=mask
            if config.get('foreground_support_mode')=='photo_relative_or_depth':
                support,geometry_mask=scene.original_foreground_support[vi]
                coverage_mask=torch.as_tensor(support,device='cuda')
                guarded=[]
                for region in [r['boundary'],r['thin']]:
                    subset=region&support
                    guarded.append(subset if subset.sum()>=config['foreground_support_minimum_fraction']*region.sum() else region)
                sensitive_mask=guarded[0]|guarded[1]
                if geometry_mask.sum()<config['foreground_geometry_minimum_fraction']*mask.sum():geometry_mask=support
            target_t = torch.as_tensor(target, device='cuda')
            fg_count = max(int(mask.sum()), 1)
            bg = torch.as_tensor(r['bg'], device='cuda')
            background = alpha[bg].sum()/fg_count
            photo = ((rgb[device_mask]-target_t[device_mask])**2).mean()
            ref_alpha = targets[vi]['mask'].cuda().squeeze()
            coverage = torch.relu(ref_alpha[coverage_mask]-alpha[coverage_mask]).square().mean() if coverage_mask.any() else photo*0
            sensitive = torch.as_tensor(sensitive_mask, device='cuda')
            boundary = torch.relu(ref_alpha[sensitive]-alpha[sensitive]).square().mean() if sensitive.any() else coverage * 0
            ref_depth = targets[vi]['median_depth'].cuda().squeeze()
            valid = torch.as_tensor(geometry_mask,device='cuda') & (ref_depth > 0) & (ref_alpha > .5)
            unit = ref_depth[valid].median().clamp_min(1e-8) if valid.any() else ref_depth.new_tensor(1.)
            geometric = (out['median_depth'].squeeze()[valid]-ref_depth[valid]).abs().div(unit).mean() if valid.any() else coverage * 0
            if config['geometry'] and depth is not None:
                valid_external = geometry_mask & np.isfinite(depth) & (depth > 0) & np.isfinite(conf)
                if valid_external.any():
                    valid_external &= conf >= np.quantile(conf[valid_external], .2)
                    vg = torch.as_tensor(valid_external, device='cuda')
                    d = torch.as_tensor(depth, device='cuda')
                    # Robust external depth term: uncertain disagreement has bounded influence.
                    external = ((out['median_depth'].squeeze()[vg]-d[vg]).abs()/unit).clamp(max=.1).mean()
                else:
                    external = coverage * 0
            else:
                external = coverage * 0
            ref_normal = targets[vi]['normal'].cuda()
            nv = valid & (torch.linalg.vector_norm(ref_normal, dim=0) > .1)
            normal = (1-torch.nn.functional.cosine_similarity(out['normal'][:, nv], ref_normal[:, nv], dim=0)).mean() if nv.any() else coverage * 0
            regularizer = delta.square().mean() + rotation_delta.square().mean()
            if opacity_trust>0:regularizer=regularizer+opacity_delta.square().mean()
            loss = background + 12*photo + 30*coverage + 60*boundary + 2*geometric + .2*external + normal + .02*regularizer + 100*saturation
            # Optional GGGS free-camera objective supplied by the adaptive
            # cleaner. Legacy callers retain the exact observed-view loss.
            if auxiliary_loss is not None:
                loss = loss + auxiliary_loss(scene, model, step)
            if observed_view_loss is not None:
                loss = loss + observed_view_loss(vi, out, target, mask, r)
            if not torch.isfinite(loss):
                history.append(dict(step=step+1, status='nonfinite_loss_rollback'))
                break
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                history.append(dict(step=step+1, status='nonfinite_gradient_rollback'))
                break
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            with torch.no_grad():
                delta.clamp_(-config['refine_scale_trust'], config['refine_scale_trust'])
                rotation_delta.clamp_(-config['refine_rotation_trust'], config['refine_rotation_trust'])
                if opacity_trust>0:opacity_delta.clamp_(-opacity_trust,0.)
            if (step+1) % config['refine_validation_interval'] == 0 or step+1 == config['refine_steps']:
                # Materialize AFTER the last optimizer step and check exact effective-alpha feasibility.
                with torch.no_grad():
                    model._scaling = base_scale.index_copy(0, index, base_scale[index]+delta)
                    model._rotation = base_rotation.index_copy(0, index, base_rotation[index]+rotation_delta)
                    sigma = torch.exp(model._scaling)
                    coef = (sigma/torch.hypot(sigma, model.filter_3D)).prod(1, keepdim=True)
                    desired_opacity=effective_opacity
                    if opacity_trust>0:
                        desired_opacity=effective_opacity.index_copy(0,index,effective_opacity[index]*torch.exp(opacity_delta))
                    raw = desired_opacity/coef.clamp_min(1e-12)
                    feasible_rows = raw[index, 0] < .999999
                    # One saturated Gaussian cannot block the independent feasible covariance edits.
                    # Restore only infeasible scale/rotation/opacity rows; whole-candidate quality still applies.
                    bad = index[~feasible_rows]
                    model._scaling[bad] = base_scale[bad]
                    if not config.get('allow_rotation_only_fallback',False):
                        model._rotation[bad] = base_rotation[bad]
                    model._opacity = torch.logit(raw.clamp(1e-8, .999999))
                    model._opacity[bad] = base_opacity[bad]
                    rotation_only = bool((model._rotation[bad]!=base_rotation[bad]).any()) if len(bad) else False
                    feasible = bool(feasible_rows.any()) or rotation_only
                    rows = {'scale': model._scaling[index].cpu().numpy(), 'rotation': model._rotation[index].cpu().numpy(),
                        'opacity': model._opacity[index].cpu().numpy()}
                    candidate_opacity_ratio=(model.get_opacity_with_3D_filter[index]/effective_opacity[index].clamp_min(1e-12)).detach()
                result = checkpoint(step+1, ids, rows) if feasible else {'accepted': False, 'reasons': ['effective alpha infeasible']}
                history.append(dict(step=step+1, loss=float(loss.detach()),
                    local_opacity_trust=opacity_trust,
                    requested_minimum_effective_opacity_ratio=float(torch.exp(opacity_delta).min().detach()) if opacity_trust>0 else 1.,
                    candidate_minimum_effective_opacity_ratio=float(candidate_opacity_ratio.min()),
                    trust_scope='Per local call relative to the current selected group; repeated calls may compound. Full-scene quality always refers to the untouched original.', **result))
                print('refinement checkpoint', step+1, result, flush=True)
    finally:
        model._scaling, model._rotation, model._opacity = base_scale, base_rotation, base_opacity
    return history
