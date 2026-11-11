"""Exact occlusion-aware alpha contribution, remeasured on every current model."""
import math
import numpy as np
import torch
from .foreground_protection import regions


def contribution(camera, model, weights):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    settings = GaussianRasterizationSettings(image_height=camera.image_height, image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx/2), tanfovy=math.tan(camera.FoVy/2), kernel_size=0.,
        bg=torch.zeros(3, device='cuda'), scale_modifier=1., viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform, sh_degree=model.active_sh_degree, sg_degree=model.active_sg_degree,
        campos=camera.camera_center, prefiltered=False, require_depth=True, debug=False)
    colors = torch.ones((len(model.get_xyz), 3), device='cuda', requires_grad=True)
    scales, opacity = model.get_scaling_n_opacity_with_3D_filter
    out = GaussianRasterizer(raster_settings=settings)(means3D=model.get_xyz.detach(),
        means2D=torch.zeros_like(model.get_xyz), shs=None, sg_axis=model.get_sg_axis.detach(),
        sg_sharpness=model.get_sg_sharpness.detach(), sg_color=model.get_sg_color.detach(), colors_precomp=colors,
        opacities=opacity.detach(), scales=scales.detach(), rotations=model.get_rotation.detach(), cov3Ds_precomp=None)
    weights = np.asarray(weights, np.float32)
    grad = torch.autograd.grad(out[0], colors, grad_outputs=torch.as_tensor(weights, device='cuda'))[0].detach().cpu().numpy()
    alpha = out[3].detach().cpu().numpy().squeeze()
    expected = (weights * alpha[None]).sum((1, 2), dtype=np.float64)
    actual = grad.sum(0, dtype=np.float64)
    error = np.abs(actual-expected)/np.maximum(expected, 1.)
    if error.max() >= .002:
        raise RuntimeError(f'Raster contribution/alpha conservation failed: {error}')
    return grad, out[1].detach().cpu().numpy(), error


def analyze(scene, model, indices, config):
    n = len(model.get_xyz)
    names = ['fg', 'bg', 'boundary', 'thin', 'geometry_mass', 'depth_sum', 'depth_sq_sum', 'geometry_bad', 'high_bg']
    evidence = {k: np.zeros(n, np.float64) for k in names}
    for k in ['visible_views', 'fg_views', 'bg_views', 'geometry_views', 'occlusion_views']:
        evidence[k] = np.zeros(n, np.int32)
    evidence['max_radius_fraction'] = np.zeros(n)
    evidence['depth_error_sum'] = np.zeros(n)
    records = []
    xyz = model.get_xyz.detach().cpu().numpy()
    for index in indices:
        c, rgb, mask, depth, conf = scene.get_view(index)
        r = regions(mask, config['boundary_relative_width'])
        denominator = max(mask.sum(), 1)
        g, radii, err = contribution(c, model, np.stack([r['fg'], r['bg'], r['boundary']]))
        for j, key in enumerate(['fg', 'bg', 'boundary']):
            evidence[key] += g[:, j] / denominator
        # A per-view visibility floor expressed as fraction of the foreground image area.
        visibility_floor = denominator * 1e-7
        evidence['visible_views'] += (g[:, :2].sum(1) > visibility_floor)
        evidence['fg_views'] += (g[:, 0] > visibility_floor)
        evidence['bg_views'] += (g[:, 1] > visibility_floor)
        evidence['max_radius_fraction'] = np.maximum(evidence['max_radius_fraction'], radii / np.sqrt(mask.size))
        valid = np.zeros_like(mask)
        zn = np.zeros_like(mask, dtype=np.float32)
        unit = 1.
        if config['geometry'] and depth is not None:
            valid = mask & np.isfinite(depth) & (depth > 0) & np.isfinite(conf)
            if valid.any():
                valid &= conf >= np.quantile(conf[valid], .2)
                unit = max(float(np.median(depth[valid])), 1e-9)
                zn[valid] = depth[valid] / unit
        g2, _, err2 = contribution(c, model, np.stack([r['thin'], valid, zn]))
        for j, key in enumerate(['thin', 'geometry_mass', 'depth_sum']):
            evidence[key] += g2[:, j] / denominator
        with torch.no_grad():
            out = scene.render(c, model)
            alpha = out['mask'].detach().cpu().numpy().squeeze()
            rendered_depth = out['median_depth'].detach().cpu().numpy().squeeze()
        bad = valid & (alpha > .5) & (np.abs(rendered_depth - (depth if depth is not None else 0)) / unit > .05)
        g3, _, err3 = contribution(c, model, np.stack([zn * zn, bad, r['bg'] & (alpha > .5)]))
        for j, key in enumerate(['depth_sq_sum', 'geometry_bad', 'high_bg']):
            evidence[key] += g3[:, j] / denominator
        # Per-Gaussian weighted depth moments use actual foreground footprint support.
        # Occluded points behind the external visible surface are uncertain, never automatically invalid.
        mass = g2[:, 1]
        mean = np.divide(g2[:, 2], mass, out=np.zeros(n), where=mass > 1e-12)
        sq = np.divide(g3[:, 0], mass, out=np.zeros(n), where=mass > 1e-12)
        spread = np.sqrt(np.maximum(0., sq-mean*mean))
        z = (xyz @ scene.views[index]['E'][:3, :3].T + scene.views[index]['E'][:3, 3])[:, 2] / unit
        sigma = model.get_scaling_with_3D_filter.detach().cpu().numpy().max(1) / unit
        tolerance = np.maximum(.03, 3 * spread + 2 * sigma)
        observed = mass > visibility_floor
        consistent = observed & (np.abs(z-mean) <= tolerance)
        occluded = observed & (z > mean+tolerance)
        evidence['geometry_views'] += consistent
        evidence['occlusion_views'] += occluded
        evidence['depth_error_sum'] += np.where(observed, np.abs(z-mean), 0.)
        records.append(dict(index=index, name=c.image_name, sum_error=float(max(err.max(), err2.max(), err3.max())),
            fg_pixels=int(mask.sum()), geometry_pixels=int(valid.sum()), depth_unit=unit))
        if len(records) % 8 == 0 or len(records) == len(indices):
            print(f'contribution {len(records)}/{len(indices)}', flush=True)
    for k in names:
        evidence[k] /= len(indices)
    evidence['records'] = records
    return evidence
