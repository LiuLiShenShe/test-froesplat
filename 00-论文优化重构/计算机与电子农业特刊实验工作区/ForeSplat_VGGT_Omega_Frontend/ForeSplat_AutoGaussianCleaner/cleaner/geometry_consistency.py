"""External foreground point map and local scale; sparse structures are never pruned for density."""
import numpy as np
from scipy.spatial import cKDTree


def external_surface(scene, indices):
    points = []
    for index in indices:
        _, _, mask, depth, conf = scene.get_view(index)
        if depth is None:
            continue
        valid = mask & np.isfinite(depth) & (depth > 0) & np.isfinite(conf)
        if not valid.any():
            continue
        valid &= conf >= np.quantile(conf[valid], .2)
        y, x = np.nonzero(valid)
        # Deterministic sampling affects only auxiliary distances/free-view proxies.
        stride = max(1, len(x)//3000)
        y, x = y[::stride], x[::stride]
        v = scene.views[index]
        rays = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(v['K']).T
        cam = rays * depth[y, x, None]
        world = (cam-v['E'][:3, 3]) @ v['E'][:3, :3]
        points.append(world)
    if not points:
        return None
    points = np.concatenate(points)
    lo, hi = np.quantile(points, [.001, .999], axis=0)
    tree = cKDTree(points)
    sample = points[::max(1, len(points)//10000)]
    spacing = np.median(tree.query(sample, k=2, workers=8)[0][:, 1])
    return dict(points=points, tree=tree, lo=lo, hi=hi,
        spacing=max(float(spacing), np.linalg.norm(hi-lo)*1e-6), extent=float(np.linalg.norm(hi-lo)))


def add_spatial_features(evidence, model, surface):
    sigma = model.get_scaling_with_3D_filter.detach().cpu().numpy()
    order = np.sort(sigma, axis=1)
    evidence['needle_ratio'] = order[:, 2] / np.maximum(order[:, 1], 1e-12)
    evidence['sheet_ratio'] = order[:, 1] / np.maximum(order[:, 0], 1e-12)
    if surface is None:
        evidence['surface_distance_local'] = np.full(len(sigma), -1.)
    else:
        xyz = model.get_xyz.detach().cpu().numpy()
        distance = surface['tree'].query(xyz, workers=8)[0]
        evidence['surface_distance_local'] = distance / np.maximum(surface['spacing'], order[:, 1])
        from utils.general_utils import build_rotation
        rotation = build_rotation(model._rotation).detach().cpu().numpy()
        short_axis = sigma.argmin(1)
        gaussian_normal = rotation[np.arange(len(sigma)), :, short_axis]
        angles = np.zeros(len(sigma))
        variation = np.zeros(len(sigma))
        for start in range(0, len(xyz), 32768):
            end = min(start+32768, len(xyz))
            _, neighbors = surface['tree'].query(xyz[start:end], k=8, workers=8)
            local = surface['points'][neighbors]
            centered = local-local.mean(1, keepdims=True)
            covariance = np.einsum('nki,nkj->nij', centered, centered)/8
            values, vectors = np.linalg.eigh(covariance)
            variation[start:end] = values[:, 0]/np.maximum(values.sum(1), 1e-15)
            dot = np.sum(gaussian_normal[start:end]*vectors[:, :, 0], axis=1)
            angles[start:end] = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
        evidence['surface_normal_angle_deg'] = angles
        evidence['local_surface_variation'] = variation
    # These are diagnostic features, not sufficient pruning conditions.
