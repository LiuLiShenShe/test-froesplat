"""Faithful V2/V3 rules and scene-relative GGGS variants.

DC is an appearance prior, NOT the view-dependent SH pixel brightness.
Raw historical variants use exp(log-scale), sigmoid(logit-opacity), all three
GGGS axes. Adaptive variants use native filtered scales/effective opacity.
Every native row retains XYZ, quaternion, full SH and filter_3D unchanged.
"""
import numpy as np
from scipy.special import expit
from scipy.spatial import cKDTree


def features(rows, knn=True):
    xyz = np.column_stack([rows[k] for k in ['x', 'y', 'z']]).astype(np.float64)
    sigma = np.exp(np.column_stack([rows[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    effective = np.hypot(sigma, rows['filter_3D'][:, None])
    rgb = np.column_stack([rows[f'f_dc_{i}'] for i in range(3)]) * .28209479177387814 + .5
    bright = rgb.mean(1)
    alpha = expit(rows['opacity'].astype(np.float64))
    f = dict(xyz=xyz, sigma=sigma, effective=effective, brightness=bright,
             color_range=np.ptp(rgb, axis=1), raw_alpha=alpha,
             effective_alpha=alpha*(sigma/effective).prod(1))
    extent = max(float(np.linalg.norm(np.diff(np.quantile(xyz, [.01, .99], axis=0), axis=0))), 1e-12)
    f['extent'] = extent
    f['scale_max'] = sigma.max(1)
    f['effective_max'] = effective.max(1)
    f['knn_original'] = np.zeros(len(rows), bool)
    f['knn_normalized'] = np.zeros(len(rows), bool)
    if knn and len(rows) > 21:
        tree = cKDTree(xyz)
        # Chunked queries avoid N*K temporary arrays on multi-million models.
        spacing = np.empty(len(rows), np.float64)
        for start in range(0, len(rows), 32768):
            end = min(start+32768, len(rows))
            d, ni = tree.query(xyz[start:end], k=5, workers=8)
            # Explicitly exclude identity; duplicate centers need not put self first.
            d[ni == np.arange(start, end)[:, None]] = np.inf
            spacing[start:end] = np.median(np.sort(d, axis=1)[:, :4], axis=1)
        spacing = np.maximum(spacing, extent*1e-8)
        f['spacing'] = spacing
        f['local_size_ratio'] = effective.max(1)/np.maximum(spacing, np.median(spacing))
        candidates = np.flatnonzero(bright < .35)
        for start in range(0, len(candidates), 16384):
            ids = candidates[start:start+16384]
            d, ni = tree.query(xyz[ids], k=min(65, len(rows)), workers=8)
            d[ni == ids[:, None]] = np.inf
            order = np.argsort(d, axis=1)
            d, ni = np.take_along_axis(d, order, 1), np.take_along_axis(ni, order, 1)
            f['knn_original'][ids] = bright[ids] < .4*np.median(bright[ni[:, :20]], axis=1)
            radius = 4*np.maximum(spacing[ids], np.sort(effective[ids], axis=1)[:, 1])
            valid = d[:, :20] <= radius[:, None]
            values = np.where(valid, bright[ni[:, :20]], np.nan)
            enough = valid.sum(1) >= 5
            med = np.zeros(len(ids))
            med[enough] = np.nanmedian(values[enough], axis=1)
            f['knn_normalized'][ids] = enough & (bright[ids] < .4*med)
    else:
        f['spacing'] = np.full(len(rows), extent/max(len(rows), 1)**(1/3))
        f['local_size_ratio'] = effective.max(1)/f['spacing']
    return f


def rules(f, version=3, normalized=False):
    s = f['effective_max'] if normalized else f['scale_max']
    alpha = f['effective_alpha'] if normalized else f['raw_alpha']
    b = f['brightness']
    scale_limit = float(np.quantile(s, .995)) if normalized else .5
    dark_scale = float(np.quantile(s, .9)) if normalized else .05
    distance = np.linalg.norm(f['xyz']-np.median(f['xyz'], axis=0), axis=1)
    masks = dict(low_opacity=alpha < .01, oversized=s > scale_limit,
                 dark=b < .10, dark_large=(b < .20) & (s > dark_scale),
                 dark_desaturated=(b < .25) & (f['color_range'] < .06),
                 spatial_outlier=distance > distance.mean()+3*distance.std())
    if version == 3:
        relative = f['local_size_ratio'] if normalized else s
        median = max(float(np.median(relative)), 1e-12)
        masks['scale_brightness'] = np.zeros(len(b), bool)
        for mult, floor in [(1., .15), (1.5, .20), (2.5, .28)]:
            masks['scale_brightness'] |= (relative > median*mult) & (b < floor)
        masks['knn_dark'] = f['knn_normalized' if normalized else 'knn_original']
    removed = np.logical_or.reduce(list(masks.values()))
    bits = np.zeros(len(b), np.uint16)
    assigned = np.zeros(len(b), bool)
    stats = {}
    for i, (name, mask) in enumerate(masks.items()):
        bits |= mask.astype(np.uint16) << i
        stats[name] = dict(overlapping_count=int(mask.sum()), exclusive_first_reason_count=int((mask & ~assigned).sum()))
        assigned |= mask
    return removed, bits, dict(version=version, normalized=normalized,
        thresholds=dict(opacity=.01, brightness=.10, scale_max=scale_limit, dark_scale=dark_scale,
                        knn_k=20, knn_ratio=.4, knn_brightness_max=.35), reasons=stats,
        scale_encoding='exp(log scale); 3 axes; isotropic filter adds f^2 to covariance eigenvalues',
        alpha_encoding='sigmoid(logit); normalized variant multiplies native determinant filter coefficient',
        color_definition='Unclipped mean(SH_DC*C0+0.5), not true rendered brightness',
        removed=int(removed.sum()), scene_extent=f['extent'])
