"""Lossless native row editing and parameter-equivalent viewer export."""
import hashlib
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray)
        else x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    tmp.replace(path)


def read_ply(path):
    rows = PlyData.read(str(path))['vertex'].data.copy()
    required = ['x', 'y', 'z', 'opacity', 'filter_3D'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    if not all(k in rows.dtype.names for k in required):
        raise ValueError('Input must be native GGGS PLY, including filter_3D and 3 scales; 2DGS/point maps are unsupported')
    if not len(rows):
        raise ValueError('Empty Gaussian model')
    for k in rows.dtype.names:
        if not np.isfinite(rows[k]).all():
            raise ValueError(f'Nonfinite input field: {k}')
    if np.any(rows['filter_3D'] < 0):
        raise ValueError('Negative 3D filter')
    q = np.column_stack([rows[f'rot_{i}'] for i in range(4)])
    if np.any(np.linalg.norm(q, axis=1) < 1e-10):
        raise ValueError('Zero Gaussian rotation')
    return rows


def save_ply(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.ply')
    PlyData([PlyElement.describe(rows.copy(), 'vertex')], text=False, byte_order='<').write(str(tmp))
    tmp.replace(path)


def bake_viewer(rows):
    # GGGS SG lobes have no standard SuperSplat representation. Never silently discard them.
    if any(k.startswith('sg_color_') for k in rows.dtype.names):
        raise ValueError('Nonzero SG degree is unsupported by equivalent SuperSplat export; use SH-only GGGS')
    out = rows.copy()
    sigma = np.exp(np.column_stack([rows[f'scale_{k}'] for k in range(3)]).astype(np.float64))
    filtered = np.hypot(sigma, rows['filter_3D'].astype(np.float64)[:, None])
    from scipy.special import expit, logit
    alpha = expit(rows['opacity'].astype(np.float64)) * (sigma / filtered).prod(1)
    alpha = np.clip(alpha, np.finfo(np.float32).tiny, 1 - 1e-7)
    for k in range(3):
        out[f'scale_{k}'] = np.log(filtered[:, k])
    out['opacity'] = logit(alpha)
    out['filter_3D'] = 0
    actual = np.exp(np.column_stack([out[f'scale_{k}'] for k in range(3)]).astype(np.float64))
    error = float(np.max(np.abs(actual - filtered))) if len(rows) else 0.
    return out, {'effective_scale_max_abs_error': error,
        'effective_opacity_max_abs_error': float(np.max(np.abs(expit(out['opacity']) - alpha))) if len(rows) else 0.,
        'formula': 'sigma_eff=hypot(sigma,filter_3D); alpha_eff=sigmoid(opacity)*prod(sigma/sigma_eff)',
        'xyz_rotation_sh_unchanged': True}
