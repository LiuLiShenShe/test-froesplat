"""Compact filtered Gaussian quantities, computed in bounded row chunks."""
import numpy as np
from scipy.special import expit

C0=0.28209479177387814
REQUIRED={'x','y','z','opacity','filter_3D',*[f'scale_{i}' for i in range(3)],
          *[f'rot_{i}' for i in range(4)],*[f'f_dc_{i}' for i in range(3)]}
# 3 xyz, maximum/middle/minimum effective scale, effective alpha, brightness.
WIDTH=8


def validate(rows):
    if not REQUIRED.issubset(rows.dtype.names or ()):
        raise ValueError('Native GGGS fields required: '+str(sorted(REQUIRED-set(rows.dtype.names or ()))))
    for name in rows.dtype.names:
        if not np.isfinite(rows[name]).all(): raise ValueError(f'Nonfinite native field: {name}')
    if np.any(rows['filter_3D']<0): raise ValueError('Negative filter_3D')
    q=np.column_stack([rows[f'rot_{i}'] for i in range(4)]).astype('f8')
    if np.any(np.linalg.norm(q,axis=1)<1e-10): raise ValueError('Zero Gaussian rotation')


def extract_features(rows):
    validate(rows)
    sigma=np.exp(np.column_stack([rows[f'scale_{i}'] for i in range(3)]).astype('f8'))
    effective=np.hypot(sigma,rows['filter_3D'].astype('f8')[:,None])
    alpha=expit(rows['opacity'].astype('f8'))*(sigma/np.maximum(effective,np.finfo('f8').tiny)).prod(1)
    bright=np.column_stack([rows[f'f_dc_{i}'] for i in range(3)]).astype('f8').mean(1)*C0+.5
    if not np.isfinite(effective).all() or np.any(effective<=0) or np.any(effective>np.finfo('f4').max):
        raise ValueError('Effective scale is not representable as finite positive float32')
    return {'xyz':np.column_stack([rows[k] for k in ['x','y','z']]).astype('f4'),
            'effective_max':effective.max(1).astype('f4'),
            'effective_median':np.median(effective,axis=1).astype('f4'),
            'effective_min':effective.min(1).astype('f4'),
            'alpha_eff':alpha.astype('f4'),'brightness':bright.astype('f4')}


def pack(rows):
    f=extract_features(rows)
    return np.column_stack([f['xyz'],f['effective_max'],f['effective_median'],f['effective_min'],f['alpha_eff'],f['brightness']])


def unpack(a):
    return dict(xyz=a[:,:3],effective_max=a[:,3],effective_median=a[:,4],effective_min=a[:,5],alpha_eff=a[:,6],brightness=a[:,7])
