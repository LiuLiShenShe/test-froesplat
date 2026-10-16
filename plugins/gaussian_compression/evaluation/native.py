"""Read a PLY through the untouched local 2DGS inference model."""
from pathlib import Path
import sys
import numpy as np
from ..io import read_model


def load_native(path, gs_root, active_sh_degree=None):
    gs_root=Path(gs_root).resolve(strict=True)
    for name in ['scene','gaussian_renderer','utils']:
        existing=sys.modules.get(name)
        if existing is not None and getattr(existing,'__file__',None):
            origin=Path(existing.__file__).resolve()
            if gs_root not in origin.parents:
                raise RuntimeError(f'{name} already imported from a different renderer: {origin}')
    for sub in ['submodules/simple-knn','submodules/diff-surfel-rasterization','']:
        sys.path.insert(0,str(gs_root/sub))
    import torch
    from scene.gaussian_model import GaussianModel
    ply=read_model(path)
    model=GaussianModel(ply.sh_degree)
    model.load_ply(str(path))
    expected={
        '_xyz':np.column_stack([ply.vertices[k] for k in ['x','y','z']]),
        '_opacity':ply.vertices['opacity'][:,None],
        '_scaling':np.column_stack([ply.vertices[f'scale_{i}'] for i in range(2)]),
        '_rotation':np.column_stack([ply.vertices[f'rot_{i}'] for i in range(4)]),
        '_features_dc':np.column_stack([ply.vertices[f'f_dc_{i}'] for i in range(3)])[:,None,:],
    }
    nrest=3*((ply.sh_degree+1)**2-1)
    rest=np.column_stack([ply.vertices[f'f_rest_{i}'] for i in range(nrest)]) if nrest else np.empty((ply.count,0),dtype=np.float32)
    expected['_features_rest']=rest.reshape(ply.count,3,-1).transpose(0,2,1)
    for key,array in expected.items():
        actual=getattr(model,key).detach().cpu().numpy()
        if actual.shape != array.shape or not np.array_equal(actual,array.astype(np.float32)):
            raise RuntimeError(f'Native reload changed {key}')
    for value in [model.get_xyz,model.get_scaling,model.get_rotation,model.get_opacity,model.get_features]:
        if not torch.isfinite(value).all():
            raise RuntimeError('Native activated parameters contain nonfinite values')
    if active_sh_degree is not None:
        if not 0<=active_sh_degree<=model.max_sh_degree:
            raise ValueError('Invalid active SH degree')
        model.active_sh_degree=active_sh_degree
    return model
