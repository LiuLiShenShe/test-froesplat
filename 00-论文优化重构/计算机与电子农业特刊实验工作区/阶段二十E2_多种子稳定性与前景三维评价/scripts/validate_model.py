from __future__ import annotations
import argparse
import json
from argparse import Namespace
from pathlib import Path
from common import *
from protocol import tensor_sha, assert_gpu


def validate(sample,group,seed,attempt):
    import_paths()
    import numpy as np
    import torch
    from plyfile import PlyData
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    assert_gpu()
    label=run_id(sample,group,seed,attempt)
    runtime=read_json(STAGE/f'reports/{label}_RUNTIME.json')
    assert runtime['exit_code']==0 and runtime['observed_iteration']==30000 and runtime['optimizer_steps']==29999
    output=Path(runtime['output_path']); config=read_json(output/'FULL_FROZEN_TRAIN_CONFIG.json')
    checkpoint=output/'chkpnt30000.pth'; ply=output/'point_cloud/iteration_30000/point_cloud.ply'
    vertex=PlyData.read(ply)['vertex'].data
    assert len(vertex)>0 and all(np.isfinite(vertex[k]).all() for k in vertex.dtype.names)
    captured,iteration=torch.load(checkpoint,map_location='cuda',weights_only=False)
    assert iteration==30000
    model=GaussianModel(config['model']['sh_degree'])
    scene=Scene(Namespace(**config['model']),model,load_iteration=30000,shuffle=False)
    fields=('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity')
    for index,field in enumerate(fields,1):
        assert torch.isfinite(getattr(model,field)).all()
        assert torch.equal(getattr(model,field),captured[index]),field
    assert (model.get_scaling>0).all()
    assert ((model.get_opacity>=0)&(model.get_opacity<=1)).all()
    restored=GaussianModel(config['model']['sh_degree']); restored.restore(captured,Namespace(**config['optimization']))
    expected={r['loader_image_name'] for r in read_csv(E0/'reports/EVALUATION_REFERENCE_MANIFEST.csv') if r['sample']==sample}
    assert {c.image_name for c in scene.getTestCameras()}==expected
    camera=scene.getTestCameras()[0]
    with torch.no_grad():
        bg=torch.zeros(3,device='cuda'); pipe=Namespace(**config['pipeline'])
        first=render(camera,model,pipe,bg)['render']; second=render(camera,restored,pipe,bg)['render']
        reload_error=(first-second).abs().max().item()
    assert reload_error<=1e-6,reload_error
    result={'sample':sample,'group':group,'seed':seed,'attempt':attempt,'status':'PASS','iteration':iteration,
            'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
            'PLY_path':str(ply),'PLY_sha256':sha256(ply),'PLY_bytes':ply.stat().st_size,
            'Gaussian_count':len(vertex),'parameter_sha256':{k:tensor_sha(getattr(model,k)) for k in fields},
            'reload_render_max_abs_error':reload_error,'expected_heldout':len(expected),
            'activated_parameters_finite':bool(all(torch.isfinite(t).all() for t in (model.get_xyz,model.get_scaling,model.get_opacity,model.get_rotation,model.get_features))),
            'GPU_UUID':GPU_UUID,'timestamp':now()}
    write_json(STAGE/f'reports/{label}_VALIDATION.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--sample',required=True);p.add_argument('--group',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args(); validate(a.sample,a.group,a.seed,a.attempt)
