"""Retain the actual1K passive native capture and export it with official save_ply."""
import os,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'scripts'))
import qgs_common as q
os.environ['CUDA_VISIBLE_DEVICES']=''
from scene.gaussian_model import GaussianModel
from omegaconf import OmegaConf
from plyfile import PlyData

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    audit=json.loads((ROOT/'recovery/rolling_checkpoint_validation_1000.json').read_text())
    assert audit['observed_iteration']==1000
    target=Path('/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/audited_snapshots/iteration_1000')
    target.mkdir(parents=True,exist_ok=True)
    alias=ROOT/'retained_native_states'
    if not alias.exists():alias.symlink_to(target.parent,target_is_directory=True)
    assert alias.resolve()==target.parent
    checkpoint=target/'native_state.pth'
    if not checkpoint.exists():os.link(ROOT/'recovery/latest_native_state.pth',checkpoint)
    assert sha(checkpoint)==audit['observed_sha256'],'Retained inode is not the actually audited1K capture'
    saved,iteration=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert iteration==1000 and len(saved[1])==591494
    cfg=OmegaConf.load(ROOT/'actual_config.yaml');model=GaussianModel(cfg.gs_model)
    model.active_sh_degree=saved[0]
    for name,index in [('_xyz',1),('_features_dc',2),('_features_rest',3),('_scaling',4),('_rotation',5),('_opacity',6)]:
        setattr(model,name,torch.nn.Parameter(saved[index]))
    ply=target/'native_quadratic.ply';model.save_ply(str(ply))
    data=PlyData.read(ply)['vertex'].data
    fields=[('xyz',['x','y','z'],saved[1].detach().numpy()),
        ('f_dc',[f'f_dc_{i}' for i in range(3)],saved[2].detach().transpose(1,2).flatten(1).numpy()),
        ('f_rest',[f'f_rest_{i}' for i in range(45)],saved[3].detach().transpose(1,2).flatten(1).numpy()),
        ('scaling',[f'scale_{i}' for i in range(6)],saved[4].detach().numpy()),
        ('rotation',[f'rot_{i}' for i in range(4)],saved[5].detach().numpy()),
        ('opacity',['opacity'],saved[6].detach().numpy())]
    for name,columns,expected in fields:
        for i,column in enumerate(columns):
            assert np.array_equal(np.asarray(data[column]).view(np.uint32),np.ascontiguousarray(expected[:,i]).view(np.uint32)),name
    assert not torch.cuda.is_initialized(),'CPU-only native export unexpectedly initialized CUDA'
    result=dict(status='actual_partial_native_export_bitwise_verified',iteration=1000,phase=audit['phase'],native_ply=str(ply.resolve()),
        source_checkpoint=str(checkpoint.resolve()),source_checkpoint_SHA256=audit['observed_sha256'],native_ply_SHA256=sha(ply),
        primitives=591494,native_signed_scale_channels=6,native_preset='Mip',active_sh_degree=int(saved[0]),nonfinite_native_primitives=audit['nonfinite_native_primitives'],
        all_native_PLY_fields_bitwise_equal_to_actual_checkpoint=True,official_save_ply_used=True,CPU_only=True,
        retention='Hard link to the actual atomically published rolling inode; later os.replace creates a different rolling inode. No model bytes copied.',
        model_stage='Retained passive1K capture before optimizer/densification; not a completed formal30K model',
        config_source=str((ROOT/'actual_config.yaml').resolve()))
    (target/'NATIVE_EXPORT_PROVENANCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
