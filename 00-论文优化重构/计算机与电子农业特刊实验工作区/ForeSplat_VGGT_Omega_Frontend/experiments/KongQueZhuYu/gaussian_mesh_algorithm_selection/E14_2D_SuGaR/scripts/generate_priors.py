"""Run official Metric3Dv2 prior with frozen COLMAP intrinsics/RGB, recording its extra evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import sys,json,time,hashlib,importlib.util
from pathlib import Path
import torch,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1]
OFF=ROOT/'official'
DATA=ROOT/'input_colmap'
sys.path.insert(0,str(ROOT/'metric3d_official'))
sys.path.insert(0,str(OFF))
spec=importlib.util.spec_from_file_location('sugar_priors',OFF/'estimate_depth_normal_priors.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    started=time.time();torch.manual_seed(0);np.random.seed(0)
    weight=ROOT/'weights/metric_depth_vit_giant2_800k.pth'
    h=hashlib.sha256()
    with weight.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    assert h.hexdigest()=='22d64c4dec098fa56eb17b6f22b1fb7d16b4742c66a9c8ee73e7dae61c0d2a1b',h.hexdigest()
    model=torch.hub.load(str(ROOT/'metric3d_official'),'metric3d_vit_giant2',source='local',pretrain=False)
    state=torch.load(weight,map_location='cpu',weights_only=False)['model_state_dict']
    load=model.load_state_dict(state,strict=False);del state
    unused_keys={'depth_model.encoder.mask_token','depth_model.encoder.norm.weight','depth_model.encoder.norm.bias'}
    assert model.depth_model.encoder.multi_output is True
    assert set(load.missing_keys)<=unused_keys and not load.unexpected_keys,load
    write(ROOT/'prior_weight_loading.json',dict(status='loaded',missing_unused_keys=load.missing_keys,unexpected_keys=load.unexpected_keys,all_inference_weights_loaded=True,reason='Official vit_giant2_reg multi_output=True bypasses final encoder.norm; default tensor inference uses masks=None so mask_token is unused; same native strictFalse behavior'))
    print('Metric3D loading',load,flush=True)
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    spec=importlib.util.spec_from_file_location('colmap_loader',OFF/'gaussian_splatting_2d/scene/colmap_loader.py')
    colmap=importlib.util.module_from_spec(spec);spec.loader.exec_module(colmap)
    cs=colmap.read_intrinsics_binary(DATA/'sparse/0/cameras.bin')
    ims=colmap.read_extrinsics_binary(DATA/'sparse/0/images.bin')
    assert len(ims)==208
    for name in ['estimated_dense_depth','estimated_dense_normal']:(DATA/name).mkdir(exist_ok=True)
    records=[]
    for image in sorted(ims.values(),key=lambda x:x.name):
        stem=Path(image.name).stem
        dep=DATA/'estimated_dense_depth'/f'{stem}.npy';normal=DATA/'estimated_dense_normal'/f'{stem}.npy'
        cam=cs[image.camera_id];assert cam.model=='PINHOLE'
        fx,fy,cx,cy=map(float,cam.params)
        rgb=cv2.imread(str(DATA/'images'/image.name))[:,:,::-1].copy()
        if not (dep.exists() and normal.exists()):
            depth,n=prior.run_metric3d(rgb,model,[fx,fy,cx,cy])
            assert n is not None
            np.save(dep,depth.cpu().numpy().astype(np.float32))
            np.save(normal,n.cpu().numpy().astype(np.float32))
        d=np.load(dep,mmap_mode='r')
        nominal=min(616/cam.height,1064/cam.width);rh,rw=int(cam.height*nominal),int(cam.width*nominal)
        sx,sy=rw/cam.width,rh/cam.height;pad_y=(616-rh)//2;pad_x=(1064-rw)//2
        rec=dict(prior_resized_size=[rw,rh],prior_nominal_resize=nominal,prior_actual_resize=[sx,sy],prior_resized_intrinsic=[fx*sx,fy*sy,(cx+.5)*sx-.5,(cy+.5)*sy-.5],prior_padded_intrinsic=[fx*sx,fy*sy,(cx+.5)*sx-.5+pad_x,(cy+.5)*sy-.5+pad_y],prior_depth_decanonical_factor=fx*sx/1000,source_id=int(stem.split('_')[-1]),image=str((DATA/'images'/image.name).resolve()),image_sha256=hashlib.sha256((DATA/'images'/image.name).read_bytes()).hexdigest(),depth=str(dep),normal=str(normal),native_intrinsic=[fx,fy,cx,cy],source_size=[cam.width,cam.height],depth_min=float(d.min()),depth_max=float(d.max()))
        records.append(rec)
        write(ROOT/'prior_index.json',records)
        write(ROOT/'STATUS.json',dict(status='running',stage='Metric3Dv2_prior',completed_views=len(records),total_views=208,elapsed_seconds=time.time()-started))
        print('E14 prior',stem,len(records),flush=True)
    torch.cuda.synchronize()
    write(ROOT/'priors_complete.json',dict(status='complete',views=208,model='Metric3Dv2 ViT-giant2 800K official',weight=str(weight),weight_sha256=h.hexdigest(),input_rgb='exact E11 training RGB, includes original black RAP background; no frontend rerun',preprocessing='Official keep-aspect to616x1064, mean padding, RGB channel normalization, unpad, bilinear to original, scale actual-width-resized corrected focal/1000; actual sx/sy halfpixel K replaces nominal scalar after integer resize',world_scale='Metric3D monocular outputs subsequently rescaled to VGGT gauge by official median per-image point-track scale; not independent metric ground truth',elapsed_seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
if __name__=='__main__':main()

