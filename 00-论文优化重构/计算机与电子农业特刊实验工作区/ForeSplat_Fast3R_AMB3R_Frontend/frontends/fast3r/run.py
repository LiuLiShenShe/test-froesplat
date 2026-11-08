#!/usr/bin/env python3
"""Official Fast3R all-view inference, with observational PnP instrumentation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import contextlib
import hashlib
import json
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party'/'fast3r'))
import cv2
import numpy as np
import torch
from frontends.common import EXPERIMENT,load_inputs,export_bundle,json_write,sha256
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
import fast3r.models.multiview_dust3r_module as pnp_module
from fast3r.dust3r.inference_multiview import inference

def signature(x):
    a=np.asarray(x.detach().cpu() if hasattr(x,'detach') else x).reshape(-1,3)
    return hashlib.sha256(a[::max(1,len(a)//11)].astype(np.float32).tobytes()).hexdigest()

@contextlib.contextmanager
def observe_pnp(preds):
    """Record actual official solver results without changing solver arguments."""
    ids={signature(p['pts3d_in_other_view']):i for i,p in enumerate(preds)}
    records=[dict(status='not_called') for _ in preds];local=threading.local()
    original=pnp_module.fast_pnp;original_cv=cv2.solvePnPRansac
    def cv_observer(*args,**kwargs):
        result=original_cv(*args,**kwargs)
        if hasattr(local,'index'):
            records[local.index]['pnp_inliers']=0 if result[3] is None else len(result[3])
        return result
    def observer(pts,focal,mask,*args,**kwargs):
        i=ids[signature(pts)];local.index=i
        records[i]=dict(status='started',valid_points=int(mask.sum()),iterations=kwargs.get('niter_PnP',100))
        try:
            result=original(pts,focal,mask,*args,**kwargs)
            records[i]['status']='success' if result[1] is not None else 'failed_identity_fallback'
            return result
        finally:del local.index
    pnp_module.fast_pnp=observer;cv2.solvePnPRansac=cv_observer
    try:yield records
    finally:pnp_module.fast_pnp=original;cv2.solvePnPRansac=original_cv

def validate_geometry(preds,poses,focals,masks,status):
    measurements=[]
    for i,(p,pose,focal) in enumerate(zip(preds,poses,focals)):
        xyz=p['pts3d_in_other_view'][0].float().numpy();conf=p['conf'][0].float().numpy()
        h,w=conf.shape;y,x=np.indices((h,w));valid=masks[i]&np.isfinite(xyz).all(-1)&(conf>1)
        camera=xyz@np.linalg.inv(pose)[:3,:3].T+np.linalg.inv(pose)[:3,3]
        valid &= camera[...,2]>1e-8
        projection=camera[valid,:2]/camera[valid,2:3]*focal+np.array([w/2,h/2])
        error=np.linalg.norm(projection-np.stack((x[valid],y[valid]),-1),axis=-1)
        measurements.append(dict(frame_index=i,foreground_reprojection_median_px=float(np.median(error)) if error.size else None,
            foreground_reprojection_p90_px=float(np.quantile(error,.9)) if error.size else None,points=int(error.size),
            rotation_determinant=float(np.linalg.det(pose[:3,:3])),orthogonality_error=float(np.linalg.norm(pose[:3,:3].T@pose[:3,:3]-np.eye(3))),
            positive_depth_fraction=float((camera[...,2][masks[i]]>0).mean()),pnp_status=status[i]['status']))
    return dict(first_camera_distance_to_identity=float(np.linalg.norm(poses[0]-np.eye(4))),
        first_camera_note='Global head predicts first-camera reference; PnP estimates it and is not forcibly fixed to identity',
        convention='C2W validated by inversion followed by model-pixel reprojection; OpenCV solver emits W2C and official code inverts it',frames=measurements)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--views',type=int,default=0)
    parser.add_argument('--long-edge',type=int,default=512);parser.add_argument('--head-views',type=int,default=2)
    parser.add_argument('--precision',choices=('bfloat16','float32'),default='bfloat16')
    parser.add_argument('--manifest',type=Path,default=EXPERIMENT/'manifest.csv')
    parser.add_argument('--output',type=Path,default=EXPERIMENT/'Fast3R');parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();out=args.output
    if (out/'bundle.json').exists():raise FileExistsError(f'preserving existing bundle {out}')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);cv2.setRNGSeed(args.seed)
    torch.set_num_threads(8)
    with open(args.manifest) as f:
        import csv
        n=sum(r['status']=='ok' for r in csv.DictReader(f))
    indices=None if args.views==0 or args.views>=n else list(range(args.views))
    inputs=load_inputs(args.manifest,args.long_edge,16,indices)
    h,w=inputs['masks'].shape[-2:]
    views=[dict(img=inputs['images'][i:i+1],true_shape=torch.tensor([[h,w]]),idx=i,instance=r['name']) for i,r in enumerate(inputs['rows'])]
    device=torch.device('cuda:0');model=Fast3R.from_pretrained(str(ROOT/'weights'/'fast3r')).to(device).eval()
    model.set_max_parallel_views_for_head(args.head_views)
    lit=MultiViewDUSt3RLitModule.load_for_inference(model);lit.eval()
    weights=ROOT/'weights'/'fast3r'/'model.safetensors'
    meta=dict(method='Fast3R',scale='ambiguous; arbitrary learned scene scale, not metres',
        world_frame='first input camera reference',git_commit=subprocess.check_output(['git','-C',str(ROOT/'third_party'/'fast3r'),'rev-parse','HEAD'],text=True).strip(),
        weight_sha256=sha256(weights),weight_source='jedyang97/Fast3R_ViT_Large_512',
        license='FAIR Noncommercial Research License',seed=args.seed,gpu_physical=1,
        torch_version=torch.__version__,torch_cuda=torch.version.cuda,
        input_views=len(views),available_views=n,full_sequence=len(views)==n,
        inference_mode='one single globally interacting forward pass',precision=args.precision,
        precision_protocol='bf16 autocast' if args.precision=='bfloat16' else 'official precision="32" disables outer autocast; original internal attention precision is retained',
        head_parallel_views=args.head_views,long_edge=args.long_edge,padding_alignment=16,crop='none',
        manifest_path=str(args.manifest.resolve()),manifest_sha256=sha256(args.manifest),
        pnp_iterations=100,focal_method='first_view_from_global_head')
    out.mkdir(parents=True,exist_ok=True);json_write(out/'run_parameters.json',meta)
    torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    print(f'BEGIN Fast3R {len(views)} views {w}x{h} physical GPU1',flush=True)
    try:
        # Upstream loss_of_one_batch recognizes the STRING "32". Passing
        # torch.float32 falls through to default FP16 autocast in this checkout.
        dtype=torch.bfloat16 if args.precision=='bfloat16' else '32'
        with torch.inference_mode():output,profile=inference(views,model,device,dtype=dtype,verbose=True,profiling=True)
        torch.cuda.synchronize();meta['inference_seconds']=time.perf_counter()-t;meta['profiling']=profile
        meta['peak_vram_gib']=torch.cuda.max_memory_allocated()/2**30
        preds=output['preds'];print('FORWARD_DONE',meta['inference_seconds'],meta['peak_vram_gib'],flush=True)
        del model,lit
        torch.cuda.empty_cache()
        pnp_time=time.perf_counter()
        with observe_pnp(preds) as status:
            poses, focals=MultiViewDUSt3RLitModule.estimate_camera_poses(preds,niter_PnP=100,focal_length_estimation_method='first_view_from_global_head')
        poses=np.asarray(poses[0]);focals=np.asarray(focals[0],float)
        meta['pnp_seconds']=time.perf_counter()-pnp_time
        meta['inference_time_scope']='official multi-view inference call and GPU synchronization; excludes model loading, input preparation, PnP and output serialization'
        meta['total_frontend_seconds']=meta['inference_seconds']+meta['pnp_seconds']
        meta['total_frontend_time_scope']='network inference plus official CPU PnP; excludes input preparation, model loading, geometry diagnostics and Bundle serialization'
        k=np.repeat(np.eye(3)[None],len(views),axis=0);k[:,0,0]=focals;k[:,1,1]=focals;k[:,0,2]=w/2;k[:,1,2]=h/2
        json_write(out/'cameras'/'estimated_focal_lengths.json',[
            dict(name=r['name'],focal_length_model_px=float(focals[i]),
                 focal_length_original_xy_px=[float(focals[i]/s) for s in inputs['transforms'][i]['resize_scale_xy']],
                 estimation_method=meta['focal_method'],pnp_status=status[i]['status'])
            for i,r in enumerate(inputs['rows'])])
        json_write(out/'geometry_validation.json',validate_geometry(preds,poses,focals,inputs['masks'],status))
        predictions=[dict(world_points=p['pts3d_in_other_view'][0],local_points=p['pts3d_local'][0],world_points_conf=p['conf'][0],local_confidence=p['conf_local'][0],
            pts3d_in_other_view=p['pts3d_in_other_view'][0],pts3d_local=p['pts3d_local'][0],conf=p['conf'][0],conf_local=p['conf_local'][0]) for p in preds]
        export_bundle(out,inputs,predictions,poses,k,meta,status)
        json_write(out/'status.json',dict(status='complete',views=len(views),pnp_success=sum(s['status']=='success' for s in status)))
        print('BUNDLE_DONE',str(out),flush=True)
    except Exception as e:
        meta.update(status='failed',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.perf_counter()-t,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
        json_write(out/'status.json',meta);(out/'error.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
