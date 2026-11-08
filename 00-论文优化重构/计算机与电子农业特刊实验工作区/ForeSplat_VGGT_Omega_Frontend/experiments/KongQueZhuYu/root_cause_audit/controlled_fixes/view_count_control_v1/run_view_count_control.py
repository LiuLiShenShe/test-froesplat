"""Isolated 70/110 view controls; historical208 is references only.

Call through each model's existing uv env. --prepare-only is CPU metadata work;
real inference requires --run after root allocates physical GPU1. Original model
and preprocessing functions are called unchanged. No training/fusion/registration.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np

CONTROL=Path(__file__).resolve().parent
AUDIT=CONTROL.parents[1]
OMEGA=AUDIT.parents[2]
STANDARD=OMEGA.parent/'ForeSplat_Standard_VGGT_Frontend'
DATASET=OMEGA/'runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset'
STD_BASE=STANDARD/'experiments/KongQueZhuYu/standard_vggt_comparison'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    def sanitize(x):
        if isinstance(x,dict):return {k:sanitize(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [sanitize(v) for v in x]
        if isinstance(x,np.ndarray):return sanitize(x.tolist())
        if isinstance(x,np.generic):return sanitize(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(sanitize(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def source_rows():
    original=json.loads((DATASET/'metadata/image_manifest.json').read_text())['frames']
    historical=json.loads((STD_BASE/'Standard_VGGT_Frontend/metadata/input_manifest.json').read_text())['frames']
    hashes={r['image_name']:r for r in historical}
    if len(original)!=208 or [r['image_name'] for r in original]!=[r['image_name'] for r in historical]:
        raise ValueError('Require original identical208 ordered names')
    rows=[]
    for i,row in enumerate(original):
        name=row['image_name'];im=DATASET/'images'/name;mask=DATASET/'masks'/name
        if not im.exists() or not mask.exists():raise FileNotFoundError(name)
        rows.append(dict(usable_index=i,source_index=int(Path(name).stem.split('_')[-1]),image_name=name,
            image_path=str(im),mask_path=str(mask),original_width=row['original_width'],original_height=row['original_height'],
            rgb_sha256_historical=hashes[name]['rgb_sha256'],mask_sha256_historical=hashes[name]['mask_sha256'],
            hash_scope='Reused historical208 content hashes; new actual model-input hashes saved after preprocessing.'))
    return rows

def manifest(out,rows):
    out.mkdir(parents=True,exist_ok=True)
    fields=['selected_index']+list(rows[0])
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,row in enumerate(rows):w.writerow(dict(selected_index=i,**row))

def model_metadata(model):
    if model=='Omega':
        checkpoint=OMEGA/'weights/vggt_omega_1b_512.pt'
        legacy=OMEGA.parent/'ForeSplat_Fast3R_AMB3R_Frontend/experiments/KongQueZhuYu/VGGT_Omega_Existing/bundle.json'
        info=json.loads(legacy.read_text());weight_sha=info['metadata']['weight_sha256'] if 'metadata' in info and 'weight_sha256' in info['metadata'] else None
        if weight_sha is None:
            def find(x):
                if isinstance(x,dict):
                    if 'weight_sha256' in x:return x['weight_sha256']
                    for v in x.values():
                        found=find(v)
                        if found:return found
                return None
            weight_sha=find(info)
        source_files=[OMEGA/'foresplat_adapter/inference.py',OMEGA/'vggt_omega/models/vggt_omega.py',OMEGA/'vggt_omega/models/aggregator.py',OMEGA/'vggt_omega/utils/load_fn.py']
        reference=dict(camera_file=str(DATASET/'per_frame/cameras.json'),depth_dir=str(DATASET/'per_frame'),confidence_dir=str(DATASET/'per_frame'),
                       transform_file=str(DATASET/'metadata/preprocess_transforms.json'))
        commit=None;strategy=dict(name='balanced',image_resolution=512,patch_size=16,model_size_wh=[384,688],rgb_background='black',mask_interpolation='nearest',
            numerical_policy='Existing run_inference outerFP16; model FP16 aggregator, FP32 heads; pose conversion same existing adapter.')
    else:
        checkpoint=STANDARD/'checkpoints/model.pt';weight_info=STD_BASE/'model_metadata.json'
        weight_sha=json.loads(weight_info.read_text())['checkpoint_sha256']
        source_files=[STANDARD/'vggt_foreground/inference.py',STANDARD/'vggt_foreground/preprocess.py',STANDARD/'vggt/vggt/models/vggt.py']
        reference=dict(colmap_camera_dir=str(STD_BASE/'Standard_VGGT_Dataset/sparse/0'),depth_dir=str(STD_BASE/'Standard_VGGT_Frontend/depth'),
                       confidence_dir=str(STD_BASE/'Standard_VGGT_Frontend/confidence'),transform_file=str(STD_BASE/'Standard_VGGT_Frontend/metadata/input_manifest.json'))
        commit='a288dd0f14786c93483e45524328726ab7b1b4ce'
        strategy=dict(name='original square black padding',load_resolution=1024,model_resolution=518,model_size_wh=[518,518],
            rgb_interpolation=['PIL bicubic square→1024','torch bilinear1024→518 align_corners=False'],mask_interpolation='original nearest square→518',
            numerical_policy='Existing run_vggt_heads: BF16 aggregator on A6000; camera/depth heads and pose conversion FP32.',
            affine_policy='Keep historical nominal A for count-only comparison; separately save correct half-pixel A, without changing network input/predictions.')
    if not checkpoint.exists() or not weight_sha:raise ValueError('Existing checkpoint/hash required; no download or replacement')
    return dict(model=model,checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=weight_sha,
        weight_hash_scope='Reuse already recorded SHA; not re-read whole multiGB checkpoint solely for hashing.',official_model_commit=commit,
        commit_scope='Omega adapter is not an independent official git repository; source fingerprints used, enclosing workspace commit is not a model version.' if model=='Omega' else 'Independent official VGGT repository',
        source_file_sha256={str(p):sha(p) for p in source_files},preprocessing=strategy,seed=0,
        historical208_reference=reference,historical208_inference_rerun=False,
        scale_semantics='Learned native world scale; no measured metric calibration, alignment, fusion or scan scale fitting.',
        point_head_executed=False,depth_and_camera_support=True)

def prepare(model,counts):
    allrows=source_rows();info=model_metadata(model)
    write_json(CONTROL/f'{model}_model_metadata.json',info)
    selected={}
    for n in counts:
        indices=np.rint(np.linspace(0,207,n)).astype(np.int64)
        if len(np.unique(indices))!=n or np.any(np.diff(indices)<=0):raise ValueError('Uniform selection must be unique/in-order')
        rows=[allrows[int(i)] for i in indices];out=CONTROL/f'{model}_{n}'
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status')=='complete':
            raise ValueError('Refuse rerun/overwrite of completed new control '+str(out))
        manifest(out,rows)
        write_json(out/'run_parameters.json',dict(**info,input_views=n,selection='np.rint(np.linspace(0,207,N)).astype(int) on historical208 usable order',
                    usable_indices=indices.tolist(),original_input_order_preserved=True,only_experimental_factor='input view count N',
                    output_original_experiments_modified=False,training=False,CUDA_VISIBLE_DEVICES='1'))
        write_json(out/'status.json',dict(status='prepared',input_views=n,model=model,GPU_inference_executed=False))
        selected[n]=(out,rows)
    manifest(CONTROL/f'{model}_208_REFERENCE_ONLY',allrows)
    write_json(CONTROL/f'{model}_208_REFERENCE_ONLY/reference.json',dict(**info,status='historical_reference',input_views=208,
        note='No208 predictions copied or inferred; use exact existing source artifacts, all original predictions read-only.'))
    return info,selected

def run(model,info,selected):
    # Imports are deferred so --prepare-only does not access CUDA.
    import torch
    if model=='Omega':
        sys.path.insert(0,str(OMEGA))
        from foresplat_adapter.inference import load_model,run_inference
        from vggt_omega.utils.load_fn import load_and_preprocess_images_with_masks
    else:
        sys.path.insert(0,str(STANDARD));sys.path.insert(0,str(STANDARD/'vggt'))
        from vggt_foreground.inference import load_model,run_vggt_heads
        from vggt_foreground.preprocess import load_and_preprocess_images_square_with_masks
    if not torch.cuda.is_available():raise RuntimeError('GPU1 is required; do not fall back CPU/model substitution')
    device_name=torch.cuda.get_device_name(0)
    print('physical CUDA_VISIBLE_DEVICES=1, logicalcuda0:',device_name,flush=True)
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    t=time.perf_counter()
    try:
        network=load_model(info['checkpoint'],device='cuda');torch.cuda.synchronize();load_seconds=time.perf_counter()-t
    except Exception as exc:
        for key,(out,rows) in selected.items():
            n=len(rows)
            write_json(out/'status.json',dict(status='failed',failure_stage='model_load',model=model,input_views=n,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),GPU_inference_executed=False))
            (out/'RUN_REPORT.md').write_text('# Actual model-load failure\n\n'+traceback.format_exc())
        raise
    for key,(out,rows) in selected.items():
        n=len(rows)
        started=time.perf_counter();write_json(out/'status.json',dict(status='running',model=model,input_views=n))
        try:
            random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            prep=time.perf_counter();image_paths=[r['image_path'] for r in rows];mask_paths=[r['mask_path'] for r in rows]
            if model=='Omega':
                images,masks,transforms=load_and_preprocess_images_with_masks(image_paths,mask_paths,mode='balanced',image_resolution=512,patch_size=16,background='black')
                a=np.array([r['pixel_transform'] for r in transforms],np.float64);corrected=a.copy()
                model_rgb=images.numpy()
            else:
                images,masks,affines,sizes=load_and_preprocess_images_square_with_masks(image_paths,mask_paths,load_res=1024,model_res=518)
                a=affines.numpy();corrected=a.copy()
                for i,(w,h) in enumerate(sizes):corrected[i,:2,2]+=(518/max(w,h)-1)/2
                transforms=[dict(original_width=int(w),original_height=int(h),pad_left=int((max(w,h)-w)//2),pad_top=int((max(w,h)-h)//2),
                    load_width=1024,load_height=1024,model_width=518,model_height=518,pixel_transform=a[i].tolist(),pixel_transform_pixel_center=corrected[i].tolist(),
                    crop=[0,0,int(w),int(h)],mask_interpolation='nearest',rgb_interpolation='bicubic then bilinear') for i,(w,h) in enumerate(sizes)]
                model_rgb=None
            prep_seconds=time.perf_counter()-prep
            torch.cuda.reset_peak_memory_stats();images_cuda=images.to('cuda')
            if model=='Standard':
                # Repeat the exact original FP32 GPU interpolation to persist the
                # model input; inference performs the same deterministic operation.
                model_rgb=torch.nn.functional.interpolate(images_cuda,size=(518,518),mode='bilinear',align_corners=False).float().cpu().numpy()
            expected=info['preprocessing']['model_size_wh'];h,w=model_rgb.shape[-2:]
            if [w,h]!=expected or masks.shape!=(n,h,w):raise ValueError('Require fixed original preprocessing size/mask alignment')
            torch.cuda.synchronize();infer=time.perf_counter()
            predictions=run_inference(network,images_cuda,device='cuda') if model=='Omega' else run_vggt_heads(network,images_cuda,model_res=518,device='cuda')
            torch.cuda.synchronize();call_seconds=time.perf_counter()-infer
            peak_bytes=int(torch.cuda.max_memory_allocated());peak_reserved=int(torch.cuda.max_memory_reserved())
            depth=predictions['depth'];depth=depth[...,0] if depth.ndim==4 else depth
            conf=predictions['depth_conf'] if model=='Omega' else predictions['conf']
            e=np.asarray(predictions['extrinsic'],np.float64);k=np.asarray(predictions['intrinsic'],np.float64)
            e4=np.tile(np.eye(4),(n,1,1));e4[:,:3,:4]=e[:,:3,:4]
            if depth.shape!=(n,h,w) or conf.shape!=(n,h,w):raise ValueError('Raw depth/conf shape mismatch')
            per=out/'per_frame';per.mkdir(exist_ok=True);cameras=[];mask_np=masks.numpy().astype(bool)
            for i,row in enumerate(rows):
                stem=Path(row['image_name']).stem
                np.save(per/f'depth_{stem}.npy',depth[i].astype(np.float32))
                np.save(per/f'confidence_{stem}.npy',conf[i].astype(np.float32))
                np.save(per/f'model_rgb_float32_{stem}.npy',model_rgb[i].astype(np.float32))
                np.save(per/f'model_mask_{stem}.npy',mask_np[i].astype(np.uint8))
                cameras.append(dict(sequence_index=i,usable_index=row['usable_index'],source_index=row['source_index'],image_name=row['image_name'],
                    intrinsic_model=k[i],extrinsic_model_w2c=e4[i],c2w=np.linalg.inv(e4[i]),pixel_transform=a[i],pixel_transform_pixel_center=corrected[i],
                    intrinsic_original_historical=np.linalg.inv(a[i])@k[i],intrinsic_original_pixel_center=np.linalg.inv(corrected[i])@k[i],
                    model_width=w,model_height=h,original_width=row['original_width'],original_height=row['original_height'],
                    input_float32_sha256=hashlib.sha256(np.ascontiguousarray(model_rgb[i]).tobytes()).hexdigest(),
                    mask_sha256_model=hashlib.sha256(np.ascontiguousarray(mask_np[i]).tobytes()).hexdigest(),
                    foreground_coverage_valid_depth=float(np.mean((np.isfinite(depth[i])&(depth[i]>0))[mask_np[i]]))))
            write_json(out/'cameras.json',dict(frames=cameras,input_views=n,coordinate_convention='OpenCV W2C; c2w is inverse; native model reference, no scene alignment/fusion',
                       source_order=[r['image_name'] for r in rows],depth_support=True,point_head_executed=False))
            write_json(out/'pixel_transforms.json',transforms)
            aux={key:val for key,val in predictions.items() if isinstance(val,np.ndarray) and key not in ['depth','conf','depth_conf','images','intrinsic','extrinsic','world_points_from_depth']}
            if aux:np.savez(out/'raw_auxiliary_predictions.npz',**aux)
            np.savez(out/'cameras.npz',extrinsic_model_w2c=e4,intrinsic_model=k,c2w=np.linalg.inv(e4),pixel_transform=a,pixel_transform_pixel_center=corrected)
            metadata=dict(**info,status='complete',input_views=n,torch_version=torch.__version__,python=sys.version,device_name=device_name,CUDA_VISIBLE_DEVICES='1',
                condition=rows[0].get('condition','RAP_foreground'),
                input_tensor_dtype=str(images.dtype),saved_model_rgb_dtype=str(model_rgb.dtype),raw_depth_dtype=str(depth.dtype),raw_confidence_dtype=str(conf.dtype),
                model_load_seconds=load_seconds,preprocessing_seconds=prep_seconds,official_inference_call_seconds=call_seconds,
                inference_time_scope='Unchanged official adapter call incl tensor CPU export; Omega also computes depth unprojection. Not pure network-only walltime.',
                total_pipeline_seconds=time.perf_counter()-started,peak_vram_gib=peak_bytes/2**30,peak_reserved_gib=peak_reserved/2**30,
                peak_scope='Per-count reset after model loaded, includes exact repeated model-input interpolation on Standard and unchanged inference, excluding model-load transient.',
                all_depth_fields_saved=True,raw_point_head_saved=False,derived_world_points_saved=False,
                point_output_note='Both selected APIs run camera/depth, no point head; depth+K+W2C is preserved for exact unprojection. Omega computes derived pointmap internally but it is not duplicated on disk.',
                source_order=[r['image_name'] for r in rows],training=False,original_experiments_modified=False)
            metadata['control_factor']=info.get('control_factor','input_view_count')
            write_json(out/'metadata.json',metadata);write_json(out/'status.json',dict(status='complete',model=model,input_views=n,per_frame_depth=n,per_frame_confidence=n,GPU_inference_executed=True))
            (out/'RUN_REPORT.md').write_text(f'# {model} {n}-view controlled inference\n\nActual completed {n} ordered usable frames, condition={metadata["condition"]}, factor={metadata["control_factor"]}, seed0, physical GPU1, existing original model and preprocessing. No training/fusion; historical208 reused.\n\nOfficial adapter call {call_seconds:.3f}s; peak allocated {peak_bytes/2**30:.3f}GiB. Raw depth/conf, exact float32 model RGB, nearest Mask, native W2C/K, historical and corrected pixel-center A are saved. No point head.\n\nSee metadata.json/run_parameters.json/manifest.csv; statistical geometry assessment is independent of the predicted factor effect.\n')
            print(model,n,metadata['condition'],'COMPLETE',call_seconds,'s',peak_bytes/2**30,'GiB',flush=True)
            del predictions,images,images_cuda,model_rgb,masks,depth,conf;gc.collect();torch.cuda.empty_cache()
        except Exception as exc:
            write_json(out/'status.json',dict(status='failed',model=model,input_views=n,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),GPU_inference_executed=True))
            (out/'RUN_REPORT.md').write_text('# Actual failed count-control attempt\n\n'+traceback.format_exc()+'\nNo alternate resolution/count/model substituted.\n')
            raise
    del network;gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['Omega','Standard'],required=True);parser.add_argument('--counts',type=int,nargs='+',default=[70,110]);parser.add_argument('--run',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if any(n not in [70,110] for n in args.counts):parser.error('Only new70/110 are authorized;208 is references only')
    if args.run and args.prepare_only:parser.error('Choose prepare-only or run')
    info,selected=prepare(args.model,args.counts)
    print(args.model,'prepared',[(n,len(r)) for n,(_,r) in selected.items()],flush=True)
    if args.run:run(args.model,info,selected)

if __name__=='__main__':main()
