"""Explicit opt-in final-model GPU evaluation, separate from the CPU cleaner.

Runs matched observed views and, when available, the existing historical free
protocol. No per-view NPZ, attribution, LPIPS or optimization. Original and
ablation models are held in memory. All PLY reads use the same rate-limited I/O.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
from plyfile import PlyData,PlyElement
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.evaluation.foreground_metrics import photometry
from ForeSplat_AutoGaussianCleaner.evaluation.artifact_metrics import artifact_metrics
from .core import CleanupConfig,decide_keep
from .features import extract_features
from .io import Meter,read_header,chunks
from .spatial import build


def read_rows(path,meter):
    with Path(path).open('rb',buffering=0) as stream:
        header=read_header(stream,meter);rows=np.empty(header.count,header.dtype)
        for start,block in chunks(stream,header,meter,16384):rows[start:start+len(block)]=block
    return header,rows


def model_from_rows(scene,header,rows):
    # Reuse native GGGS loader unchanged with an in-memory PLY stream.
    # This prevents its internal PlyData.read from bypassing disk throttling.
    sh=round(np.sqrt(sum(k.startswith('f_rest_') for k in rows.dtype.names)/3+1)-1)
    sg=sum(k.startswith('sg_sharpness_') for k in rows.dtype.names)
    model=scene.Model(sh,sg)
    # Native load_ply remains the authoritative field-to-tensor implementation.
    # Its PlyData.read gets the already decoded in-memory records, avoiding the
    # per-property binary parser and any second filesystem access.
    native_ply=PlyData([PlyElement.describe(rows,'vertex')],text=False,byte_order="<")
    with patch.object(PlyData,'read',return_value=native_ply):
        model.load_ply(io.BytesIO())
    for value in vars(model).values():
        if isinstance(value,torch.Tensor):value.requires_grad_(False)
    return model


def render_array(scene,camera,model):
    with torch.no_grad():out=scene.render(camera,model)
    return dict(rgb=out['render'].detach().cpu().numpy().transpose(1,2,0),
                alpha=out['mask'].detach().cpu().numpy().squeeze(),
                depth=out['median_depth'].detach().cpu().numpy().squeeze())


def run(args):
    if (args.output/"paired_quality.json").exists():
        raise FileExistsError("Choose a new evaluation directory")
    args.output.mkdir(parents=True,exist_ok=True)
    global torch
    import torch
    from scipy import ndimage
    torch.set_num_threads(2)
    spec=json.loads(args.scene.read_text())
    # External geometry is unnecessary for this evaluation; masks/RGB/cameras
    # come from the same existing corrected Scene implementation as old results.
    spec.pop('geometry_cameras',None);spec.pop('geometry_dir',None)
    scene=Scene(spec,dict(evaluation_max_side=args.max_side,evidence_views=args.views))
    indices=scene.evidence_indices()
    meter=Meter(20)
    header,original=read_rows(spec['model'],meter)
    variants={'original':None};ablations={}
    if not args.external_models:
        features=extract_features(original)
        index=build(features,4,16384)
        cfg=CleanupConfig()
        for stage in (['joint'] if args.skip_ablations else ['attributes','appearance','isolation','joint']):
            start=time.perf_counter();keep,stats=decide_keep(features,replace(cfg,stage=stage),index)
            name='fast' if stage=='joint' else 'ablation_'+stage
            variants[name]=keep
            ablations[stage]=dict(**stats,decision_s=time.perf_counter()-start)
        del features,index
    models=[]
    if spec['plant']=='KongQueZhuYu' and not args.skip_baselines:
        models.extend([('C1',ROOT/'experiments/GGGS_Cleaner_Comparison/baselines/C1_raw/cleaned_gaussians.ply'),
                       ('C2',ROOT/'experiments/GGGS_Cleaner_Comparison/baselines/C2_raw/cleaned_gaussians.ply'),
                       ('R5',ROOT.parent/'experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/additional_rounds/round_05/gaussians.ply')])
    if args.external_models:
        models=[(n,p) for n,p in models if n!='C1']
        extra=json.loads(args.external_models.read_text())
        if not isinstance(extra,dict) or not extra:raise ValueError('External manifest must be a nonempty name:path mapping')
        occupied=set(variants)|{n for n,p in models}
        for name,path in extra.items():
            if name in occupied or not name.replace('_','').isalnum():raise ValueError('Unique alphanumeric/underscore model names required')
            p=Path(path)
            if not p.is_file():raise FileNotFoundError(p)
            models.append((name,p));occupied.add(name)
        for name,path in models:
            if not path.is_file():raise FileNotFoundError(path)
    else:
        final=args.result_root/spec['plant']/'cleaned_gaussians.ply'
        retained_ids=np.flatnonzero(variants['fast'])
        with final.open('rb',buffering=0) as stream:
            h_final=read_header(stream,meter)
            if h_final.count!=len(retained_ids):raise ValueError('Final file count differs from decision')
            for start,block in chunks(stream,h_final,meter,16384):
                if block.tobytes()!=original[retained_ids[start:start+len(block)]].tobytes():raise ValueError('Final record bytes differ from derived decision')
        del retained_ids
    all_names=list(variants)+[name for name,p in models if p.exists()]
    baseline={};records={};thumbs={};free_thumbs={}
    protocol=json.loads(Path(spec['historical_free_protocol']).read_text()) if spec.get('historical_free_protocol') else None
    free_views=[]
    if protocol:
        for uid,view in enumerate(protocol['views']):
            c=argparse.Namespace(uid=uid,image_name=view['name'],image_width=view['width'],image_height=view['height'],
                                 FoVx=2*np.arctan(view['width']/(2*view['Fx'])),FoVy=2*np.arctan(view['height']/(2*view['Fy'])))
            for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
                setattr(c,k,torch.tensor(view[k],dtype=torch.float32,device='cuda'))
            with np.load(Path(spec['historical_free_silhouettes'])/(view['name']+'.npz')) as z:
                outside=ndimage.distance_transform_edt(~z['reference_mesh_silhouette'])>5
            free_views.append((c,outside))
    for name in all_names:
        if name in variants:
            rows=original if variants[name] is None else original[variants[name]]
            model=model_from_rows(scene,header,rows)
        else:
            h,rows=read_rows(dict(models)[name],meter);model=model_from_rows(scene,h,rows)
        count=len(rows);view_records=[];free_records=[]
        print('Evaluating',spec['plant'],name,count,flush=True)
        for order,vi in enumerate(indices):
            c,target,mask,*_=scene.get_view(vi);r=regions(mask,.002)
            a=render_array(scene,c,model)
            rec=dict(index=vi,name=c.image_name,**photometry(a['rgb'],target,mask,r,a['alpha']),**artifact_metrics(a['alpha'],a['rgb'],r))
            if name=='original':baseline[vi]=dict(alpha=a['alpha'],depth=a['depth'])
            old=baseline[vi];base_valid=mask&(old['alpha']>.5)&(old['depth']>0)
            rec['new_fg_alpha_hole_fraction']=float((base_valid&(a['alpha']<=.5)).sum()/max(base_valid.sum(),1))
            common=base_valid&(a['alpha']>.5)&(a['depth']>0)
            unit=max(float(np.median(old['depth'][base_valid])),1e-9) if base_valid.any() else 1
            rec['depth_changed_gt_3pct_fraction']=float((np.abs(a['depth'][common]-old['depth'][common])/unit>.03).sum()/max(common.sum(),1))
            view_records.append(rec)
            if order==0 and (name in ['original','fast'] or args.external_models):
                Image.fromarray(np.uint8(np.clip(a['rgb']+(1-a['alpha'][...,None])*.5,0,1)*255)).save(args.output/(name+'_first_observed_full.png'))
            if args.external_models:
                Image.fromarray(np.uint8(np.clip(a['rgb']+(1-a['alpha'][...,None])*.5,0,1)*255)).save(args.output/(name+'_observed_'+str(order).zfill(2)+'.png'))
                Image.fromarray(np.uint8(np.clip(a['alpha'],0,1)*255)).save(args.output/(name+'_alpha_'+str(order).zfill(2)+'.png'))
            if order<4 and (args.external_models or name in ['original','fast','C1','C2','R5']):
                im=Image.fromarray(np.uint8(np.clip(a['rgb']+(1-a['alpha'][...,None])*.5,0,1)*255));im.thumbnail((360,240))
                thumbs[(name,order)]=im
            scene.view_cache.clear()
        for order,(c,outside) in enumerate(free_views):
            a=render_array(scene,c,model)
            gray=a['rgb']+(1-a['alpha'][...,None])*protocol['background']
            occupied=outside&(a['alpha']>.5)
            black=occupied&(gray@np.array([.2126,.7152,.0722])<.12)
            labels,_=ndimage.label(black);sizes=np.bincount(labels.ravel())[1:]
            free_records.append(dict(name=c.image_name,black_pixels=int(black.sum()),background_alpha=float(a['alpha'][outside].sum()),
                                     small_black_components=int(((sizes>0)&(sizes<=200)).sum()),largest_black_component=int(sizes.max()) if len(sizes) else 0))
            if args.external_models:
                Image.fromarray(np.uint8(np.clip(gray,0,1)*255)).save(args.output/(name+'_free_'+str(order).zfill(2)+'.png'))
            if order<4 and (args.external_models or name in ['original','fast','C1','C2','R5']):
                im=Image.fromarray(np.uint8(np.clip(gray,0,1)*255));im.thumbnail((360,240));free_thumbs[(name,order)]=im
        means={k:float(np.mean([r[k] for r in view_records if r[k] is not None])) for k in view_records[0] if isinstance(view_records[0][k],(int,float)) and k!='index'}
        records[name]=dict(gaussians=count,mean=means,observed_views=view_records,free_views=free_records,
                           historical_free_black_pixels=sum(r['black_pixels'] for r in free_records) if free_records else None)
        (args.output/'progress.json').write_text(json.dumps(dict(status='running',results=records),ensure_ascii=False,indent=2)+'\n')
        print('Finished',name,'FG-PSNR',records[name]['mean']['fg_psnr'],'free black',records[name]['historical_free_black_pixels'],flush=True)
        del model,rows;torch.cuda.empty_cache()
    args.output.mkdir(parents=True,exist_ok=True)
    for kind,images in [('observed',thumbs),('free',free_thumbs)]:
        if not images:continue
        names=[n for n in (all_names if args.external_models else ['original','C1','C2','R5','fast']) if (n,0) in images]
        canvas=Image.new('RGB',(4*360,len(names)*270),'white');draw=ImageDraw.Draw(canvas)
        for row,name in enumerate(names):
            for col in range(4):canvas.paste(images[name,col],(col*360,row*270+25))
            draw.text((4,row*270+4),name,fill='black')
        canvas.save(args.output/(kind+'_contact.png'))
    result=dict(plant=spec['plant'],observed_indices=indices,observed_max_side=args.max_side,
                historical_protocol=spec.get('historical_free_protocol'),ablations=ablations,results=records,
                cleaning_stage="external_final_models" if args.external_models else "joint",
                external_models={n:str(p) for n,p in models} if args.external_models else None,
                measured_gpu='physical GPU1, final paired evaluation only',logical_ply_read_bytes=meter.read_bytes,
                semantic_boundary='RAP mask/skeleton and P3 silhouette proxies, not botanical semantic ground truth; observed subset is not full-view evaluation')
    (args.output/'paired_quality.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.output/'progress.json').unlink(missing_ok=True)
    print(json.dumps({k:dict(gaussians=v['gaussians'],mean=v['mean'],free_black=v['historical_free_black_pixels']) for k,v in records.items()},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--views',type=int,default=16);p.add_argument('--max-side',type=int,default=960);p.add_argument('--skip-ablations',action='store_true');p.add_argument('--skip-baselines',action='store_true')
    p.add_argument('--result-root',type=Path,default=ROOT/'experiments/fast_ply_validation/v2')
    p.add_argument('--external-models',type=Path,help='JSON name:path mapping; evaluate original/baselines and these final PLYs, skip Fast computation')
    run(p.parse_args())

if __name__=='__main__':
    main()
    # Native GGGS extension teardown can stall after all completed outputs.
    # This is evaluation-only; all image/JSON files are closed before exit.
    sys.stdout.flush()
    os._exit(0)
