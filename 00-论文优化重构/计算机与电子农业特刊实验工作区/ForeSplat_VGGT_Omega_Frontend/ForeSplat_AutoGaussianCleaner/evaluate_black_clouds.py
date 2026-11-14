#!/usr/bin/env python3
"""Add actual black-cloud connected regions to completed native GGGS comparisons."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import argparse
import json
import time
import numpy as np
import torch
import yaml
from scipy import ndimage
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene,camera
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.free_support import free_support
from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha


def measure(rgb,alpha,outside,foreground_pixels):
    dark=outside&(alpha>.5)&(rgb@np.array([.2126,.7152,.0722])<.12)
    labels,count=ndimage.label(dark)
    sizes=np.bincount(labels.ravel())[1:]
    tiny=sizes<=max(1,int(.0005*foreground_pixels))
    return dict(dark_bg_high_alpha_pixels=int(dark.sum()),
        largest_dark_bg_component_pixels=int(sizes.max()) if count else 0,
        largest_dark_bg_component_fraction=float(sizes.max()/alpha.size) if count else 0.,
        dark_bg_components=int(count),isolated_small_dark_bg_components=int(tiny.sum()),
        isolated_small_dark_bg_pixels=int(sizes[tiny].sum()))


def measure_model(scene, model_path, token, input_sha256, cfg, supports, label):
    model,_=scene.load_model(model_path);start=time.perf_counter();records=[]
    for index in range(len(scene.views)):
        camera_object,_,mask,*_=scene.get_view(index);r=regions(mask,cfg['boundary_relative_width'])
        with torch.no_grad():native=scene.render(camera_object,model);render=arrays(native)
        for background,rgb in [('black',render['rgb']),('white',render['rgb']+1-render['alpha'][...,None])]:
            records.append(dict(index=index,name=camera_object.image_name,background=background,
                **measure(rgb,render['alpha'],r['bg'],int(mask.sum()))))
        if index in [0,len(scene.views)//2,len(scene.views)-1]:
            with torch.no_grad():white=arrays(scene.renderer.render(camera_object,model,scene.pipe,torch.ones(3,device='cuda'),0.,require_depth=True))
            error=float(np.abs(white['rgb']-(render['rgb']+1-render['alpha'][...,None])).max())
            assert error<2e-6,error
        if (index+1)%64==0:print(label,'training black clouds',index+1,flush=True)
    free=[]
    for index,support in enumerate(supports):
        v=support['view'];co=camera(v['K'],v['E'],v['width'],v['height'],v['name'],index)
        with torch.no_grad():render=arrays(scene.render(co,model))
        for background,rgb in [('black',render['rgb']),('white',render['rgb']+1-render['alpha'][...,None])]:
            free.append(dict(name=v['name'],background=background,
                **measure(rgb,render['alpha'],support['outside_surface'],int(support['surface_mask'].sum()))))
    metrics=['dark_bg_high_alpha_pixels','largest_dark_bg_component_pixels','largest_dark_bg_component_fraction',
        'dark_bg_components','isolated_small_dark_bg_components','isolated_small_dark_bg_pixels']
    result=dict(native_sha256=token,input_sha256=input_sha256,native_ply=str(Path(model_path).resolve()),
        training_mean={bg:{k:float(np.mean([r[k] for r in records if r['background']==bg])) for k in metrics} for bg in ['black','white']},
        training_views=records,free_views=free,seconds=time.perf_counter()-start,
        resolution_max_side=cfg['evaluation_max_side'],source_sha256=sha(Path(__file__).resolve()),
        definition='Actual native RGB luminance<0.12 and alpha>0.5, 4-connected pixels outside the shared target-mask boundary band; small component area<=0.0005*foreground area',
        free_view_status='Generated cameras with independently projected VGGT surface support; floating-component counts are image-space proxies, not manual 3D artifact labels',
        white_render_validation='Linear background compositing checked against three explicit native white-background renders per model')
    del model;torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outputs',nargs='+',type=Path)
    parser.add_argument('--include-original',action='store_true',help='Measure the same original native GGGS input as well; never substitute a cleaned model as baseline')
    a=parser.parse_args();base=Path(__file__).resolve().parent;c=base/'experiments/GGGS_Cleaner_Comparison'
    cfg=yaml.safe_load((base/'configs/default.yaml').read_text());torch.set_num_threads(8)
    outputs=a.outputs or [c/'baselines'/k for k in ['C0','C1_raw','C1_adaptive','C2_raw','C2_adaptive','C3','C3_pre_marker','C4']]
    cache={};scene=None;scene_key=None
    for output in outputs:
        complete=json.loads((output/'COMPLETE.json').read_text())
        if (output/'INVALIDATED.json').exists():raise ValueError(f'Invalidated result: {output}')
        if sha(output/'cleaned_gaussians.ply')!=complete['native_sha256']:
            raise ValueError(f'Native PLY differs from completion marker: {output}')
        reference_path=output/'original_reference.json'
        if not reference_path.exists():reference_path=output/'cleanup_metrics.json'
        reference=json.loads(reference_path.read_text())
        spec=reference['scene'];key=json.dumps(spec,sort_keys=True)
        if key!=scene_key:
            scene=Scene(spec,cfg);scene_key=key
            surface=external_surface(scene,scene.evidence_indices());supports=free_support(scene,surface)
        jobs=[(output/'cleaned_gaussians.ply',complete['native_sha256'],output/'BLACK_CLOUD_METRICS.json','cleaned')]
        if a.include_original:jobs.append((Path(spec['model']),complete['input_sha256'],output/'ORIGINAL_BLACK_CLOUD_METRICS.json','original'))
        for model_path,token,destination,label in jobs:
            if sha(model_path)!=token:raise ValueError('Original or cleaned native model differs from recorded execution')
            cache_key=(token,key,cfg['evaluation_max_side'],cfg['boundary_relative_width'])
            if destination.exists():
                old=json.loads(destination.read_text());assert old['native_sha256']==token
                assert old['resolution_max_side']==cfg['evaluation_max_side']
                cache[cache_key]=old;continue
            # Reuse the exact original E10 C0 measurement after checking the
            # original model, scene and common render resolution.
            if label=='original' and spec['plant']=='KongQueZhuYu':
                baseline=c/'baselines/C0';baseline_file=baseline/'BLACK_CLOUD_METRICS.json'
                if baseline_file.exists():
                    bm=json.loads((baseline/'COMPLETE.json').read_text());old=json.loads(baseline_file.read_text())
                    baseline_reference=baseline/'original_reference.json'
                    if not baseline_reference.exists():baseline_reference=baseline/'cleanup_metrics.json'
                    baseline_scene=bm.get('scene') or json.loads(baseline_reference.read_text())['scene']
                    if baseline_scene==spec and bm['native_sha256']==token==old['native_sha256'] and old['resolution_max_side']==cfg['evaluation_max_side']:
                        cache[cache_key]={**old,'original_baseline_reuse':str(baseline_file.resolve())}
            if cache_key in cache:
                result={**cache[cache_key],'native_ply':str(model_path.resolve()),'reused_identical_model':True}
            else:
                result=measure_model(scene,model_path,token,complete['input_sha256'],cfg,supports,output.name+' '+label)
            result['model_role']=label
            write_json(destination,result);cache[cache_key]=result
            print(output.name,label,result['training_mean'],'seconds',result['seconds'],flush=True)


if __name__=='__main__':main()
