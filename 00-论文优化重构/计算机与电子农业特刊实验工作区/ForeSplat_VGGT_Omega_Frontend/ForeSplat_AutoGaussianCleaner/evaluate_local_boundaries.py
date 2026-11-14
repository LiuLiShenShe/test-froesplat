#!/usr/bin/env python3
"""Add the same foreground-local contour metric to completed C0–C4 models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import json
import argparse
import time
import numpy as np
import torch
import yaml
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import contour_metrics
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--methods',nargs='+',default=['C0','C1_raw','C1_adaptive','C2_raw','C2_adaptive','C3','C3_pre_marker','C4'])
    args=parser.parse_args()
    base=Path(__file__).resolve().parent;c=base/'experiments/GGGS_Cleaner_Comparison'
    cfg=yaml.safe_load((base/'configs/default.yaml').read_text());spec=json.loads((base/'configs/scenes/KongQueZhuYu.json').read_text())
    scene=Scene(spec,cfg);torch.set_num_threads(8)
    original=base/'experiments/all_19_plants_v3/development/KongQueZhuYu/C/comparison/initial'
    cache={};references={}
    for method in args.methods:
        out=c/'baselines'/method;complete=json.loads((out/'COMPLETE.json').read_text());model_hash=complete['native_sha256']
        destination=out/'FG_LOCAL_METRICS.json'
        if destination.exists():
            old=json.loads(destination.read_text())
            if old['native_sha256']!=model_hash:raise ValueError('Extra-metric model changed')
            cache[model_hash]=old;continue
        if model_hash in cache:
            report={**cache[model_hash],'reused_identical_model_metric':True,'method':method,'native_ply':str((out/'cleaned_gaussians.ply').resolve())}
            write_json(destination,report);continue
        model,_=scene.load_model(out/'cleaned_gaussians.ply');start=time.perf_counter();records=[]
        for index in range(len(scene.views)):
            camera,_,mask,*_=scene.get_view(index)
            if index not in references:
                with np.load(original/f'{index:04d}.npz') as z:references[index]=z['alpha'].copy()
            with torch.no_grad():alpha=scene.render(camera,model)['mask'].detach().cpu().numpy().squeeze()
            r=regions(mask,cfg['boundary_relative_width'])
            records.append(dict(index=index,name=camera.image_name,**contour_metrics(alpha,references[index],mask,r['radius'])))
        report=dict(method=method,native_sha256=model_hash,input_sha256=complete['input_sha256'],
            mean={k:float(np.mean([v[k] for v in records])) for k in ['fg_local_boundary_f1','fg_alpha_loss_pixels','fg_alpha_loss_mass']},
            views=records,resolution_max_side=cfg['evaluation_max_side'],seconds=time.perf_counter()-start,
            native_ply=str((out/'cleaned_gaussians.ply').resolve()),source_sha256=sha(Path(__file__).resolve()),
            definition='Same native GGGS cameras and target silhouette band distance<=3*boundary radius; additional metrics, original evaluations remain intact')
        write_json(destination,report);cache[model_hash]=report;del model
        print(method,report['mean'],'seconds',report['seconds'],flush=True)


if __name__=='__main__':main()
