#!/usr/bin/env python3
"""Measure why generalized historical background pruning stops too early."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse
from pathlib import Path
import json
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.historical.runner import HistoricalEvaluator
from ForeSplat_AutoGaussianCleaner.cleaner.historical.evidence import camera_indices
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.foreground_support import support_masks, support_metrics
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();out=a.experiment.resolve()
    r=json.loads((out/'execution.json').read_text())['reference']
    config=r['config']
    scene=Scene(r['scene'],config);torch.set_num_threads(8)
    initial=json.loads((out/'stage1_generalized/comparison/initial/metrics.json').read_text());selected=json.loads((out/'stage1_generalized/comparison/selected/metrics.json').read_text());base={v['index']:v for v in initial['views']}
    worst=sorted(selected['views'],key=lambda v:base[v['index']]['supported_boundary_coverage']-v['supported_boundary_coverage'],reverse=True)
    ids=[v['index'] for v in worst[:8]]
    old,_=scene.load_model(scene.spec['model'])
    paths={'rejected_broad_prune':out/'stage1_generalized/rejected_selected.ply'}
    dest=out/'comparison/full_view_boundary_diagnosis';dest.mkdir(parents=True,exist_ok=True)
    references={}
    for vi in ids:
        c,target,mask,depth,conf=scene.get_view(vi)
        with torch.no_grad():before=arrays(scene.render(c,old))
        references[vi]=before
    summaries={};worst=[]
    for label,path in paths.items():
        model,_=scene.load_model(path);records=[]
        evaluator=HistoricalEvaluator(scene,dest/label,original_model=old)
        initial=evaluator.evaluate(old,dest/label/'initial',ids)
        measured=evaluator.evaluate(model,dest/label/'candidate',ids)
        for vi in ids:
            c,target,mask,depth,conf=scene.get_view(vi)
            before=references[vi]
            with torch.no_grad():after=arrays(scene.render(c,model))
            reg=regions(mask,config['boundary_relative_width'])
            supported,geometry=support_masks(before,target,mask,depth,conf,config)
            metrics=support_metrics(after,before,before['rgb'],target,mask,reg,supported,geometry,config)
            lost=reg['boundary']&(before['alpha']>.5)&(after['alpha']<=.5)
            supported_lost=lost&supported
            old_error=((before['rgb']-target)**2).mean(2)
            new_error=((after['rgb']-target)**2).mean(2)
            metrics.update(index=vi,name=c.image_name,lost_boundary_pixels=int(lost.sum()),
                lost_boundary_depth_supported=int((lost&geometry).sum()),
                lost_boundary_photo_supported=int((lost&supported).sum()),
                lost_boundary_photo_improved=int((lost&(new_error<=old_error)).sum()),
                lost_supported_boundary=int(supported_lost.sum()),lost_supported_photo_improved=int((supported_lost&(new_error<=old_error)).sum()),
                lost_boundary_old_rgb_mse=float(old_error[lost].mean()) if lost.any() else 0.,
                lost_boundary_new_rgb_mse=float(new_error[lost].mean()) if lost.any() else 0.)
            records.append(metrics)
            if label=='rejected_broad_prune' and lost.any():
                worst.append((int(lost.sum()),vi,before,after,target,lost,geometry))
        keys=[k for k,v in records[0].items() if isinstance(v,(int,float)) and k!='index']
        summaries[label]=dict(model=str(path),sha256=sha(path),raw_initial=initial['mean'],raw_final=measured['mean'],
            mean={k:float(np.mean([row[k] for row in records])) for k in keys},views=records)
    for rank,(_,vi,before,after,target,lost,geometry) in enumerate(sorted(worst,key=lambda x:-x[0])[:6]):
        imgs=[]
        for img in [target,before['rgb'],after['rgb']]:
            im=Image.fromarray(np.uint8(np.clip(img,0,1)*255));im.thumbnail((400,500));imgs.append(im)
        overlay=np.clip(after['rgb'],0,1).copy();overlay[lost]=[1,0,0];overlay[lost&geometry]=[0,1,1]
        im=Image.fromarray(np.uint8(overlay*255));im.thumbnail((400,500));imgs.append(im)
        sheet=Image.new('RGB',(sum(i.width for i in imgs),max(i.height for i in imgs)+25),'white');draw=ImageDraw.Draw(sheet)
        x=0
        for label,im in zip(['RGB target','Original','Rejected broad prune','Lost boundary red; depth-supported cyan'],imgs):
            sheet.paste(im,(x,25));draw.text((x+3,5),label,fill='black');x+=im.width
        sheet.save(dest/f'worst_boundary_{rank}_{scene.views[vi]["name"]}.png')
    write_json(dest/'diagnosis.json',summaries)
    for label,s in summaries.items():
        print(label,'raw boundary',s['raw_initial']['boundary_coverage'],'->',s['raw_final']['boundary_coverage'],
            'supported boundary',s['mean']['supported_boundary_coverage'],
            'lost/improved/depth',s['mean']['lost_boundary_pixels'],s['mean']['lost_boundary_photo_improved'],s['mean']['lost_boundary_depth_supported'],flush=True)


if __name__=='__main__':main()
