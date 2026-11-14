#!/usr/bin/env python3
"""Separate actual foreground-edge support from background occupancy in a rejected trial."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import native_subset
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(8)
    e=json.loads((args.audit/'PIXEL_REJECTION_EVIDENCE.json').read_text())
    v=json.loads((args.audit/'PHOTOMETRIC_SPLIT_FULL_VALIDATION.json').read_text());run=Path(e['run'])
    reference=json.loads((run/'original_reference.json').read_text())
    if reference['variant']!='C5-C' or sha(reference['scene']['model'])!=e['input_sha256']:raise ValueError('Invalid original context')
    folder=run/'rounds'/e['round']
    with np.load(folder/'evidence.npz') as z:ids=z['original_ids'].copy()
    decisions=[json.loads(p.read_text()) for p in folder.glob('group_*/decision.json')]
    earlier={g for d in decisions if d['accepted'] and d['group']<e['decision']['group'] for g in d['removed_ids']}
    ids=ids[~np.isin(ids,list(earlier))]
    scene=Scene(reference['scene'],reference['config']);original,_=scene.load_model(scene.spec['model'])
    keep=np.zeros(len(original.get_xyz),bool);keep[ids]=True
    before=native_subset(scene,original,keep);after=native_subset(scene,before,~np.isin(ids,v['candidate_original_ids']))
    names={r.split(':',1)[0] for r in v['reasons'] if 'boundary_coverage' in r or 'dark_leaf_alpha' in r};records=[]
    for index,view in enumerate(scene.views):
        if view['name'] not in names:continue
        camera,target,mask,depth,confidence=scene.get_view(index);r=regions(mask,scene.config['boundary_relative_width'])
        with torch.no_grad():o=arrays(scene.render(camera,original));b=arrays(scene.render(camera,before));a=arrays(scene.render(camera,after))
        valid=mask&np.isfinite(depth)&(depth>0)&np.isfinite(confidence);unit=max(float(np.median(depth[valid])),1e-9)
        high=valid&(confidence>=np.quantile(confidence[valid],.2))
        depth_good=high&(o['alpha']>.5)&np.isfinite(o['depth'])&(o['depth']>0)&(np.abs(o['depth']-depth)/unit<=.03)
        trusted=mask&((((o['rgb']-target)**2).mean(2)<=.01)|depth_good)
        dark=mask&((target@np.array([.2126,.7152,.0722]))<.12)
        loss=(b['alpha']>.5)&(a['alpha']<=.5)
        def mean(value,m):return float(value[m].mean()) if m.any() else None
        record=dict(name=camera.image_name,index=index)
        for label,m in [('raw_boundary',r['boundary']),('inside_mask_boundary',r['boundary']&mask),('outside_mask_boundary',r['boundary']&~mask)]:
            record[label]=dict(pixels=int(m.sum()),new_holes=int((m&loss).sum()),before_coverage=mean((b['alpha']>.5).astype(float),m),after_coverage=mean((a['alpha']>.5).astype(float),m))
        record['dark']={label:dict(pixels=int(m.sum()),original_alpha=mean(o['alpha'],m),before_alpha=mean(b['alpha'],m),after_alpha=mean(a['alpha'],m),
            new_holes=int((loss&m).sum()),before_mse=mean(((b['rgb']-target)**2).mean(2),m),after_mse=mean(((a['rgb']-target)**2).mean(2),m))
            for label,m in [('all',dark),('trusted',dark&trusted),('untrusted',dark&~trusted)]}
        records.append(record);print(record,flush=True)
    write_json(args.audit/'BOUNDARY_FOREGROUND_BACKGROUND_AUDIT.json',dict(views=records,input_sha256=e['input_sha256'],
        scope='Exact native trial reconstruction and diagnostic inside/outside decomposition only; no quality thresholds or cleaner parameters changed.',source_sha256=sha(Path(__file__).resolve())))


if __name__=='__main__':main()
