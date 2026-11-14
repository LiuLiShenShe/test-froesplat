#!/usr/bin/env python3
"""Reconstruct one automatically selected completed-round C5-C rejection for pixel evidence."""
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(8)
    reference=json.loads((args.run/'original_reference.json').read_text())
    if reference['variant']!='C5-C':raise ValueError('Row reconstruction is restricted to the unmodified-parameter C5-C variant')
    if sha(reference['scene']['model'])!=reference['input_sha256']:raise ValueError('Original model has changed')
    rounds=sorted(p for p in (args.run/'rounds').iterdir() if (p/'group_scheduling.json').exists())
    if not rounds:raise ValueError('No completed grouped round')
    round_dir=rounds[-1]
    decisions=[json.loads(p.read_text()) for p in round_dir.glob('group_*/decision.json')]
    rejected=[d for d in decisions if not d['accepted'] and d['candidates']<=64 and any('benefit' not in r for r in d['reasons'])]
    if not rejected:raise ValueError('No rejected local group in the completed round')
    selected=max(rejected,key=lambda d:d['predicted_bg'])
    with np.load(round_dir/'evidence.npz') as evidence:ids=evidence['original_ids'].copy()
    previous=set(g for d in decisions if d['accepted'] and d['group']<selected['group'] for g in d['removed_ids'])
    ids=ids[~np.isin(ids,list(previous))]
    keep=~np.isin(ids,selected['removed_ids'])
    if int((~keep).sum())!=selected['candidates']:raise ValueError('Could not reconstruct exact trial rows')
    scene=Scene(reference['scene'],reference['config']);original,_=scene.load_model(scene.spec['model'])
    base_keep=np.zeros(len(original.get_xyz),bool);base_keep[ids]=True
    before=native_subset(scene,original,base_keep);after=native_subset(scene,before,keep)
    names={r.split(':',1)[0] for r in selected['reasons'] if ':' in r}
    indices=[i for i,v in enumerate(scene.views) if v['name'] in names]
    if not indices:raise ValueError('No recorded rejecting camera')
    records=[]
    for index in indices:
        camera,target,mask,depth,confidence=scene.get_view(index);r=regions(mask,scene.config['boundary_relative_width'])
        with torch.no_grad():o=arrays(scene.render(camera,original));b=arrays(scene.render(camera,before));a=arrays(scene.render(camera,after))
        error=((o['rgb']-target)**2).mean(2)
        valid=mask&np.isfinite(depth)&(depth>0)&np.isfinite(confidence)
        unit=max(float(np.median(depth[valid])),1e-9)
        high=valid&(confidence>=np.quantile(confidence[valid],.2))
        depth_good=high&(o['alpha']>.5)&(o['depth']>0)&np.isfinite(o['depth'])&(np.abs(o['depth']-depth)/unit<=.03)
        trusted=mask&((error<=.01)|depth_good)
        holes=mask&(b['alpha']>.5)&(a['alpha']<=.5)
        changed=mask&((np.abs(a['alpha']-b['alpha'])>1e-6)|(np.abs(a['rgb']-b['rgb']).max(2)>1e-6))
        dark=mask&((target@np.array([.2126,.7152,.0722]))<.12)
        def mean(value,selection):return float(value[selection].mean()) if selection.any() else None
        def mse(render,selection):return mean(((render['rgb']-target)**2).mean(2),selection)
        def thin(render,selection):return mean((render['alpha']>.5).astype(float),selection)
        depth_valid=mask&(b['alpha']>.5)&(a['alpha']>.5)&np.isfinite(b['depth'])&np.isfinite(a['depth'])&(b['depth']>0)&(a['depth']>0)
        depth_changed=depth_valid&(np.abs(a['depth']-b['depth'])/unit>.03)
        records.append(dict(name=camera.image_name,index=index,rejection_rules=[s for s in selected['reasons'] if s.startswith(camera.image_name+':')],
            changed_foreground_pixels=int(changed.sum()),changed_trusted_pixels=int((changed&trusted).sum()),
            alpha_holes=int(holes.sum()),trusted_alpha_holes=int((holes&trusted).sum()),untrusted_alpha_holes=int((holes&~trusted).sum()),
            original_rgb_mse=mse(o,mask),before_rgb_mse=mse(b,mask),after_rgb_mse=mse(a,mask),
            before_trusted_mse=mse(b,trusted),after_trusted_mse=mse(a,trusted),
            before_untrusted_mse=mse(b,mask&~trusted),after_untrusted_mse=mse(a,mask&~trusted),
            before_changed_dark_mse=mse(b,changed&dark),after_changed_dark_mse=mse(a,changed&dark),
            before_changed_dark_trusted_mse=mse(b,changed&dark&trusted),after_changed_dark_trusted_mse=mse(a,changed&dark&trusted),
            trusted_thin_before=thin(b,r['thin']&trusted),trusted_thin_after=thin(a,r['thin']&trusted),
            raw_depth_changed_pixels=int(depth_changed.sum()),trusted_depth_changed_pixels=int((depth_changed&depth_good).sum()),
            raw_depth_changed_toward_external_fraction=mean((np.abs(a['depth']-depth)<=np.abs(b['depth']-depth)).astype(float),depth_changed&high),
            changed_pixel_original_rgb_mse=mse(o,changed),changed_pixel_before_rgb_mse=mse(b,changed),changed_pixel_after_rgb_mse=mse(a,changed)))
        print(camera.image_name,records[-1],flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(args.output/'candidate_original_ids.npy',np.asarray(selected['removed_ids'],np.int64))
    write_json(args.output/'PIXEL_REJECTION_EVIDENCE.json',dict(selection='Highest predicted background contribution rejected group of at most 64 primitives from the latest completed C5-C round; automatic diagnostic selection only',
        run=str(args.run.resolve()),round=round_dir.name,decision=selected,input_sha256=reference['input_sha256'],
        before_original_ids_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest(),views=records,
        trust_definition='Diagnostic original RGB MSE<=0.01 OR original depth within3% of high-confidence existing VGGT; not a cleaner acceptance rule',
        dark_definition='Same target RGB luminance<0.12 as the actual dark-foreground evaluator',
        scope='No algorithm thresholds or Gaussians modified. Exact deleted row IDs and prior accepted groups reconstruct the recorded trial. No manual ROI or ID is used for operational cleaning.',source_sha256=sha(Path(__file__).resolve())))


if __name__=='__main__':main()
