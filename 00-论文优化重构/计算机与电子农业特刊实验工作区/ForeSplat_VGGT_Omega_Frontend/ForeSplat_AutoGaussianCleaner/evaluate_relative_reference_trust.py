#!/usr/bin/env python3
"""Development-only audit of false foreground occupancy in a fixed GGGS baseline."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays


def main():
    p=Path(__file__).resolve().parent/'experiments/GGGS_Cleaner_Comparison'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs',type=Path,nargs='+')
    args=parser.parse_args()
    outputs=args.outputs or [p/'baselines/C3_pre_marker']
    for output in outputs:
        evaluate(output,legacy_destination=(p/'comparisons/RELATIVE_ORIGINAL_FOREGROUND_REFERENCE_TRUST_AUDIT.json') if not args.outputs else None)


def evaluate(output,legacy_destination=None):
    begin=time.perf_counter();output=output.resolve()
    if not (output/'COMPLETE.json').exists() or (output/'INVALIDATED.json').exists():
        raise ValueError(f'Requires a completed, non-invalidated model: {output}')
    report=json.loads((output/'cleanup_metrics.json').read_text())
    complete=json.loads((output/'COMPLETE.json').read_text())
    specification=complete.get('scene') or report.get('scene')
    scene=Scene(specification,report['config'])
    if sha(scene.spec['model'])!=report['input_sha256']:
        raise ValueError('Input model differs from the completed experiment')
    if complete.get('native_sha256') and sha(output/'cleaned_gaussians.ply')!=complete['native_sha256']:
        raise ValueError('Candidate model differs from its completion marker')
    original,_=scene.load_model(scene.spec['model'])
    candidate,_=scene.load_model(output/'cleaned_gaussians.ply');records=[]
    for index in range(len(scene.views)):
        camera,target,mask,depth,confidence=scene.get_view(index);r=regions(mask,scene.config['boundary_relative_width'])
        with torch.no_grad():before=arrays(scene.render(camera,original));after=arrays(scene.render(camera,candidate))
        error=((before['rgb']-target)**2).mean(2)
        valid=mask&np.isfinite(depth)&(depth>0)&np.isfinite(confidence)
        high=valid&(confidence>=np.quantile(confidence[valid],.2)) if valid.any() else np.zeros_like(mask)
        unit=max(float(np.median(depth[valid])),1e-9) if valid.any() else 1.
        depth_good=high&(before['alpha']>.5)&np.isfinite(before['depth'])&(before['depth']>0)&(np.abs(before['depth']-depth)/unit<=.03)
        relative_error=error/np.maximum((target**2).mean(2)+(1/255)**2,1e-12)
        appearance_good=mask&(relative_error<=.25)
        trusted=mask&(appearance_good|depth_good)
        untrusted=mask&~trusted
        holes=mask&(before['alpha']>.5)&(after['alpha']<=.5)
        def mean(a,selection):return float(a[selection].mean()) if selection.any() else None
        def coverage(a,selection):return float((a[selection]>.5).mean()) if selection.any() else 1.
        tg=depth_good&(after['alpha']>.5)&np.isfinite(after['depth'])&(after['depth']>0)
        delta=np.abs(before['depth'][tg]-after['depth'][tg])/unit
        no,nn=np.linalg.norm(before['normal'],axis=2),np.linalg.norm(after['normal'],axis=2)
        normals=tg&(no>.1)&(nn>.1)
        dot=(before['normal'][normals]*after['normal'][normals]).sum(1)/np.maximum(no[normals]*nn[normals],1e-9)
        angles=np.degrees(np.arccos(np.clip(dot,-1,1)))
        record=dict(index=index,name=camera.image_name,foreground_pixels=int(mask.sum()),trusted_fraction=float(trusted.sum()/max(mask.sum(),1)),
            original_trusted_alpha=mean(before['alpha'],trusted),cleaned_trusted_alpha=mean(after['alpha'],trusted),
            original_raw_alpha=mean(before['alpha'],mask),cleaned_raw_alpha=mean(after['alpha'],mask),
            raw_new_hole_pixels=int(holes.sum()),trusted_new_hole_pixels=int((holes&trusted).sum()),
            untrusted_new_hole_pixels=int((holes&untrusted).sum()),
            original_trusted_thin=coverage(before['alpha'],r['thin']&trusted),cleaned_trusted_thin=coverage(after['alpha'],r['thin']&trusted),
            trusted_depth_change_p95=float(np.quantile(delta,.95)) if len(delta) else 0.,
            trusted_depth_changed_fraction=float((delta>.03).mean()) if len(delta) else 0.,
            trusted_depth_new_invalid=float((depth_good&~tg).sum()/max(depth_good.sum(),1)),
            trusted_normal_change_p95=float(np.quantile(angles,.95)) if len(angles) else 0.,
            original_untrusted_rgb_mse=mean(((before['rgb']-target)**2).mean(2),untrusted),
            cleaned_untrusted_rgb_mse=mean(((after['rgb']-target)**2).mean(2),untrusted))
        records.append(record)
        if (index+1)%32==0:print('reference trust diagnostic',index+1,flush=True)
    keys=['trusted_fraction','original_trusted_alpha','cleaned_trusted_alpha','original_raw_alpha','cleaned_raw_alpha',
        'original_trusted_thin','cleaned_trusted_thin','trusted_depth_change_p95','trusted_depth_changed_fraction','trusted_depth_new_invalid','trusted_normal_change_p95']
    result=dict(input_sha256=report['input_sha256'],method=report['method'],plant=scene.spec.get('plant',report.get('plant')),
        native_sha256=sha(output/'cleaned_gaussians.ply'),scene=scene.spec,views=records,
        mean={key:float(np.mean([v[key] for v in records if v[key] is not None])) for key in keys},
        raw_hole_pixels=sum(v['raw_new_hole_pixels'] for v in records),trusted_hole_pixels=sum(v['trusted_new_hole_pixels'] for v in records),
        definition='Development diagnostic only: alpha/appearance trust where original RGB relative squared error <= 0.25, normalized by target RGB energy plus one 8-bit quantization step squared OR original depth agrees within 3% with existing high-confidence VGGT; geometry trust requires the depth condition',
        scope='Does not alter any cleaner or accept a candidate. Photo MSE remains evaluated on the entire original foreground mask. Thresholds are fixed diagnostic proposals, not independently validated acceptance rules.',
        source_sha256=sha(Path(__file__).resolve()),seconds=time.perf_counter()-begin)
    write_json(legacy_destination or output/'RELATIVE_FOREGROUND_REFERENCE_TRUST_AUDIT.json',result);print(output,result['mean'],flush=True)


if __name__=='__main__':main()
