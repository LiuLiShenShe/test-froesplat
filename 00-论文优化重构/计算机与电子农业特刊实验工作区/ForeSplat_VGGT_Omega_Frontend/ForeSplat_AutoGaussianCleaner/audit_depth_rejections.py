#!/usr/bin/env python3
"""Distinguish original-depth changes from improvement toward existing VGGT depth."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import json
import numpy as np
import torch
import yaml
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import native_subset
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions


def main():
    base=Path(__file__).resolve().parent;c=base/'experiments/GGGS_Cleaner_Comparison'
    cfg=yaml.safe_load((base/'configs/default.yaml').read_text());spec=json.loads((base/'configs/scenes/KongQueZhuYu.json').read_text())
    scene=Scene(spec,cfg);torch.set_num_threads(8);original,rows=scene.load_model(spec['model'])
    legacy=base/'experiments/all_19_plants_v3/development/KongQueZhuYu/C'
    item=next(i for i in json.loads((legacy/'edit_manifest.json').read_text())['edits'] if i['action']=='prune' and len(i['removed_ids'])==31955)
    names={reason.split(':')[0] for reason in item['reasons'] if 'depth' in reason and ':' in reason}
    keep=np.ones(len(rows),bool);keep[np.asarray(item['removed_ids'],np.int64)]=False
    proposal=native_subset(scene,original,keep);history,_=scene.load_model(c/'baselines/C3_pre_marker/cleaned_gaussians.ply')
    records=[]
    for index,v in enumerate(scene.views):
        if v['name'] not in names:continue
        co,_,mask,depth,confidence=scene.get_view(index)
        valid=mask&np.isfinite(depth)&(depth>0)&np.isfinite(confidence)
        valid &= confidence>=np.quantile(confidence[valid],.2)
        unit=float(np.median(depth[valid]));r=regions(mask,cfg['boundary_relative_width'])
        with torch.no_grad():before=arrays(scene.render(co,original))
        for label,model in [('C4_full_rejected',proposal),('C3_before_marker',history)]:
            with torch.no_grad():after=arrays(scene.render(co,model))
            shared=valid&(before['alpha']>.5)&(after['alpha']>.5)&(before['depth']>0)&(after['depth']>0)
            error_before=np.abs(before['depth']-depth)/unit;error_after=np.abs(after['depth']-depth)/unit
            changed=shared&(np.abs(after['depth']-before['depth'])/unit>cfg['quality']['depth_relative_p95'])
            m=dict(candidate=label,index=index,view=v['name'],reference_pixels=int(shared.sum()),changed_pixels=int(changed.sum()),
                original_external_mae=float(error_before[shared].mean()),candidate_external_mae=float(error_after[shared].mean()),
                original_external_p95=float(np.quantile(error_before[shared],.95)),candidate_external_p95=float(np.quantile(error_after[shared],.95)),
                changed_toward_external_fraction=float((error_after[changed]<error_before[changed]).mean()) if changed.any() else None,
                changed_external_mae_before=float(error_before[changed].mean()) if changed.any() else None,
                changed_external_mae_after=float(error_after[changed].mean()) if changed.any() else None,
                changed_boundary_pixels=int((changed&r['boundary']).sum()),changed_thin_pixels=int((changed&r['thin']).sum()))
            records.append(m)
    write_json(c/'comparisons/DEPTH_REJECTION_EXTERNAL_REFERENCE_AUDIT.json',dict(records=records,input_sha256=sha(spec['model']),
        full_proposal_gaussians_removed=len(item['removed_ids']),rejected_views_automatically_selected=sorted(names),
        reference='Already inferred VGGT depth/confidence through the corrected scene interface; auxiliary reference, not botanical ground truth',
        scope='Measured development-scene diagnostic only; neither a new acceptance rule nor cleanup success'))
    print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':main()
