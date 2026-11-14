#!/usr/bin/env python3
"""Diagnose candidate grouping using native opacity gradients, without relaxing quality."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import native_subset


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--validate',action='store_true',help='Actually rerender every camera for the proposed nonpositive-risk subset')
    args=parser.parse_args();torch.set_num_threads(8)
    evidence=json.loads(args.evidence.read_text());run=Path(evidence['run']);selected=evidence['decision']
    reference=json.loads((run/'original_reference.json').read_text())
    if reference['variant']!='C5-C' or sha(reference['scene']['model'])!=evidence['input_sha256']:raise ValueError('Invalid unmodified-parameter input')
    round_dir=run/'rounds'/evidence['round']
    with np.load(round_dir/'evidence.npz') as z:
        start_ids=z['original_ids'].copy();bg=z['bg'].copy()+z['free_bg'].copy()
    decisions=[json.loads(p.read_text()) for p in round_dir.glob('group_*/decision.json')]
    removed={g for d in decisions if d['accepted'] and d['group']<selected['group'] for g in d['removed_ids']}
    ids=start_ids[~np.isin(start_ids,list(removed))]
    scene=Scene(reference['scene'],reference['config']);original,_=scene.load_model(scene.spec['model'])
    keep=np.zeros(len(original.get_xyz),bool);keep[ids]=True
    model=native_subset(scene,original,keep)
    group=np.asarray(selected['removed_ids'],np.int64);local=np.flatnonzero(np.isin(ids,group))
    if len(local)!=len(group):raise ValueError('Exact group reconstruction failed')
    tensor_ids=torch.as_tensor(local,device='cuda');base=model._opacity.detach().clone();raw_alpha=torch.sigmoid(base)
    risks=[];observations=[]
    for record in evidence['views']:
        camera,target,mask,*_=scene.get_view(record['index'])
        dark=mask&((target@np.array([.2126,.7152,.0722]))<.12)
        roi=dark if any('dark_leaf_mse' in rule for rule in record['rejection_rules']) and dark.any() else mask
        target_tensor=torch.as_tensor(target.transpose(2,0,1).copy(),device='cuda')
        mask_tensor=torch.as_tensor(roi,device='cuda')
        per_view=[]
        for attenuation in [1.,.5]:
            opacity=base.clone()
            opacity[tensor_ids]=torch.logit((raw_alpha[tensor_ids]*attenuation).clamp(1e-7,1-1e-7))
            model._opacity=opacity.requires_grad_(True)
            render=scene.render(camera,model)['render']
            loss=((render-target_tensor)**2)[:,mask_tensor].mean()
            grad=torch.autograd.grad(loss,model._opacity)[0].flatten()
            score=(-grad[tensor_ids]/(1-torch.sigmoid(model._opacity.flatten()[tensor_ids])).clamp_min(1e-4)).detach().cpu().numpy()
            if not np.isfinite(score).all():raise ValueError('Nonfinite native opacity gradient')
            per_view.append(score);observations.append(dict(view=camera.image_name,attenuation=attenuation,loss=float(loss.detach()),maximum_risk=float(score.max())))
            model._opacity=base
            print(camera.image_name,attenuation,float(loss.detach()),'damaging',int((score>1e-10).sum()),flush=True)
        risks.append(np.maximum.reduce(per_view))
    risk=np.maximum.reduce(risks);safe=risk<=1e-10
    position={int(g):i for i,g in enumerate(start_ids)}
    bg_local=np.asarray([bg[position[int(g)]] for g in ids[local]])
    rows=[dict(original_id=int(g),worst_photo_risk=float(value),predicted_background=float(background),nonpositive_photo_risk=bool(allowed)) for g,value,background,allowed in zip(ids[local],risk,bg_local,safe)]
    output=args.evidence.parent
    with (output/'PHOTOMETRIC_SPLIT_RISKS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.save(output/'diagnostic_nonpositive_risk_original_ids.npy',ids[local][safe])
    write_json(output/'PHOTOMETRIC_SPLIT_DIAGNOSTIC.json',dict(candidate_count=len(local),nonpositive_risk_count=int(safe.sum()),
        nonpositive_risk_predicted_background=float(bg_local[safe].sum()),total_predicted_background=float(bg_local.sum()),observations=observations,
        scope='Native gradients at original and half local opacity estimate a diagnostic grouping. No candidate is accepted, no PLY is modified, and all-view nonlinear trial validation is still required. Quality constraints are unchanged.',
        source_sha256=sha(Path(__file__).resolve())))
    if args.validate and safe.any():
        import lpips
        from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.evaluation import Evaluator
        from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import choose
        from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
        surface=external_surface(scene,scene.evidence_indices())
        evaluator=Evaluator(scene,output/'nonlinear_validation',None,lpips.LPIPS(net='alex').cuda().eval(),surface)
        initial=evaluator.evaluate(original,output/'nonlinear_validation/original')
        current=evaluator.evaluate(model,output/'nonlinear_validation/before')
        candidate_keep=np.ones(len(ids),bool);candidate_keep[local[safe]]=False
        candidate=native_subset(scene,model,candidate_keep)
        measured=evaluator.evaluate(candidate,output/'nonlinear_validation/proposed')
        accepted,gain,local_gain,reasons=choose(measured,current,initial,reference['config'])
        write_json(output/'PHOTOMETRIC_SPLIT_FULL_VALIDATION.json',dict(accepted=accepted,candidates=int(safe.sum()),
            measured_gain=gain,local_gain=local_gain,reasons=reasons,initial=initial,current=current,proposed=measured,
            input_sha256=evidence['input_sha256'],candidate_original_ids=ids[local][safe],
            scope='Actual all-camera nonlinear trial on the exact reconstructed parent context. A diagnostic group result, not a completed new whole-scene cleaner or cross-plant validation.',
            source_sha256=sha(Path(__file__).resolve())))
        print('full nonlinear validation',accepted,gain,reasons,flush=True)


if __name__=='__main__':main()
