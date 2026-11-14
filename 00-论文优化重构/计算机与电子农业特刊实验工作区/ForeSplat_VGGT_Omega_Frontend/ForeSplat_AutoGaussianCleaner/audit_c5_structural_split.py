#!/usr/bin/env python3
"""Use exact native alpha derivatives to subdivide an actually rejected photo-risk group."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import native_subset
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays


def alpha_image(camera,model):
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    settings=GaussianRasterizationSettings(image_height=camera.image_height,image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx/2),tanfovy=math.tan(camera.FoVy/2),kernel_size=0.,bg=torch.zeros(3,device='cuda'),
        scale_modifier=1.,viewmatrix=camera.world_view_transform,projmatrix=camera.full_proj_transform,
        sh_degree=model.active_sh_degree,sg_degree=model.active_sg_degree,campos=camera.camera_center,prefiltered=False,require_depth=True,debug=False)
    scales,opacity=model.get_scaling_n_opacity_with_3D_filter
    out=GaussianRasterizer(raster_settings=settings)(means3D=model.get_xyz.detach(),means2D=torch.zeros_like(model.get_xyz),
        shs=None,sg_axis=model.get_sg_axis.detach(),sg_sharpness=model.get_sg_sharpness.detach(),sg_color=model.get_sg_color.detach(),
        colors_precomp=torch.ones((len(model.get_xyz),3),device='cuda'),opacities=opacity,scales=scales.detach(),
        rotations=model.get_rotation.detach(),cov3Ds_precomp=None)
    return out[0].mean(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--validate',action='store_true');args=parser.parse_args();torch.set_num_threads(8)
    evidence=json.loads((args.audit/'PIXEL_REJECTION_EVIDENCE.json').read_text())
    trial=json.loads((args.audit/'PHOTOMETRIC_SPLIT_FULL_VALIDATION.json').read_text());run=Path(evidence['run'])
    reference=json.loads((run/'original_reference.json').read_text())
    if reference['variant']!='C5-C' or sha(reference['scene']['model'])!=evidence['input_sha256']:raise ValueError('Invalid original context')
    folder=run/'rounds'/evidence['round']
    with np.load(folder/'evidence.npz') as z:round_ids=z['original_ids'].copy();bg=z['bg'].copy()+z['free_bg'].copy()
    decisions=[json.loads(p.read_text()) for p in folder.glob('group_*/decision.json')]
    earlier={g for d in decisions if d['accepted'] and d['group']<evidence['decision']['group'] for g in d['removed_ids']}
    ids=round_ids[~np.isin(round_ids,list(earlier))]
    if hashlib.sha256(ids.tobytes()).hexdigest()!=evidence['before_original_ids_sha256']:raise ValueError('Exact model context differs')
    scene=Scene(reference['scene'],reference['config']);original,_=scene.load_model(scene.spec['model'])
    keep=np.zeros(len(original.get_xyz),bool);keep[ids]=True;model=native_subset(scene,original,keep)
    group=np.asarray(trial['candidate_original_ids'],np.int64);local=np.flatnonzero(np.isin(ids,group))
    keep_candidate=~np.isin(ids,group);candidate=native_subset(scene,model,keep_candidate)
    base=model._opacity.detach().clone();raw=torch.sigmoid(base);ti=torch.as_tensor(local,device='cuda')
    dimensions=[];risks=[]
    names={reason.split(':',1)[0] for reason in trial['reasons'] if 'boundary_coverage' in reason or 'dark_leaf_alpha' in reason}
    for index,view in enumerate(scene.views):
        if view['name'] not in names:continue
        camera,target,mask,*_=scene.get_view(index);r=regions(mask,scene.config['boundary_relative_width'])
        with torch.no_grad():b=arrays(scene.render(camera,model));a=arrays(scene.render(camera,candidate))
        old=next(v for v in trial['initial']['views'] if v['name']==camera.image_name)
        view_reasons=[reason for reason in trial['reasons'] if reason.startswith(camera.image_name+':')]
        supports=[]
        if any('dark_leaf_alpha' in reason for reason in view_reasons):
            dark=mask&((target@np.array([.2126,.7152,.0722]))<.12)
            budget=max(0.,(float(b['alpha'][dark].mean())-old['dark_foreground_alpha']+reference['config']['quality']['view_foreground_alpha_drop'])*dark.sum())
            supports.append(('dark_alpha',dark,budget))
        if any('boundary_coverage' in reason for reason in view_reasons):
            lost=r['boundary']&(b['alpha']>.5)&(a['alpha']<=.5)
            if lost.any():supports.append(('boundary_hole',lost,float((b['alpha'][lost]-.5).sum())*.5))
        for label,support,budget in supports:
            scores=[]
            for attenuation in [1.,.5,.05]:
                value=base.clone();sample=(raw[ti]*attenuation).clamp(1e-7,1-1e-7);value[ti]=torch.logit(sample)
                model._opacity=value.requires_grad_(True)
                image=alpha_image(camera,model);loss=image[torch.as_tensor(support,device='cuda')].sum()
                gradient=torch.autograd.grad(loss,model._opacity)[0][ti]
                estimate=(gradient*raw[ti]/(sample*(1-sample)).clamp_min(1e-8)).flatten().detach().cpu().numpy()
                if not np.isfinite(estimate).all():raise ValueError('Nonfinite structural opacity gradient')
                scores.append(np.maximum(estimate,0));model._opacity=base
            score=np.maximum.reduce(scores);risks.append(score);dimensions.append(dict(view=camera.image_name,proxy=label,budget=budget,pixels=int(support.sum())))
            print(camera.image_name,label,'budget',budget,'total_estimated_loss',float(score.sum()),flush=True)
    if not dimensions:raise ValueError('No structural rejecting proxy')
    risk=np.stack(risks);budgets=np.asarray([item['budget'] for item in dimensions])
    positions={int(g):i for i,g in enumerate(round_ids)};benefit=np.asarray([bg[positions[int(g)]] for g in ids[local]])
    ratio=risk/np.maximum(budgets[:,None],1e-10)
    order=np.argsort(-benefit/(1+ratio.max(0)),kind='stable');chosen=[];used=np.zeros(len(budgets))
    for position in order:
        if np.all(used+risk[:,position]<=budgets+1e-10):chosen.append(position);used+=risk[:,position]
    chosen=np.asarray(chosen,np.int64);accepted_ids=ids[local][chosen]
    rows=[dict(original_id=int(g),predicted_background=float(benefit[i]),worst_budget_ratio=float(ratio[:,i].max()),selected=bool(i in chosen)) for i,g in enumerate(ids[local])]
    with (args.audit/'STRUCTURAL_SPLIT_RISKS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(args.audit/'STRUCTURAL_SPLIT_DIAGNOSTIC.json',dict(original_candidates=len(local),selected=len(chosen),selected_ids=accepted_ids,
        predicted_background=float(benefit[chosen].sum()),dimensions=dimensions,estimated_used=used,
        scope='Automatically derived rejecting proxy supports and native alpha gradients partition an existing trial. Quality thresholds unchanged; no PLY modified. Gradient bounds require full nonlinear validation.',source_sha256=sha(Path(__file__).resolve())))
    if args.validate and len(chosen):
        import lpips
        from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.evaluation import Evaluator
        from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import choose
        from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
        destination=args.audit/'structural_validation';surface=external_surface(scene,scene.evidence_indices())
        evaluator=Evaluator(scene,destination,args.audit/'nonlinear_validation/original',lpips.LPIPS(net='alex').cuda().eval(),surface)
        selected_model=native_subset(scene,model,~np.isin(ids,accepted_ids))
        measured=evaluator.evaluate(selected_model,destination/'proposed')
        accepted,gain,local_gain,reasons=choose(measured,trial['current'],trial['initial'],reference['config'])
        write_json(args.audit/'STRUCTURAL_SPLIT_FULL_VALIDATION.json',dict(accepted=accepted,candidates=len(chosen),measured_gain=gain,local_gain=local_gain,
            reasons=reasons,proposed=measured,original_ids=accepted_ids,input_sha256=evidence['input_sha256'],
            scope='Actual all-view nonlinear trial with all original C8 quality constraints; initial/current metrics reused only from the exact SHA-verified model and unchanged scene context. Not a whole-scene C5 or cross-plant result.',source_sha256=sha(Path(__file__).resolve())))
        print('structural full validation',accepted,gain,reasons,flush=True)


if __name__=='__main__':main()
