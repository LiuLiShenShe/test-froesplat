#!/usr/bin/env python3
"""Distinguish alpha loss from removal of demonstrably valid foreground geometry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from scipy import ndimage
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.contribution_analysis import contribution
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays


def main():
    p=Path(__file__).resolve().parent/'experiments/GGGS_Cleaner_Comparison'
    source=p/'baselines/C0';cleaned=p/'baselines/C3_pre_marker'
    reference=json.loads((source/'cleanup_metrics.json').read_text());scene=Scene(reference['scene'],reference['config'])
    index=next(i for i,v in enumerate(scene.views) if v['name']=='crop_0035')
    camera,target,mask,depth,confidence=scene.get_view(index)
    original,rows=scene.load_model(scene.spec['model']);candidate,_=scene.load_model(cleaned/'cleaned_gaussians.ply')
    with torch.no_grad():before=arrays(scene.render(camera,original));after=arrays(scene.render(camera,candidate))
    lost=mask&(before['alpha']>.5)&(after['alpha']<=.5)
    labels,_=ndimage.label(lost);sizes=np.bincount(labels.ravel())[1:];region=labels==sizes.argmax()+1
    valid=mask&np.isfinite(depth)&(depth>0)&np.isfinite(confidence)
    cutoff=float(np.quantile(confidence[valid],.2));supported=region&valid&(confidence>=cutoff)
    unit=max(float(np.median(depth[valid])),1e-9)
    error=np.abs(before['depth'][supported]-depth[supported])/unit
    weights,_,_=contribution(camera,original,np.stack([region,mask&~region,~mask]))
    removed=np.ones(len(rows),bool);removed[np.load(cleaned/'original_gaussian_ids.npy')]=False
    contribution_weights=weights[:,0]/max(weights[:,0].sum(),1e-12)
    top=np.argsort(-weights[:,0])[:100]
    with np.load(p/'C5_Fused_GGGS/development_v8_1_verified/C5-B/rounds/01/evidence.npz') as evidence:
        ratio=evidence['bg']/(evidence['fg']+evidence['bg']+1e-15)
        stats=dict(top_100_removed=int(removed[top].sum()),
            top_100_original_bg_fraction_weighted=float(np.sum(ratio[top]*contribution_weights[top])/max(contribution_weights[top].sum(),1e-12)),
            region_contribution_from_gaussians_supported_in_2_depth_views=float(contribution_weights[evidence['geometry_views']>=2].sum()),
            removed_region_contribution_depth_supported=float(contribution_weights[removed&(evidence['geometry_views']>=2)].sum()),
            region_contribution_from_all_208_foreground_weighted=float(np.sum(evidence['fg']*contribution_weights)),
            top_original_ids=top.tolist())
    lum=np.array([.2126,.7152,.0722])
    result=dict(view='crop_0035',index=index,region_pixels=int(region.sum()),
        original_alpha=float(before['alpha'][region].mean()),cleaned_alpha=float(after['alpha'][region].mean()),
        target_rgb_mean=target[region].mean(0).tolist(),original_rgb_mean=before['rgb'][region].mean(0).tolist(),cleaned_rgb_mean=after['rgb'][region].mean(0).tolist(),
        target_luminance=float((target[region]@lum).mean()),original_luminance=float((before['rgb'][region]@lum).mean()),cleaned_luminance=float((after['rgb'][region]@lum).mean()),
        original_target_mse=float(((before['rgb'][region]-target[region])**2).mean()),cleaned_target_mse=float(((after['rgb'][region]-target[region])**2).mean()),
        high_confidence_external_depth_supported_fraction=float(supported.sum()/max(region.sum(),1)),
        original_depth_external_mae=float(error.mean()) if len(error) else None,
        original_depth_external_p95=float(np.quantile(error,.95)) if len(error) else None,
        original_depth_within_3_percent_fraction=float((error<=.03).mean()) if len(error) else None,
        region_alpha_contribution_removed_by_C3=float(weights[removed,0].sum()/max(weights[:,0].sum(),1e-12)),
        gaussian_multiview_evidence=stats,
        scope='Automatic largest new foreground alpha hole in the C3 worst thin-coverage camera; diagnostic evaluation only, never cleaner ROI/IDs',
        limitations='Existing confidence-filtered VGGT depth and Gaussian depth-support tests are auxiliary proxies; alpha coverage alone does not prove a leaf Gaussian existed')
    write_json(p/'comparisons/FOREGROUND_HOLE_NATIVE_EVIDENCE.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
