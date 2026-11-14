#!/usr/bin/env python3
"""Independent native GGGS controls separating scale, filter opacity and V3 additions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply,sha,write_json
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.legacy import features,rules
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.artifacts import export,parity,visuals
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import evaluate


def encode(masks):
    bits=np.zeros(len(next(iter(masks.values()))),np.uint16);stats={};assigned=np.zeros(len(bits),bool)
    for i,(name,mask) in enumerate(masks.items()):
        bits|=mask.astype(np.uint16)<<i
        stats[name]=dict(overlapping_count=int(mask.sum()),exclusive_first_reason_count=int((mask&~assigned).sum()))
        assigned|=mask
    return assigned,bits,stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    args=parser.parse_args();b=args.baseline_root.resolve()
    reference=json.loads((b/'C0/cleanup_metrics.json').read_text());scene=Scene(reference['scene'],reference['config'])
    if sha(scene.spec['model'])!=reference['input_sha256']:raise ValueError('Changed untouched GGGS input')
    original=read_ply(scene.spec['model']);f=features(original);torch.set_num_threads(8)
    import lpips
    lp=lpips.LPIPS(net='alex').cuda().eval();surface=external_surface(scene,scene.evidence_indices())
    scale_f={**f,'effective_alpha':f['raw_alpha']}
    c1,bits,detail=rules(scale_f,2,True)
    cases=[('C1_scale_only',c1,bits,{**detail,
        'alpha_encoding':'Historical sigmoid(logit opacity) unchanged; only scale-related rules use filtered three-axis sizes and scene quantiles',
        'isolation':'C1 raw versus this control isolates the GGGS scale/relative-threshold adaptation; this control versus C1_adaptive isolates use of effective filter opacity.'})]
    _,base_bits,base_detail=rules(f,2,False)
    base_masks={name:(base_bits&(1<<i))!=0 for i,name in enumerate(base_detail['reasons'])}
    median=max(float(np.median(f['local_size_ratio'])),1e-12)
    tier=np.zeros(len(original),bool)
    for multiplier,floor in [(1.,.15),(1.5,.20),(2.5,.28)]:tier|=(f['local_size_ratio']>median*multiplier)&(f['brightness']<floor)
    raw_tier=np.zeros(len(original),bool);median_raw=max(float(np.median(f['scale_max'])),1e-12)
    for multiplier,floor in [(1.,.15),(1.5,.20),(2.5,.28)]:raw_tier|=(f['scale_max']>median_raw*multiplier)&(f['brightness']<floor)
    for name,tier_mask,knn,description in [
        ('C2_normalized_tier_only',tier,f['knn_original'],'Faithful V2 base and original K20 dark test remain; only V3 scale-brightness tier uses normalized three-axis filtered scale/local spacing.'),
        ('C2_normalized_knn_only',raw_tier,f['knn_normalized'],'Faithful V2 base and original three-axis V3 tier remain; only K20 neighborhood uses Gaussian-size/spacing normalized radius with >=5 valid neighbors.')]:
        removed,bits,stats=encode({**base_masks,'scale_brightness':tier_mask,'knn_dark':knn})
        cases.append((name,removed,bits,dict(reasons=stats,isolation=description,thresholds=base_detail['thresholds'],
            color_definition=base_detail['color_definition'],alpha_encoding='Historical raw sigmoid(logit opacity); native renderer still applies its stored 3D filter')))
    for name,removed,bits,detail in cases:
        out=b/name
        if out.exists():raise FileExistsError(out)
        begin=time.perf_counter();out.mkdir();ids=np.flatnonzero(~removed);rows=original[~removed]
        export(out,original,rows,ids);np.savez(out/'deletion_reasons.npz',original_ids=np.arange(len(original)),reason_bits=bits,removed=removed)
        model,_=scene.load_model(out/'cleaned_gaussians.ply');torch.cuda.reset_peak_memory_stats()
        measurements=evaluate(scene,model,out/'evaluation',reference['evaluation_baseline'],lp,save_images=True)
        parity(scene,model,out);visuals(scene,model,out/'visualizations',surface)
        report=dict(method=name,config=reference['config'],scene=scene.spec,input_model=scene.spec['model'],input_sha256=reference['input_sha256'],
            original_gaussians=len(original),cleaned_gaussians=len(rows),removed_gaussians=int(removed.sum()),
            seconds=time.perf_counter()-begin,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,metrics=measurements,
            native_ply=str((out/'cleaned_gaussians.ply').resolve()),supersplat_ply=str((out/'cleaned_gaussians_supersplat.ply').resolve()),
            removed_ply=str((out/'removed_gaussians.ply').resolve()),isolation=detail,
            timing_scope='Shared feature discovery computed once; reported time covers per-control export/native evaluation/parity/visuals, not the shared feature pass',
            source_sha256=sha(Path(__file__).resolve()))
        write_json(out/'cleanup_metrics.json',report);write_json(out/'COMPLETE.json',dict(input_sha256=report['input_sha256'],native_sha256=sha(out/'cleaned_gaussians.ply')))
        print(name,removed.sum(),measurements['mean']['fg_psnr'],flush=True)
        del model


if __name__=='__main__':main()
