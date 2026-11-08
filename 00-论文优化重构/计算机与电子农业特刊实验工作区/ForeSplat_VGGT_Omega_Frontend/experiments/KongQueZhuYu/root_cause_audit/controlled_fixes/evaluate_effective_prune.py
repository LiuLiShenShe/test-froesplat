#!/usr/bin/env python3
"""CPU same-view evaluation of recorded baseline vs effective-opacity deletion."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

AUDIT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(AUDIT/'scripts'))
from render_evaluate_common import psnr_region,gradients


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=('E1','E5'),required=True)
    args=parser.parse_args();mode=args.experiment
    baseline=AUDIT/'render_error_analysis/checkpoint_rerender_v1'/mode
    control=AUDIT/'controlled_fixes/prune_effective_render'/mode
    out=control/'evaluation'
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    # Validate that actual runtime cameras, including all projection matrices, were held fixed.
    with np.load(baseline/'camera_runtime.npz') as left,np.load(control/'camera_runtime.npz') as right:
        assert set(left.files)==set(right.files)
        for key in left.files:
            a=left[key];b=right[key]
            same=np.array_equal(a,b,equal_nan=True) if np.issubdtype(a.dtype,np.number) else np.array_equal(a,b)
            if not same:raise ValueError(f'Runtime Camera changed in control: {key}')
    rows=[]
    for source in [0,95,189]:
        values={}
        for kind,root in [('baseline',baseline),('effective_pruned',control)]:
            with np.load(root/'views'/f'crop_{source:04d}'/'raw_render_outputs.npz') as raw:
                values[kind]={key:raw[key] for key in ['gt_rgb','render_rgb','alpha','foreground_mask']}
        gt=values['baseline']['gt_rgb'];fg=values['baseline']['foreground_mask']
        assert np.array_equal(gt,values['effective_pruned']['gt_rgb'])
        assert np.array_equal(fg,values['effective_pruned']['foreground_mask'])
        boundary=ndimage.binary_dilation(fg,iterations=5)^ndimage.binary_erosion(fg,iterations=5)
        gg=gradients(gt);panels=[gt];alpha_panels=[];edge_panels=[]
        for kind,value in values.items():
            image=value['render_rgb'];alpha=value['alpha'];error=(image-gt)**2;aa=alpha>.5;tp=int((aa&fg).sum());ig=gradients(image)
            row=dict(experiment=mode,source_id=source,condition=kind,
                full_psnr=psnr_region(error,np.ones_like(fg)),fg_mask_psnr=psnr_region(error,fg),
                boundary_band5_psnr=psnr_region(error,boundary),inner_fg_edge5_psnr=psnr_region(error,fg&boundary),
                foreground_alpha_mean=float(alpha[fg].mean()),background_alpha_mean=float(alpha[~fg].mean()),
                background_alpha_gt05_fraction=float((alpha[~fg]>.5).mean()),
                alpha_mask05_iou=tp/int((aa|fg).sum()),alpha_mask05_precision=tp/max(1,int(aa.sum())),alpha_mask05_recall=tp/int(fg.sum()),
                background_rgb_mean_abs_error=float(np.abs(image-gt)[~fg].mean()),
                foreground_gradient_retention=float(ig[fg].mean()/gg[fg].mean()),
                inner_edge_gradient_retention=float(ig[fg&boundary].mean()/gg[fg&boundary].mean()),
                foreground_gradient_mae=float(np.abs(ig-gg)[fg].mean()))
            rows.append(row);panels.append(image);alpha_panels.append(np.repeat(np.clip(alpha,0,1)[...,None],3,axis=2));edge_panels.append(np.repeat((np.abs(image-gt).mean(axis=2)*boundary/.5)[...,None],3,axis=2))
        for imgs,label in [(panels,'GT_baseline_effective_pruned'),(alpha_panels,'baseline_effective_pruned_alpha'),(edge_panels,'baseline_effective_pruned_edge_error')]:
            canvas=Image.new('RGB',(len(imgs)*270,480))
            for i,image in enumerate(imgs):
                picture=Image.fromarray(np.round(np.clip(image,0,1)*255).astype(np.uint8)).resize((270,480),Image.Resampling.LANCZOS);canvas.paste(picture,(i*270,0))
            canvas.save(out/f'source_{source:04d}_{label}.png')
    summaries=[]
    for kind in ['baseline','effective_pruned']:
        selected=[r for r in rows if r['condition']==kind];summary=dict(experiment=mode,condition=kind,views=3)
        for key in selected[0]:
            if key not in ['experiment','source_id','condition']:summary[key]=float(np.mean([r[key] for r in selected]))
        summaries.append(summary)
    deltas={key:summaries[1][key]-summaries[0][key] for key in summaries[0] if key not in ['experiment','condition','views']}
    with (out/'per_view_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    provenance=json.loads((control/'render_run.json').read_text())
    report=dict(experiment=mode,status='complete',source_ids=[0,95,189],single_factor='remove native effective-opacity<0.05 Gaussian rows',
        new_training=False,all208_runtime_camera_arrays_exactly_equal=True,GT_and_masks_exactly_equal=True,
        retained_parameter_selection=provenance['point_subset_control'],checkpoint_sha256=provenance['checkpoint_sha256'],
        summaries=summaries,deltas_effective_pruned_minus_baseline=deltas,
        metrics_scope='CPU same-mask float32 RGB, no LPIPS/new GPU model job; baseline subset is exactly same three source IDs',
        limitation='Three existing training views test this deletion rule only. Results do not establish retraining or generalization gains.')
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(dict(experiment=mode,summaries=summaries,deltas=deltas),indent=2),flush=True)


if __name__=='__main__':main()
