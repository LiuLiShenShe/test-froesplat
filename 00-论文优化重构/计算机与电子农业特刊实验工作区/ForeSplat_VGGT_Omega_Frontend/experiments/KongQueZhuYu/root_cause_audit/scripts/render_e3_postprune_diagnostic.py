#!/usr/bin/env python3
"""Compare existing E1/E3 actual RGB/alpha on exactly three common views, CPU only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from render_evaluate_common import psnr_region

AUDIT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=AUDIT/'render_error_analysis/e3_postprune_diagnostic_v1/comparison_with_E1')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    roots={'E1':AUDIT/'render_error_analysis/checkpoint_rerender_v1/E1',
           'E3':AUDIT/'render_error_analysis/e3_postprune_diagnostic_v1'}
    rows=[]
    for source in [0,95,189]:
        values={}
        for experiment,root in roots.items():
            with np.load(root/'views'/f'crop_{source:04d}'/'raw_render_outputs.npz') as raw:
                values[experiment]={key:raw[key] for key in ('gt_rgb','render_rgb','alpha','foreground_mask')}
        reference=values['E1'];gt=reference['gt_rgb'];fg=reference['foreground_mask'];boundary=ndimage.binary_dilation(fg,iterations=5)^ndimage.binary_erosion(fg,iterations=5)
        assert np.array_equal(gt,values['E3']['gt_rgb']) and np.array_equal(fg,values['E3']['foreground_mask'])
        rgb_panels=[gt];alpha_panels=[]
        for experiment,value in values.items():
            image=value['render_rgb'];alpha=value['alpha'];error=(image-gt)**2;binary=alpha>.5;tp=int((binary&fg).sum())
            row=dict(experiment=experiment,source_id=source,full_psnr=psnr_region(error,np.ones_like(fg)),
                fg_mask_psnr=psnr_region(error,fg),boundary_band5_psnr=psnr_region(error,boundary),
                foreground_alpha_mean=float(alpha[fg].mean()),background_alpha_mean=float(alpha[~fg].mean()),
                background_alpha_gt01_fraction=float((alpha[~fg]>.1).mean()),
                background_alpha_gt05_fraction=float((alpha[~fg]>.5).mean()),
                background_alpha_gt09_fraction=float((alpha[~fg]>.9).mean()),
                alpha_mask05_iou=tp/int((binary|fg).sum()),alpha_mask05_precision=tp/max(1,int(binary.sum())),
                alpha_mask05_recall=tp/int(fg.sum()),background_rgb_mean_abs_error=float(np.abs(image-gt)[~fg].mean()))
            rows.append(row);rgb_panels.append(image);alpha_panels.append(np.repeat(np.clip(alpha,0,1)[...,None],3,axis=2))
        for panels,label in ((rgb_panels,'GT_E1_E3_RGB'),(alpha_panels,'E1_E3_ALPHA')):
            canvas=Image.new('RGB',(len(panels)*270,480))
            for i,panel in enumerate(panels):
                image=Image.fromarray(np.round(np.clip(panel,0,1)*255).astype(np.uint8)).resize((270,480),Image.Resampling.LANCZOS);canvas.paste(image,(i*270,0))
            canvas.save(args.output/f'source_{source:04d}_{label}.png')
    with (args.output/'per_view_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    for experiment in ('E1','E3'):
        selected=[r for r in rows if r['experiment']==experiment];summary=dict(experiment=experiment,views=3)
        for key in selected[0]:
            if key not in ('experiment','source_id'):summary[key]=float(np.mean([r[key] for r in selected]))
        summaries.append(summary)
    (args.output/'diagnostic.json').write_text(json.dumps(dict(scope='3 common training views, existing checkpoint re-render; no training',
        E3_checkpoint='final post-prune PLY 1036619 Gaussians; saved iteration30000 PLY has1037020 and is a different artifact',
        source_ids=[0,95,189],GT_and_masks_identical=True,summaries=summaries,
        limitation='This3-view diagnostic does not replace old208-view E3 evaluation or establish a preferred mask-loss weight.'),indent=2,allow_nan=False))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
