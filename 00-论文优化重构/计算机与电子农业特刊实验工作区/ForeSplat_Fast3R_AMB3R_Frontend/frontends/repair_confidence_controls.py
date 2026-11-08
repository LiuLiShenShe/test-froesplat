#!/usr/bin/env python3
"""Correct only derived q85 controls; preserve predictions, cameras and old PLY.

Confidence quantiles must be intersected with baseline confidence>1 validity.
Ties at the numerical floor do not imply a selective top15% control.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import csv
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from frontends.common import FrontendBundle,json_write,write_ply

def ply_count(path):
    if not Path(path).exists():return None
    with open(path,'rb') as f:
        for _ in range(32):
            line=f.readline()
            if line.startswith(b'element vertex '):return int(line.split()[-1])
            if line==b'end_header\n':break
    return None

def repair(path):
    bundle=FrontendBundle(path);path=bundle.path
    transforms=json.loads((path/'pixel_transforms.json').read_text())
    stats_path=path/'per_frame_metrics.csv'
    with open(stats_path) as f:rows=list(csv.DictReader(f))
    rows_by_name={r['name']:r for r in rows};points=[];colors=[];expected=0;audits=[]
    for name,t in zip(bundle.metadata['input_names'],transforms):
        stem=Path(name).stem
        with np.load(path/'raw_predictions'/f'{stem}.npz') as f:
            xyz=f['world_points'];conf=f['world_points_conf']
        mask=np.asarray(Image.open(path/'mask'/f'{stem}.png'))>0
        rgb=np.asarray(Image.open(path/'rgb'/f'{stem}.png'));xyz=xyz.reshape(*mask.shape,3);conf=conf.reshape(mask.shape)
        left,top,right,bottom=t['padding_ltrb'];v=np.zeros(mask.shape,bool);v[top:mask.shape[0]-bottom,left:mask.shape[1]-right]=True
        finite=v&np.isfinite(xyz).all(-1)&np.isfinite(conf);good=finite&(conf>1)
        q=float(np.quantile(conf[finite],.85)) if finite.any() else float('nan')
        selected=good&mask&(conf>=q);count=int(selected.sum());expected+=count
        points.append(xyz[selected]);colors.append(rgb[selected])
        row=rows_by_name[name];row['conf_q85']=q;row['mask_conf_q85_points']=count
        row['conf_q85_tie_fraction']=float(((conf==q)&finite).sum()/max(finite.sum(),1))
        row['conf_floor_fg_fraction']=float(((conf<=1)&finite&mask).sum()/max((finite&mask).sum(),1))
        row['conf_q85_retained_valid_fraction']=float((good&(conf>=q)).sum()/max(good.sum(),1))
        audits.append(dict(name=name,q85=q,baseline_valid_fg=int((good&mask).sum()),corrected_q85_fg=count,
            q85_tie_fraction=row['conf_q85_tie_fraction'],confidence_floor_fg_fraction=row['conf_floor_fg_fraction']))
    ply=path/'pointcloud'/'global_mask_confidence_q85.ply';actual=ply_count(ply)
    previous=None
    if actual!=expected:
        if ply.exists():
            previous=ply.with_name(f'legacy_q85_without_baseline_validity_{actual}_points.ply')
            if previous.exists():raise FileExistsError(f'preserving existing legacy control: {previous}')
            ply.rename(previous)
        temporary=ply.with_suffix('.tmp.ply');write_ply(temporary,np.concatenate(points),np.concatenate(colors));temporary.replace(ply)
    assert ply_count(ply)==expected
    before=path/'per_frame_metrics_before_q85_validity_fix.csv'
    if actual!=expected and not before.exists():before.write_bytes(stats_path.read_bytes())
    fields=list(dict.fromkeys(k for r in rows for k in r))
    temporary=stats_path.with_suffix('.tmp.csv')
    with open(temporary,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    temporary.replace(stats_path)
    meta=bundle.metadata
    meta['cloud_counts']['mask_confidence_q85']=expected
    meta['confidence_quantile']='per-frame q85 over finite unpadded pixels, intersected with baseline confidence>1; ties recorded; never reaccept floor-invalid points'
    audit=dict(policy='finite XYZ + unpadded mask + confidence>1 + confidence>=per-frame q85',
        previous_ply_points=actual,corrected_ply_points=expected,previous_ply=str(previous.relative_to(path)) if previous else None,
        raw_predictions_unchanged=True,cameras_unchanged=True,rgb_unchanged=True,masks_unchanged=True,
        no_confidence_value_changes=True,no_averaging_or_scale_changes=True,frames=audits)
    meta['metadata']['confidence_q85_validity_audit']='confidence_control_audit.json'
    json_write(path/'confidence_control_audit.json',audit);json_write(path/'bundle.json',meta)
    print(f'{path.name}: q85 control {actual}->{expected} points; raw data preserved',flush=True)
    return audit

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args();repair(args.bundle)

if __name__=='__main__':main()
