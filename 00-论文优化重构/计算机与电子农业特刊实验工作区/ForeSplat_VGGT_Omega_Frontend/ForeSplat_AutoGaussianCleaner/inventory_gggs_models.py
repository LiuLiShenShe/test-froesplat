#!/usr/bin/env python3
"""Audit available original GGGS inputs and preparation costs without rendering or tuning."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from plyfile import PlyData
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.auto_clean_batch import model_ready


def audit(path):
    rows= PlyData.read(str(path),mmap='r')['vertex'].data
    names=rows.dtype.names
    required=['x','y','z','opacity','filter_3D']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_dc_{i}' for i in range(3)]
    if not all(k in names for k in required):raise ValueError('Not a native three-scale GGGS model')
    if not len(rows):raise ValueError('Empty original model')
    for start in range(0,len(rows),250000):
        block=rows[start:start+250000]
        for name in names:
            if not np.isfinite(block[name]).all():raise ValueError(f'Nonfinite original field {name}')
        if np.any(block['filter_3D']<0):raise ValueError('Negative native filter')
        scales=np.exp(np.column_stack([block[f'scale_{i}'] for i in range(3)]).astype(np.float64))
        if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Invalid encoded native scale')
        quaternion=np.column_stack([block[f'rot_{i}'] for i in range(4)])
        if np.any(np.linalg.norm(quaternion,axis=1)<1e-10):raise ValueError('Zero native rotation')
    rest=sum(k.startswith('f_rest_') for k in names)
    degree=int(round(np.sqrt(rest/3+1)-1))
    if rest!=3*((degree+1)**2-1):raise ValueError('Incomplete SH coefficient schema')
    sg=sum(k.startswith('sg_sharpness_') for k in names)
    if sg:raise ValueError('Selected equivalent SuperSplat workflow requires SG degree zero')
    return dict(gaussians=len(rows),sh_degree=degree,sg_degree=sg,fields=len(names),native_sha256=sha(path),schema_valid=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());rows=[]
    base=Path(__file__).resolve().parent/'experiments/GGGS_Cleaner_Comparison'
    pause_file=base/'comparisons/TRAINING_SCHEDULING_COST_AUDIT.json'
    pause=json.loads(pause_file.read_text()) if pause_file.exists() else {}
    preparation=base/'all_19_plants/model_preparation/TRAINING_STATUS.json'
    statuses=json.loads(preparation.read_text()) if preparation.exists() else {}
    for job in manifest['plants']:
        scene=json.loads(Path(job['scene']).read_text());path=Path(scene['model'])
        row=dict(plant=job['plant'],group=job['group'],status='missing_or_incomplete_gggs_checkpoint',native_ply=str(path.resolve()),
            scene=str(Path(job['scene']).resolve()),gaussians='',sh_degree='',sg_degree='',fields='',native_sha256='',schema_valid=False,
            preparation_status=statuses.get(job['plant'],{}).get('status','not_observed'),
            training_wall_seconds='',documented_training_pause_seconds=0,training_wall_excluding_documented_pause='',
            training_peak_allocated_gib='',training_peak_reserved_gib='',training_report='')
        if model_ready(job,scene):
            try:row.update(audit(path));row['status']='verified_original_gggs_checkpoint'
            except (OSError,ValueError,KeyError) as error:row['status']='invalid_original_gggs_checkpoint';row['reason']=str(error)
        training=Path(job['training_dir'])/'densification_events.complete.json' if job.get('training_dir') else None
        if training and training.exists():
            cost=json.loads(training.read_text());wall=cost['wall_seconds']
            seconds=pause.get('documented_scheduler_pause_seconds',0) if pause.get('plant')==job['plant'] else 0
            row.update(training_wall_seconds=wall,documented_training_pause_seconds=seconds,
                training_wall_excluding_documented_pause=wall-seconds,training_peak_allocated_gib=cost.get('peak_allocated_gib',''),
                training_peak_reserved_gib=cost.get('peak_reserved_gib',''),training_report=str(training.resolve()))
        rows.append(row);print(job['plant'],row['status'],row['gaussians'],flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with (args.output/'ORIGINAL_GGGS_INPUT_INVENTORY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    write_json(args.output/'ORIGINAL_GGGS_INPUT_INVENTORY.json',dict(plants=rows,
        verified=sum(row['status']=='verified_original_gggs_checkpoint' for row in rows),expected=len(rows),
        scope='Input format, completeness and preparation costs only. No test-plant appearance statistics, cleaning decisions or threshold tuning. Historical training costs are reported separately from new preparation status.'))


if __name__=='__main__':main()
