#!/usr/bin/env python3
"""Compare every reused original alpha/depth/normal array against untouched native GGGS."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();ref=json.loads(args.reference.read_text())
    if sha(ref['scene']['model'])!=ref['input_sha256']:raise ValueError('Changed untouched native model')
    scene=Scene(ref['scene'],ref['config']);torch.set_num_threads(8)
    model,_=scene.load_model(scene.spec['model']);records=[]
    for index in range(len(scene.views)):
        c,_,mask,*_=scene.get_view(index)
        with torch.no_grad():fresh=arrays(scene.render(c,model))
        path=args.cache/f'{index:04d}.npz'
        with np.load(path) as z:old={key:z[key].astype(np.float32) for key in ['alpha','depth','normal']}
        row=dict(index=index,name=c.image_name,cache_sha256=sha(path))
        for key in old:
            if old[key].shape!=fresh[key].shape:raise ValueError('Original native cache shape mismatch')
            delta=np.abs(old[key]-fresh[key]);row[key]=dict(exact=np.array_equal(old[key],fresh[key]),mean=float(delta.mean()),max=float(delta.max()),p99=float(np.quantile(delta,.99)))
        valid=mask&(old['alpha']>.5)&(fresh['alpha']>.5)&(old['depth']>0)&(fresh['depth']>0)
        unit=max(float(np.median(old['depth'][valid])),1e-9) if valid.any() else 1.
        row['foreground_depth_changed_pixels_gt3percent']=int((np.abs(old['depth'][valid]-fresh['depth'][valid])/unit>.03).sum())
        row['foreground_alpha_hole_disagreement_pixels']=int((mask&((old['alpha']>.5)!=(fresh['alpha']>.5))).sum())
        records.append(row)
        if (index+1)%32==0:print('native original cache audit',index+1,flush=True)
    write_json(args.output,dict(input_sha256=ref['input_sha256'],cache_root=str(args.cache.resolve()),views=records,
        exact_all_arrays=all(all(v[k]['exact'] for k in ['alpha','depth','normal']) for v in records),
        mean_array_difference={k:float(np.mean([v[k]['mean'] for v in records])) for k in ['alpha','depth','normal']},
        maximum_array_difference={k:max(v[k]['max'] for v in records) for k in ['alpha','depth','normal']},
        foreground_depth_changed_pixels_gt3percent=sum(v['foreground_depth_changed_pixels_gt3percent'] for v in records),
        foreground_alpha_hole_disagreement_pixels=sum(v['foreground_alpha_hole_disagreement_pixels'] for v in records),
        source_sha256=sha(Path(__file__).resolve()),scope='Actual fresh untouched original native GGGS renders versus all 208 stored geometry arrays. Exactness and any raster threshold differences are reported; no cleanup model or input is modified.'))

if __name__=='__main__':main()
