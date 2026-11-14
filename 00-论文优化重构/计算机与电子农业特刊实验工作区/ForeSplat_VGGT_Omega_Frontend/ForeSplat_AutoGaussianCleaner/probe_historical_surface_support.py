#!/usr/bin/env python3
"""Audit projected-depth proxy against independent automatic RAP visual hulls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.historical.runner import HistoricalEvaluator
from ForeSplat_AutoGaussianCleaner.cleaner.historical.evidence import camera_indices
from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.free_support import free_support
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();root=a.experiment.resolve();r=json.loads((root/'execution.json').read_text())['reference']
    scene=Scene(r['scene'],r['config']);torch.set_num_threads(8);indices=camera_indices(scene,64)
    surface=external_surface(scene,indices);xyz=surface['points'];seen=np.zeros(len(xyz),int);outside=seen.copy()
    for index in indices:
        _,_,mask,*_=scene.get_view(index);v=scene.views[index]
        mask=ndimage.binary_dilation(mask,iterations=max(2,round(np.sqrt(mask.size)*.002)))
        p=xyz@v['E'][:3,:3].T+v['E'][:3,3];h=p@v['K'].T
        xy=np.rint(h[:,:2]/np.maximum(h[:,2:3],1e-9)).astype(int);x,y=xy.T
        valid=(p[:,2]>.01)&(x>=0)&(y>=0)&(x<mask.shape[1])&(y<mask.shape[0]);ii=np.flatnonzero(valid)
        seen[ii]+=1;outside[ii]+=~mask[y[ii],x[ii]]
    records=[];models={}
    for label,path in [('historical',scene.spec['baseline_models']['R5']),('original',scene.spec['model'])]:
        models[label]=scene.load_model(path)[0]
    for fraction in [1.,.25,.1,.05]:
        keep=(seen>=4)&(outside<=fraction*seen)
        points=xyz[keep];support=dict(surface,points=points,tree=cKDTree(points))
        ev=HistoricalEvaluator(scene,root/'comparison/surface_support',surface=surface)
        # Hold camera poses and framing fixed, change only proxy point support.
        views=free_support(scene,support)
        for old,new in zip(ev.free_views,views):
            new['view']=old['view']
            from ForeSplat_AutoGaussianCleaner.evaluation.free_view_evaluation import proxy
            mask,box=proxy(old['view'],support);new.update(surface_mask=mask,outside_surface=ndimage.distance_transform_edt(~mask)>3,outside_aabb=~box)
        ev.free_views=views
        row=dict(maximum_outside_fraction=fraction,retained_depth_points=int(keep.sum()),original_points=len(xyz),
                 models={label:ev.free_background(model) for label,model in models.items()})
        records.append(row);ev.close();print(fraction,len(points),{k:v['black_pixels'] for k,v in row['models'].items()},flush=True)
    dest=root/'comparison/surface_support/SUPPORT_AUDIT.json'
    write_json(dest,dict(records=records,point_observations=dict(median=float(np.median(seen)),outside_fraction_quantiles=np.quantile(outside/np.maximum(seen,1),[0,.5,.9,.99,1]).tolist()),
        role='Diagnostic proxy sensitivity only; no final deletion or geometric ground-truth claim'))


if __name__=='__main__':main()
