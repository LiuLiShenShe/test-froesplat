#!/usr/bin/env python3
"""Check whether an actual rejected cleanup exposes repairable original foreground primitives."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from scipy import ndimage
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(8)
    report=json.loads((args.candidate/'cleanup_metrics.json').read_text());spec=report['scene']
    if sha(spec['model'])!=report['input_sha256']:raise ValueError('Original input has changed')
    scene=Scene(spec,report['config']);original,rows=scene.load_model(spec['model']);candidate,_=scene.load_model(args.candidate/'cleaned_gaussians.ply')
    kept=np.load(args.candidate/'original_gaussian_ids.npy');removed=np.flatnonzero(~np.isin(np.arange(len(rows)),kept))
    xyz=np.column_stack([rows[name][removed] for name in ['x','y','z']]);support=np.zeros(len(removed),np.int32);hit=np.zeros(len(removed),np.int32)
    rgb_sum=np.zeros((len(removed),3));rgb_count=np.zeros(len(removed),np.int32);records=[]
    n=min(16,len(scene.views));centers=np.array([np.linalg.inv(v['E'])[:3,3] for v in scene.views]);directions=np.array([v['E'][2,:3] for v in scene.views])
    features=np.column_stack([centers/max(np.linalg.norm(np.ptp(centers,axis=0)),1e-9),directions]);chosen=[0];distance=np.full(len(features),np.inf)
    for _ in range(n-1):
        distance=np.minimum(distance,((features-features[chosen[-1]])**2).sum(1));distance[chosen]=-1;chosen.append(int(distance.argmax()))
    for name in ['crop_0035','crop_0045','crop_0046','crop_0047']:
        matches=[i for i,v in enumerate(scene.views) if v['name']==name]
        chosen.extend(matches)
    # These named views are explicit diagnostic witnesses, never operational cleaner ROIs.
    for index in sorted(set(chosen)):
        camera,target,mask,depth,confidence=scene.get_view(index);r=regions(mask,scene.config['boundary_relative_width'])
        with torch.no_grad():before=arrays(scene.render(camera,original));after=arrays(scene.render(camera,candidate))
        damage=mask&(((before['alpha']>.5)&(after['alpha']<=.5))|((((after['rgb']-target)**2).mean(2)-((before['rgb']-target)**2).mean(2))>1e-5))
        roi=ndimage.binary_dilation(damage,iterations=max(2,3*r['radius']))&mask
        camera_xyz=xyz@scene.views[index]['E'][:3,:3].T+scene.views[index]['E'][:3,3];projected=camera_xyz@scene.views[index]['K'].T
        xy=np.rint(projected[:,:2]/np.maximum(projected[:,2,None],1e-9)).astype(np.int64)
        valid=(camera_xyz[:,2]>.01)&(xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0]);local=np.flatnonzero(valid)
        x,y=xy[local].T;in_foreground=mask[y,x];local=local[in_foreground];x,y=xy[local].T
        valid_depth=mask&np.isfinite(depth)&(depth>0)&np.isfinite(confidence);unit=max(float(np.median(depth[valid_depth])),1e-9)
        good=valid_depth[y,x]&(confidence[y,x]>=np.quantile(confidence[valid_depth],.2))&(np.abs(camera_xyz[local,2]-depth[y,x])/unit<=.03)
        support[local[good]]+=1;rgb_sum[local[good]]+=target[y[good],x[good]];rgb_count[local[good]]+=1
        hit[local[roi[y,x]]]+=1
        records.append(dict(name=camera.image_name,index=index,damage_pixels=int(damage.sum()),roi_projected_original_centers=int(roi[y,x].sum()),depth_supported_removed_centers=int(good.sum())))
        print(camera.image_name,records[-1],flush=True)
    selected=(support>=2)&(hit>0);ids=removed[selected]
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'FOREGROUND_RECOVERY_SOURCE_CANDIDATES.npz',original_ids=ids,depth_views=support[selected],damage_views=hit[selected],
        target_rgb_mean=rgb_sum[selected]/np.maximum(rgb_count[selected,None],1))
    write_json(args.output/'FOREGROUND_RECOVERY_SOURCE_AUDIT.json',dict(candidate=str(args.candidate.resolve()),input_sha256=report['input_sha256'],
        removed_gaussians=len(removed),depth_supported_centers=int((support>=2).sum()),repair_source_candidates=len(ids),views=records,
        scope='Diagnostic availability check only. No Gaussians restored, optimized or accepted. Named historical witnesses supplement an automatic diverse camera sample; they are not an operational cleaning configuration.',source_sha256=sha(Path(__file__).resolve())))


if __name__=='__main__':main()
