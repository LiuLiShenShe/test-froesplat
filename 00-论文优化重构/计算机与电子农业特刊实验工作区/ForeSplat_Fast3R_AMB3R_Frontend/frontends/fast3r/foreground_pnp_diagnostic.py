#!/usr/bin/env python3
"""CPU diagnostic: mask-gated official PnP on immutable Fast3R predictions.

This does not replace the prescribed all-pixel official cameras. Background is
marked invalid only in a working confidence copy; foreground scores are exact.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party'/'fast3r'))
import cv2
import numpy as np
import torch
from frontends.common import EXPERIMENT,FrontendBundle,json_write
from frontends.fast3r.run import observe_pnp,validate_geometry
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from frontends.evaluate import load_frame,adjacent_consistency,safe_quantile,table_write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=EXPERIMENT/'Fast3R');parser.add_argument('--output',type=Path,default=EXPERIMENT/'Fast3R_foreground_PnP_diagnostic')
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():raise FileExistsError(f'preserving diagnostic {output}')
    torch.set_num_threads(8);cv2.setNumThreads(1);torch.manual_seed(0);np.random.seed(0);cv2.setRNGSeed(0)
    bundle=FrontendBundle(args.source);names=bundle.metadata['input_names'];preds=[];masks=[];counts=[]
    for name in names:
        frame=load_frame(bundle,name);xyz=torch.from_numpy(frame['xyz'].copy()).unsqueeze(0)
        rawconf=torch.from_numpy(frame['conf'].copy()).unsqueeze(0);mask=torch.from_numpy(frame['mask']).unsqueeze(0)
        confidence=torch.where(mask,rawconf,torch.ones_like(rawconf))
        assert torch.equal(confidence[mask],rawconf[mask])
        preds.append(dict(pts3d_in_other_view=xyz,conf=confidence));masks.append(frame['mask'])
        counts.append(dict(name=name,foreground_mask_pixels=int(mask.sum()),original_conf_gt1=int((rawconf>1).sum()),working_conf_gt1=int((confidence>1).sum())))
    start=time.perf_counter();print(f'CPU foreground-gated official PnP on{len(preds)} views',flush=True)
    with torch.inference_mode(),observe_pnp(preds) as statuses:
        poses,focals=MultiViewDUSt3RLitModule.estimate_camera_poses(preds,niter_PnP=100,focal_length_estimation_method='first_view_from_global_head')
    poses=np.asarray(poses[0]);focals=np.asarray(focals[0],float);seconds=time.perf_counter()-start
    h,w=masks[0].shape;k=np.repeat(np.eye(3)[None],len(poses),axis=0);k[:,0,0]=focals;k[:,1,1]=focals;k[:,0,2]=w/2;k[:,1,2]=h/2
    np.savez_compressed(output/'diagnostic_cameras.npz',c2w=poses,w2c=np.linalg.inv(poses),intrinsics_model=k,focals=focals,names=np.asarray(names))
    json_write(output/'pose_status.json',statuses);json_write(output/'foreground_validity.json',counts)
    validation=validate_geometry(preds,poses,focals,np.stack(masks),statuses);json_write(output/'geometry_validation.json',validation)
    primary=json.loads((bundle.path/'geometry_metrics.json').read_text());scale=primary['normalization_extent']
    pairs=[];previous=None
    for i,name in enumerate(names):
        frame=load_frame(bundle,name);frame['c2w']=poses[i];frame['k']=k[i]
        if previous is not None:
            for source,target in ((previous,frame),(frame,previous)):
                pairs.append(dict(source=source['name'],target=target['name'],**adjacent_consistency(source,target,scale)))
        previous=frame
    table_write(output/'adjacent_pairs.csv',pairs)
    relative=np.einsum('nij,njk->nik',poses[:-1,:3,:3].transpose(0,2,1),poses[1:,:3,:3])
    rotation=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    result=dict(kind='separate foreground-validity PnP diagnostic; not a new model or a replacement primary result',source_bundle=str(bundle.path),
        input_views=len(names),gpu_used=False,cpu_seconds=seconds,seed=0,pnp_iterations=100,focal_method='first_view_from_global_head',
        confidence_rule='working copy: exact original confidence in foreground; background set to1.0 to mark invalid under official conf>1 test; no artificial increase',
        mask_processing='exact saved model-input masks; no erosion, resize, resampling or exposure change',
        pointmaps_unchanged=True,original_confidence_unchanged=True,primary_cameras_unchanged=True,
        pnp_success_rate=sum(s['status']=='success' for s in statuses)/len(statuses),
        foreground_reprojection_p50_px=safe_quantile([r['foreground_reprojection_median_px'] for r in validation['frames'] if r['foreground_reprojection_median_px'] is not None],.5),
        rotation_step_p90_deg=safe_quantile(rotation,.9),
        adjacent_point_distance_p50_normalized=safe_quantile([p['distance_p50'] for p in pairs if p.get('distance_p50') is not None],.5),
        adjacent_depth_agreement_fraction=safe_quantile([p['depth_agreement_fraction'] for p in pairs if p.get('depth_agreement_fraction') is not None],.5),
        projection_overlap_fraction=sum(p['overlap'] for p in pairs)/max(sum(p['projected'] for p in pairs),1),
        normalization_extent=scale,normalization_rule='exact primary metric scene extent reused; no geometric or scale changes')
    json_write(output/'summary.json',result)
    fields={'pnp_success_rate':'pnp_success_rate','foreground_reprojection_p50_px':'self_reprojection_p50_px','rotation_step_p90_deg':'rotation_step_p90_deg','adjacent_point_distance_p50_normalized':'adjacent_point_distance_p50_normalized','adjacent_depth_agreement_fraction':'adjacent_depth_agreement_fraction'}
    report=['# Foreground-gated Fast3R PnP diagnostic','',
        'The primary experiment and all saved raw confidence/pointmaps/cameras remain unchanged. Only a temporary confidence copy marks background pixels invalid for the official PnP validity test. Foreground scores are retained exactly; masks are used without erosion or further transforms. No GPU is used.',
        '', 'The original official PnP API runs with100 iterations and first-view/global-head focal estimation. Focal length is re-estimated from the same head using the temporary validity copy, so this control isolates foreground validity in the camera estimation procedure rather than only changing RANSAC correspondences.',
        '', '| Diagnostic | Original official all-pixel PnP | Foreground-gated diagnostic |','| --- | --- | --- |']
    for diagnostic,original in fields.items():report.append(f'| {diagnostic} | {primary.get(original)} | {result.get(diagnostic)} |')
    report += ['', 'A gain in camera/pointmap agreement would identify background sensitivity in PnP. It would not prove that raw global leaf geometry is correct. The raw whole-plant side/top pointmaps are identical in both cases, so any collapsed/banded raw geometry remains. This is not scanned-GT accuracy, bundle adjustment or a new foreground confidence calibration.',
        '', 'Sources: saved Fast3R bundle and official camera API in `third_party/fast3r/fast3r/models/multiview_dust3r_module.py`; no source network edits or pretrained weight changes.']
    (output/'RUN_REPORT.md').write_text('\n'.join(report));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
