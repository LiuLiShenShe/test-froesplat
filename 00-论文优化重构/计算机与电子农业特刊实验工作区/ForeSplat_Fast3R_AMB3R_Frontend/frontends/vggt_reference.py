#!/usr/bin/env python3
"""Import saved VGGT-Omega depth predictions without running/fusing any model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from frontends.common import EXPERIMENT,export_bundle,json_write,sha256
from frontends.lossless_foreground_ply import complete_lossless_exports,validate_per_frame_exports

def complete_existing_exports(output):
    """Fill missing derived files while preserving original imported arrays."""
    output=Path(output)
    audit_path=output/'bundle_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        backup=output/'bundle_audit_before_legacy_per_frame_export.json'
        if not audit.get('passed',False) and not backup.exists():shutil.copy2(audit_path,backup)
    params_path=output/'run_parameters.json'
    params=json.loads(params_path.read_text())
    bundle=json.loads((output/'bundle.json').read_text())
    if params.get('method')!=bundle['method']:
        backup=output/'run_parameters_before_legacy_per_frame_export.json'
        if not backup.exists():shutil.copy2(params_path,backup)
        if params.get('method') is not None:params.setdefault('legacy_import_method_label',params['method'])
        params.setdefault('legacy_model_label',bundle.get('metadata',{}).get('method',bundle['method']))
        params['method']=bundle['method']
        json_write(params_path,params)
    try:
        validation=validate_per_frame_exports(output)
    except FileNotFoundError:
        complete_lossless_exports(output)
        validation=validate_per_frame_exports(output)
    return validation

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT.parent/'ForeSplat_VGGT_Omega_Frontend'/'runs'/'full_sequence_batch'/'KongQueZhuYu'/'GGGS_Dataset')
    parser.add_argument('--output',type=Path,default=EXPERIMENT/'VGGT_Omega_Existing')
    args=parser.parse_args();source=args.source
    if (args.output/'bundle.json').exists():
        validation=complete_existing_exports(args.output)
        print(f'preserved original bundle arrays; foreground exports verified: {validation}');return
    cameras=json.loads((source/'per_frame'/'cameras.json').read_text())
    original_meta=json.loads((source/'metadata'/'reconstruction_metadata.json').read_text())
    original_report_path=source.parent/'RUN_REPORT.md'
    original_report=original_report_path.read_text() if original_report_path.exists() else ''
    timing_match=re.search(r'Inference time:\s*([0-9.]+)\s*s',original_report)
    memory_match=re.search(r'allocated\s+([0-9.]+)\s*MiB',original_report)
    transform_list=json.loads((source/'metadata'/'preprocess_transforms.json').read_text())['frames']
    transforms_by_name={t['image_name']:t for t in transform_list}
    with open(EXPERIMENT/'manifest.csv') as f:manifest={r['name']:r for r in csv.DictReader(f) if r['status']=='ok'}
    rows=[];rgbs=[];masks=[];transforms=[];predictions=[];poses=[];intrinsics=[]
    for camera in cameras['frames']:
        name=camera['image_name'];stem=Path(name).stem;rows.append(manifest[name])
        depth=np.load(source/'per_frame'/f'depth_{stem}.npy').squeeze()
        conf=np.load(source/'per_frame'/f'confidence_{stem}.npy').squeeze()
        rgb=np.load(source/'per_frame'/f'model_rgb_{stem}.npy')
        mask=np.load(source/'per_frame'/f'model_mask_{stem}.npy')>0
        h,w=depth.shape;k=np.asarray(camera['intrinsic_model'],dtype=np.float64)
        e=np.eye(4);e[:3]=np.asarray(camera['extrinsic_model_w2c']);pose=np.linalg.inv(e)
        y,x=np.indices((h,w),dtype=np.float32)
        rays=np.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],np.ones_like(x)),axis=-1)
        local=rays*depth[...,None];world=local@pose[:3,:3].T+pose[:3,3]
        predictions.append(dict(world_points=world.astype(np.float32),world_points_conf=conf,local_points=local.astype(np.float32),local_confidence=conf,depth=depth))
        rgbs.append(rgb);masks.append(mask);poses.append(pose);intrinsics.append(k)
        t=transforms_by_name[name]
        transforms.append(dict(name=name,original_size_wh=[t['original_width'],t['original_height']],input_size_wh=[w,h],
            resize_size_wh=[t['resized_width'],t['resized_height']],resize_scale_xy=[t['scale_x'],t['scale_y']],
            padding_ltrb=[t['pad_left'],t['pad_top'],t['pad_right'],t['pad_bottom']],crop_ltrb=[0,0,0,0],
            pixel_transform=t['pixel_transform'],pixel_transform_inverse=t['pixel_transform_inverse'],
            pixel_coordinate_convention='integer pixel centers; inherited Pillow half-pixel convention',
            rgb_interpolation=t['rgb_resize_filter'],mask_interpolation=t['mask_resize_filter'],exposure='inherited original run',mask_erosion=0))
    prior_root=source.parents[3]
    sha_file=prior_root/'weights'/'sha256.txt'
    checkpoint_sha=sha_file.read_text().split()[0] if sha_file.exists() else None
    model_repo=prior_root/'vggt'
    commit=subprocess.check_output(['git','-C',str(model_repo),'rev-parse','HEAD'],text=True).strip() if (model_repo/'.git').exists() else None
    meta=dict(method='VGGT-Ω (existing depth frontend)',scale='ambiguous; uncalibrated VGGT-Omega scene scale',world_frame='raw VGGT-Omega world; original visualization alignment not applied',
        source=str(source),source_cameras_sha256=sha256(source/'per_frame'/'cameras.json'),
        source_metadata_sha256=sha256(source/'metadata'/'reconstruction_metadata.json'),weight_sha256=checkpoint_sha,git_commit=commit,
        input_views=len(rows),available_views=len(manifest),full_sequence=len(rows)==len(manifest),gpu_physical=None,
        import_mode='saved depth backprojection; no new inference; no use of fused cloud',
        prediction_semantics='Global points reconstructed algebraically from saved official depth, intrinsics and raw W2C. No raw point-head maps were saved in the legacy artifact.',
        local_point_semantics='depth unprojection in each camera; not a dedicated local prediction head',
        source_confidence_export_percentile=original_meta.get('confidence_threshold_percentile'),
        source_confidence_export_threshold=original_meta.get('confidence_threshold_value'),
        evaluation_confidence='raw saved confidence >1 baseline, same threshold definition as new bundles',
        inference_seconds=float(timing_match[1]) if timing_match else original_meta.get('inference_seconds'),
        peak_vram_gib=float(memory_match[1])/1024 if memory_match else None,
        source_report_sha256=sha256(original_report_path) if original_report_path.exists() else None,
        source_export_elapsed_seconds=original_meta.get('elapsed_seconds'),timing_provenance='original RUN_REPORT.md; no inference rerun',
        inference_time_scope='legacy recorded model inference; import/backprojection excluded; exact legacy stage boundaries not recorded',
        total_frontend_seconds=None,total_frontend_time_scope='unknown in legacy artifact; no invented combined timing',
        provenance_exemptions={'git_commit':'Imported existing artifact; original model repository commit not available in saved run' if commit is None else None})
    inputs=dict(rows=rows,rgb=np.stack(rgbs),masks=np.stack(masks),valid_pixels=np.ones_like(np.stack(masks),bool),transforms=transforms)
    export_bundle(args.output,inputs,predictions,np.asarray(poses),np.asarray(intrinsics),meta)
    json_write(args.output/'run_parameters.json',dict(method=meta['method'],legacy_model_label=meta['method'],operation='import_saved_depth_frontend',new_model_inference=False,
        source=str(source),saved_fields=['depth','confidence','intrinsics','W2C','RGB','mask','pixel_transforms'],
        pointmap_reconstruction='saved depth backprojection through K and C2W; original world frame',
        confidence_baseline='finite XYZ, confidence >1, valid unpadded pixels',mask_processing='saved mask unchanged; no erosion',
        legacy_parameters_provenance=str(source/'metadata'/'reconstruction_metadata.json'),
        legacy_parameters_known=original_meta,legacy_parameters_unknown='No unsaved original model-call parameters are inferred.',training_started=False))
    # Current unified exporter writes both foreground branches per view; enforce
    # the actual-file/count postcondition rather than exempting missing exports.
    validate_per_frame_exports(args.output)
    json_write(args.output/'status.json',dict(status='complete',views=len(rows),imported_existing=True))
    print(f'VGGT raw depth bundle imported: {len(rows)} views into {args.output}',flush=True)

if __name__=='__main__':main()
