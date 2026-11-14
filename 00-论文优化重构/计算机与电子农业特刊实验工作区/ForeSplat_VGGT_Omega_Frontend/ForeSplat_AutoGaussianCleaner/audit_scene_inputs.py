#!/usr/bin/env python3
"""Check camera/image/mask/depth alignment without rendering or tuning on test plants."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json


def audit(spec):
    source=Path(spec['gggs_repo'])/'scene/colmap_loader.py'
    loader_spec=importlib.util.spec_from_file_location('gggs_camera_input_audit',source)
    loader=importlib.util.module_from_spec(loader_spec);loader_spec.loader.exec_module(loader)
    sparse=Path(spec['dataset'])/'sparse/0'
    if (sparse/'cameras.bin').exists():
        intrinsics=loader.read_intrinsics_binary(str(sparse/'cameras.bin'))
        extrinsics=loader.read_extrinsics_binary(str(sparse/'images.bin'))
    else:
        intrinsics=loader.read_intrinsics_text(str(sparse/'cameras.txt'))
        extrinsics=loader.read_extrinsics_text(str(sparse/'images.txt'))
    geometry=json.loads(Path(spec['geometry_cameras']).read_text())
    alignment=np.asarray(geometry.get('scene_alignment_matrix',np.eye(4)))
    inverse=np.linalg.inv(alignment)
    frames={Path(frame['image_name']).stem:frame for frame in geometry['frames']}
    records=[]
    for index,image in enumerate(sorted(extrinsics.values(),key=lambda item:item.name)):
        name=Path(image.name).stem;intrinsic=intrinsics[image.camera_id]
        if intrinsic.model not in ['PINHOLE','SIMPLE_PINHOLE']:raise ValueError(f'Undistorted pinhole required: {name}')
        rgb=Path(spec['dataset'])/'images'/Path(image.name).name
        mask=Path(spec['dataset'])/'masks'/(name+'.png')
        with Image.open(rgb) as picture:rgb_size=picture.size
        with Image.open(mask) as picture:mask_size=picture.size
        if rgb_size!=mask_size or rgb_size!=(intrinsic.width,intrinsic.height):
            raise ValueError(f'RGB/mask/camera dimensions differ: {name} {rgb_size} {mask_size} {(intrinsic.width,intrinsic.height)}')
        if name not in frames:raise ValueError(f'No matching existing geometry camera: {name}')
        frame=frames[name];raw=np.eye(4);raw[:3,:]=np.asarray(frame['extrinsic_model_w2c'])[:3,:]
        actual=np.eye(4);actual[:3,:3]=image.qvec2rotmat();actual[:3,3]=image.tvec
        difference=float(np.abs(raw@inverse-actual).max())
        if not np.allclose(raw@inverse,actual,atol=2e-4,rtol=2e-4):raise ValueError(f'External depth/GGGS pose mismatch: {name}: {difference}')
        stem=frame.get('stem',name);dp=Path(spec['geometry_dir'])/f'depth_{stem}.npy';cp=Path(spec['geometry_dir'])/f'confidence_{stem}.npy'
        depth=np.load(dp,mmap_mode='r').squeeze();confidence=np.load(cp,mmap_mode='r').squeeze()
        if depth.ndim!=2 or depth.shape!=confidence.shape:raise ValueError(f'Depth/confidence array dimensions differ: {name}')
        km=np.asarray(frame['intrinsic_model'])
        if km.shape!=(3,3) or not np.isfinite(km).all() or km[0,0]<=0 or km[1,1]<=0:
            raise ValueError(f'Invalid existing depth intrinsic: {name}')
        records.append(dict(index=index,name=name,rgb=str(rgb),mask=str(mask),rgb_size=rgb_size,
            depth=str(dp),confidence=str(cp),depth_shape=depth.shape,maximum_pose_difference=difference))
    if not records:raise ValueError('No GGGS cameras')
    return dict(views=len(records),maximum_pose_difference=max(item['maximum_pose_difference'] for item in records),
        camera_reader_sha256=sha(source),records=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());results=[]
    for job in manifest['plants']:
        spec=json.loads(Path(job['scene']).read_text());row=dict(plant=job['plant'],group=job['group'],scene=str(Path(job['scene']).resolve()))
        try:row.update(audit(spec));row['status']='aligned_existing_inputs'
        except (OSError,ValueError,KeyError) as error:row.update(status='invalid_scene_inputs',reason=str(error))
        results.append(row);print(job['plant'],row['status'],row.get('views'),row.get('maximum_pose_difference'),flush=True)
    write_json(args.output,dict(plants=results,aligned=sum(row['status']=='aligned_existing_inputs' for row in results),expected=len(results),
        scope='Only existing file presence, dimensions, named frame matching and camera/depth coordinate alignment. No reconstruction, image appearance analysis, cleanup or test-plant threshold tuning.'))
    if any(row['status']!='aligned_existing_inputs' for row in results):raise SystemExit(1)


if __name__=='__main__':main()
