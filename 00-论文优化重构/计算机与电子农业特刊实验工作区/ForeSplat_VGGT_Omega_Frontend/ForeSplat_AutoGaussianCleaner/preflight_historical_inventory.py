#!/usr/bin/env python3
"""Read-only19-scene native/model/RGB/mask/depth/camera compatibility audit."""
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json


def native_header(path):
    fields=[];count=None
    with path.open('rb') as f:
        if f.readline().strip()!=b'ply':raise ValueError('InvalidPLY header')
        for _ in range(256):
            line=f.readline().decode('ascii').strip()
            if line.startswith('element vertex '):count=int(line.split()[-1])
            elif line.startswith('property '):fields.append(line.split()[-1])
            elif line=='end_header':break
        else:raise ValueError('NativePLY header exceeds256 lines')
    expected=['x','y','z','opacity','filter_3D']+[f'scale_{k}' for k in range(3)]+[f'rot_{k}' for k in range(4)]+[f'f_dc_{k}' for k in range(3)]
    missing=sorted(set(expected)-set(fields))
    return dict(gaussians=count,bytes=path.stat().st_size,fields=fields,missing_fields=missing,
                sg_fields=[k for k in fields if k.startswith('sg_')])


def inspect(item):
    scene_path=Path(item['scene']);scene=json.loads(scene_path.read_text());errors=[]
    header=native_header(Path(scene['model']))
    if header['missing_fields']:errors.append('Missing GGGS native fields')
    if header['sg_fields']:errors.append('SG export requires explicit support; current historical exporter accepts SGdegree0')
    path=Path(scene['gggs_repo'])/'scene/colmap_loader.py'
    spec=importlib.util.spec_from_file_location('foresplat_preflight_colmap',path)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    dataset=Path(scene['dataset']);sparse=dataset/'sparse/0'
    extr=(loader.read_extrinsics_binary(str(sparse/'images.bin')) if (sparse/'images.bin').exists()
          else loader.read_extrinsics_text(str(sparse/'images.txt')))
    geometry=json.loads(Path(scene['geometry_cameras']).read_text())
    alignment=np.asarray(geometry.get('scene_alignment_matrix',np.eye(4)))
    frames={Path(r['image_name']).stem:r for r in geometry['frames']}
    missing=[];bad_poses=[];pose_error=0.;names=[]
    for e in sorted(extr.values(),key=lambda e:e.name):
        name=Path(e.name).stem;names.append(name)
        for path in [dataset/'images'/Path(e.name).name,dataset/'masks'/(name+'.png')]:
            if not path.is_file():missing.append(str(path))
        r=frames.get(name)
        if r is None:errors.append('No geometry frame for '+name);continue
        for prefix in ['depth','confidence']:
            path=Path(scene['geometry_dir'])/f'{prefix}_{r.get("stem",name)}.npy'
            if not path.is_file():missing.append(str(path))
        E=np.eye(4);E[:3,:3]=e.qvec2rotmat();E[:3,3]=e.tvec
        raw=np.eye(4);raw[:3,:]=np.asarray(r['extrinsic_model_w2c'])[:3,:]
        aligned=raw@np.linalg.inv(alignment)
        pose_error=max(pose_error,float(np.max(np.abs(aligned-E))))
        if not np.allclose(aligned,E,atol=2e-4,rtol=2e-4):bad_poses.append(name)
    if missing:errors.append('Missing required scene assets')
    if bad_poses:errors.append('GGGS/external-depth pose mismatch')
    if len(names)<64:errors.append('Insufficient64-camera confirmation population')
    return dict(plant=item['plant'],scene_config=str(scene_path.resolve()),native_model=scene['model'],
        native_header=header,observed_cameras=len(names),geometry_frames=len(frames),
        maximum_pose_element_error=pose_error,missing_assets=missing,mismatched_pose_names=bad_poses,
        errors=errors,status='ready_for_native_cleaning' if not errors else 'input_requires_resolution',
        scope='Input compatibility only; no quality, cleaning or viewer success is implied')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE/'experiments/Historical_TwoStage_Cleaner/INPUT_PREFLIGHT.json')
    a=p.parse_args();items=json.loads((BASE/'configs/plant_manifest.json').read_text())['plants'];records=[]
    for item in items:
        try:r=inspect(item)
        except Exception as e:r=dict(plant=item['plant'],status='input_requires_resolution',errors=[repr(e)])
        records.append(r);print(r['plant'],r['status'],r.get('observed_cameras'),r.get('errors'),flush=True)
    write_json(a.output,dict(total=len(items),ready=sum(not r.get('errors') for r in records),plants=records,
        retraining=False,mesh_edits=False,read_only=True))


if __name__=='__main__':main()
