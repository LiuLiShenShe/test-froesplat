"""Read existing full maps, derive normals, retain provenance and sample once."""
import argparse,json,time
from pathlib import Path
import numpy as np
from surface_protocol import ROOT,DUAL,PROTOCOL,frontend,scale_for,normals_from_grid,unique_observations,merge_sampled,write

def run(name):
    out=ROOT/'surface_inputs'/f'{name}_direct';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    start=time.perf_counter();folder,align=frontend(name);scale=scale_for(name)
    (out/'original_foreground.ply').symlink_to(folder/'foreground.ply') if not (out/'original_foreground.ply').exists() else None
    cams=json.loads((folder/'cameras.json').read_text())['frames'];parts=[];records=[]
    for c in cams:
        stem=Path(c['image_name']).stem;sid=c['source_index'];p=folder/'per_frame'
        xyz=np.load(p/f'world_points_{stem}.npy');depth=np.load(p/f'depth_{stem}.npy');valid=np.load(p/f'valid_foreground_{stem}.npy').astype(bool)
        rgb=np.load(p/f'appearance_rgb_{stem}.npy');confidence=np.load(p/f'confidence_{stem}.npy')
        xyz=(xyz@align[:3,:3].T+align[:3,3]).astype(np.float32)
        sensor=(align@np.asarray(c['c2w']))[:3,3].astype(np.float32)
        normal,good=normals_from_grid(xyz,depth,valid,sensor);y,x=np.nonzero(valid);n=len(x)
        full=dict(xyz=xyz[valid],rgb=rgb[valid],normal=normal[valid],normal_valid=good[valid],confidence=confidence[valid],depth=depth[valid],
            pixel_xy=np.column_stack([x,y]).astype(np.int32),source_id=np.full(n,sid,dtype=np.int16),sensor=np.broadcast_to(sensor,(n,3)).copy(),foreground_mask=np.ones(n,dtype=bool))
        dest=out/'per_frame'/f'{stem}.npz';dest.parent.mkdir(exist_ok=True)
        if not dest.exists():np.savez_compressed(dest,**full)
        filtered={k:v[full['normal_valid']] for k,v in full.items() if k not in ('normal_valid','foreground_mask')}
        parts.append(unique_observations(filtered,scale))
        records.append(dict(source_id=sid,full_foreground_points=n,valid_normal_points=int(good.sum()),normal_invalid_points=int(n-good.sum()),sampled_frame_points=len(parts[-1]['xyz'])))
        print(name,sid,'original',n,'valid normal',int(good.sum()),flush=True)
    count=merge_sampled(parts,scale,out/'nksr_input.npz')
    assert sum(r['full_foreground_points'] for r in records)==scale['original_foreground_points']
    runtime=DUAL/'render_comparison'/('E8' if name=='omega' else 'E10')/'camera_runtime.json'
    (out/'camera_runtime.json').symlink_to(runtime) if not (out/'camera_runtime.json').exists() else None
    write(out/'complete.json',dict(status='complete',frontend=name,source=str(folder),raw_point_count=scale['original_foreground_points'],nksr_points=count,
        valid_normal_total=sum(r['valid_normal_points'] for r in records),frames=records,scale=scale,protocol=PROTOCOL,elapsed_seconds=time.perf_counter()-start,
        original_cloud_preserved=str(folder/'foreground.ply'),surface_source='Full existing depth-unprojected world maps, not COLMAP sample or learned point head'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frontend',choices=['omega','standard','both'],default='both');a=p.parse_args()
    for name in ['omega','standard'] if a.frontend=='both' else [a.frontend]:run(name)
