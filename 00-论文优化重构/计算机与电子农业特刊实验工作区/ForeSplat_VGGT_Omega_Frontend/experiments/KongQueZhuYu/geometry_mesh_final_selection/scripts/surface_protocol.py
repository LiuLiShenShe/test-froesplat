"""One recorded protocol for all depth-derived NKSR surface inputs."""
import json, hashlib
import zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DUAL=ROOT.parent/'geometry_rgb_dual_frontend'
PROTOCOL=dict(version=1,normal_method='masked central or one-sided world-point finite differences; orient toward source camera',
    normal_depth_jump_relative_max=.05,surface_alpha_min=.5,
    confidence='Direct maps retain the already exported original per-frontend p20 valid mask; no new common numerical confidence threshold',
    spatial_selection='One original oriented observation per spatial voxel AND normal hemisphere; earliest frame/pixel; no averaging across opposite thin surfaces',
    sampling_divisor=1024,reconstruction_divisor=512,tsdf_sdf_voxels=5,
    extent='0.5% and 99.5% XYZ quantiles of a deterministic strided sample (about 4000 per frame) from full original foreground observations after existing rigid COLMAP transform; scale definition only, no clipping',
    coordinate_policy='Existing frontend world gauge; Omega applies only its previously saved rigid export transform; no scan or Sim(3) fitting',
    mesh_processing='Raw preserved. Post removes only components with fewer than 50 triangles; no largest-component-only selection',
    nksr_config='official ks pretrained, solver max2000 tol1e-5, voxel_size explicit, detail_level None, MISE1, trim=True')

def write(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def save_npz_fast(path,**arrays):
    """Lossless ZIP level1, atomic publication; preserve exact dtype/array values."""
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for name,value in arrays.items():
            with archive.open(name+'.npy','w',force_zip64=True) as entry:np.save(entry,np.asarray(value),allow_pickle=False)
    temporary.replace(path)

def frontend(name):
    folder=DUAL/('VGGT_Omega_Original_RGB' if name=='omega' else 'Standard_VGGT_Original_RGB')
    align=np.eye(4)
    if name=='omega':align=np.asarray(json.loads((folder/'fusion_metadata.json').read_text())['scene_alignment_matrix'])
    return folder,align

def normals_from_grid(xyz,depth,valid,sensor):
    """Preserve silhouette pixels using a valid one-sided neighbour when needed."""
    xyz=np.asarray(xyz,dtype=np.float32);depth=np.asarray(depth);valid=valid & np.isfinite(xyz).all(-1)&np.isfinite(depth)&(depth>0)
    directions=[];available=[]
    for axis in [1,0]:
        plus=np.roll(xyz,-1,axis);minus=np.roll(xyz,1,axis)
        vp=np.roll(valid,-1,axis);vm=np.roll(valid,1,axis)
        dp=np.roll(depth,-1,axis);dm=np.roll(depth,1,axis)
        end=[slice(None),slice(None)];end[axis]=-1;vp[tuple(end)]=False
        end[axis]=0;vm[tuple(end)]=False
        vp &= abs(dp-depth)<=PROTOCOL['normal_depth_jump_relative_max']*depth
        vm &= abs(dm-depth)<=PROTOCOL['normal_depth_jump_relative_max']*depth
        tangent=np.where((vp&vm)[...,None],plus-minus,np.where(vp[...,None],plus-xyz,xyz-minus))
        directions.append(tangent);available.append(vp|vm)
    n=np.cross(directions[0],directions[1]);length=np.linalg.norm(n,axis=-1)
    good=valid&available[0]&available[1]&np.isfinite(length)&(length>1e-15)
    n/=np.maximum(length[...,None],1e-15)
    n[np.sum(n*(np.asarray(sensor)-xyz),axis=-1)<0]*=-1
    n[~good]=0
    return n.astype(np.float32),good

def unproject(depth,camera):
    h,w=depth.shape;yy,xx=np.mgrid[:h,:w]
    pc=np.stack([(xx-camera['Cx'])*depth/camera['Fx'],(yy-camera['Cy'])*depth/camera['Fy'],depth],-1)
    c2w=np.linalg.inv(np.asarray(camera['world_view_transform']).T)
    return (pc@c2w[:3,:3].T+c2w[:3,3]).astype(np.float32),c2w[:3,3].astype(np.float32)

def unique_observations(data,scale):
    grid=np.floor((data['xyz']-np.asarray(scale['origin']))/scale['sampling_voxel']).astype(np.int32)
    # Opposite normal directions occupy distinct records in the same voxel.
    dominant=np.argmax(np.abs(data['normal']),axis=1)
    sign=data['normal'][np.arange(len(grid)),dominant]>=0
    orient=(dominant*2+sign).astype(np.int32)
    keys=np.column_stack([grid,orient]).copy()
    view=keys.view(np.dtype((np.void,keys.dtype.itemsize*keys.shape[1]))).ravel()
    _,index=np.unique(view,return_index=True)
    return {k:v[index] for k,v in data.items()}

def merge_sampled(parts,scale,out):
    result=unique_observations({k:np.concatenate([p[k] for p in parts]) for k in parts[0]},scale)
    np.savez_compressed(out,**result)
    return len(result['xyz'])

def scale_for(name):
    path=ROOT/'configs'/f'{name}_spatial_scale.json'
    if path.exists():
        result=json.loads(path.read_text())
        if 'tsdf_depth_trunc' in result:return result
        folder,_=frontend(name);maximum=0.
        for c in json.loads((folder/'cameras.json').read_text())['frames']:
            stem=Path(c['image_name']).stem
            depth=np.load(folder/'per_frame'/f'depth_{stem}.npy',mmap_mode='r');valid=np.load(folder/'per_frame'/f'valid_foreground_{stem}.npy',mmap_mode='r')
            maximum=max(maximum,float(depth[valid].max()))
        result.update(max_original_foreground_camera_depth=maximum,tsdf_depth_trunc=maximum*1.25,extent_sample='deterministic stride approximately 4000 observations per frame; no clipping')
        write(path,result);return result
    folder,align=frontend(name);samples=[];counts=[]
    cams=json.loads((folder/'cameras.json').read_text())['frames']
    for c in cams:
        xyz=np.load(folder/'per_frame'/f'foreground_world_points_{Path(c["image_name"]).stem}.npy',mmap_mode='r')
        counts.append(len(xyz));samples.append(np.asarray(xyz[::max(1,len(xyz)//4000)])@align[:3,:3].T+align[:3,3])
    xyz=np.concatenate(samples);low,high=np.quantile(xyz,[.005,.995],axis=0);diag=float(np.linalg.norm(high-low))
    result=dict(frontend=name,original_foreground_points=sum(counts),extent_quantiles=[.005,.995],robust_bbox_min=low.tolist(),robust_bbox_max=high.tolist(),diagonal=diag,
        origin=((low+high)/2).tolist(),sampling_voxel=diag/PROTOCOL['sampling_divisor'],nksr_voxel=diag/PROTOCOL['reconstruction_divisor'],tsdf_voxel=diag/PROTOCOL['reconstruction_divisor'],
        tsdf_sdf_trunc=diag/PROTOCOL['reconstruction_divisor']*PROTOCOL['tsdf_sdf_voxels'],transform=align.tolist(),scale_fit_to_scan=False,
        note='Same reference scale is reused by direct, GGGS and 2DGS inputs of this frontend, including TSDF; no surface-dependent retuning')
    write(path,result);write(ROOT/'configs/surface_protocol.json',PROTOCOL)
    return scale_for(name)
