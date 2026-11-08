#!/usr/bin/env python3
"""CPU-only geometry evaluation and consistent scientific figures.

All results are predicted-geometry diagnostics, not accuracy against scanned GT.
Per-view raw points remain untouched. Visualization normalization is display-only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
import csv
import json
import io
import math
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from frontends.common import EXPERIMENT,FrontendBundle,json_write

METHODS=['VGGT_Omega_Existing','Fast3R','AMB3R_Base_OFF','AMB3R_Base_ON','AMB3R_SfM','Fast3R_AMB3R_SfM']
ROIS={'leaf':[540,570,1080,1230],'leaf_edge':[510,630,720,1110]}
ROI_FRAME='crop_0000.png'
PLOT_POINT_SIZE=.5
GT_ROOT=Path('/data/fj/F2DMAS/03-GT/KongQueZhuYu')
GT_INDICES=[0,25,50,75,100]

def saved_npz(path):
    # A contiguous read avoids ZipFile's repeated disk seeks under I/O pressure.
    return np.load(io.BytesIO(Path(path).read_bytes()))

def evaluate_manual_gt(bundle,reuse=False):
    """2D manual-GT pixel support only; no claim of measured3D geometry."""
    saved=bundle.path/'manual_gt_2d_support.json'
    if reuse and saved.exists():return json.loads(saved.read_text())
    rows=[]
    if not hasattr(bundle,'transforms'):bundle.transforms=json.loads((bundle.path/'pixel_transforms.json').read_text())
    for source_index in GT_INDICES:
        name=f'crop_{source_index:04d}.png';gtpath=GT_ROOT/f'gt_{source_index:04d}.png'
        if not gtpath.exists() or name not in bundle.metadata['input_names']:continue
        gt=np.asarray(Image.open(gtpath))>0
        assert gt.ndim==2
        frame=load_frame(bundle,name);t=bundle.transforms[frame['index']];a=np.asarray(t['pixel_transform'])
        assert [gt.shape[1],gt.shape[0]]==t['original_size_wh']
        y,x=np.nonzero(gt);u=np.rint(a[0,0]*x+a[0,1]*y+a[0,2]).astype(int);v=np.rint(a[1,0]*x+a[1,1]*y+a[1,2]).astype(int)
        h,w=frame['mask'].shape;inside=(u>=0)&(u<w)&(v>=0)&(v<h)
        support=np.zeros(len(y),bool);support[inside]=frame['mask'][v[inside],u[inside]]&frame['valid'][v[inside],u[inside]]
        edge=gt&~binary_erosion(gt);is_edge=edge[y,x]
        my,mx=np.indices((h,w));original=np.stack((mx,my,np.ones_like(mx)),axis=-1)@np.linalg.inv(a).T
        ox=np.rint(original[...,0]).astype(int);oy=np.rint(original[...,1]).astype(int);model_inside=(ox>=0)&(ox<gt.shape[1])&(oy>=0)&(oy<gt.shape[0])
        gt_model=np.zeros((h,w),bool);gt_model[model_inside]=gt[oy[model_inside],ox[model_inside]]
        model_support=frame['mask']&frame['valid']
        row=dict(name=name,source_index=source_index,gt_pixels=int(gt.sum()),gt_supported_pixels=int(support.sum()),
            gt_edge_pixels=int(is_edge.sum()),gt_supported_edge_pixels=int((support&is_edge).sum()),
            manual_gt_2d_support_recall=float(support.mean()),manual_gt_2d_edge_support_recall=float((support&is_edge).sum()/max(is_edge.sum(),1)),
            model_foreground_support_precision=float((model_support&gt_model).sum()/max(model_support.sum(),1)),
            evaluation='nearest footprint of existing confidence-valid RAP foreground geometry at original GT pixels; not3D accuracy')
        rows.append(row)
    if rows:
        table_write(bundle.path/'manual_gt_2d_support.csv',rows)
        result=dict(gt_reference_frames=len(rows),manual_gt_2d_support_recall=sum(r['gt_supported_pixels'] for r in rows)/sum(r['gt_pixels'] for r in rows),
            manual_gt_2d_edge_support_recall=sum(r['gt_supported_edge_pixels'] for r in rows)/sum(r['gt_edge_pixels'] for r in rows),
            manual_gt_2d_model_support_precision_mean=float(np.mean([r['model_foreground_support_precision'] for r in rows])),
            gt_semantics='existing manually annotated binary masks at5original views; no registered3D GT')
        json_write(bundle.path/'manual_gt_2d_support.json',result);return result
    return dict(gt_reference_frames=0)

def load_frame(bundle,name,include_local=False):
    if not hasattr(bundle,'frame_cache'):bundle.frame_cache={}
    if name in bundle.frame_cache:
        cached=bundle.frame_cache[name]
        if include_local and 'local_points' not in cached['pred']:
            with saved_npz(bundle.path/'raw_predictions'/f'{Path(name).stem}.npz') as f:
                cached['pred'].update({k:f[k] for k in ('local_points','local_confidence') if k in f})
        return cached
    bulk=load_bulk_raw_frame(bundle,name) if getattr(bundle,'enable_bulk_raw',False) and not include_local else None
    if bulk is not None:
        bundle.frame_cache[name]=bulk
        return bulk
    stem=Path(name).stem
    with saved_npz(bundle.path/'raw_predictions'/f'{stem}.npz') as f:
        keys=['world_points','world_points_conf']
        if include_local:keys += [k for k in ('local_points','local_confidence') if k in f]
        pred={k:f[k] for k in keys}
    mask=np.asarray(Image.open(bundle.path/'mask'/f'{stem}.png'))>0
    rgb=np.asarray(Image.open(bundle.path/'rgb'/f'{stem}.png'))
    xyz=np.asarray(pred['world_points']).reshape(*mask.shape,3)
    conf=np.asarray(pred['world_points_conf']).reshape(mask.shape)
    i=bundle.metadata['input_names'].index(name)
    c2w=bundle.cameras['c2w'][i];k=bundle.cameras['intrinsics_model'][i]
    if not hasattr(bundle,'transforms'):
        bundle.transforms=json.loads((bundle.path/'pixel_transforms.json').read_text())
    left,top,right,bottom=bundle.transforms[i]['padding_ltrb'];valid_pixels=np.zeros(mask.shape,bool)
    valid_pixels[top:mask.shape[0]-bottom,left:mask.shape[1]-right]=True
    finite=np.isfinite(xyz).all(-1)&np.isfinite(conf)&valid_pixels
    frame=dict(name=name,pred=pred,xyz=xyz,conf=conf,mask=mask,rgb=rgb,valid=finite&(conf>1),finite=finite,c2w=c2w,k=k,index=i)
    bundle.frame_cache[name]=frame
    return frame

def load_bulk_raw_frame(bundle,name):
    """Exact saved XYZ/RGB recovery when every unpadded pixel is finite."""
    if getattr(bundle,'bulk_raw_unavailable',False):return None
    if not hasattr(bundle,'bulk_raw'):
        transforms=bundle.pixel_transforms
        areas=[]
        for t in transforms:
            w,h=t['input_size_wh'];l,top,r,b=t['padding_ltrb'];areas.append((w-l-r)*(h-top-b))
        count=sum(areas);path=bundle.path/'pointcloud'/'global_raw.ply'
        if count!=bundle.metadata['cloud_counts']['raw'] or not path.exists():bundle.bulk_raw_unavailable=True;return None
        with (bundle.path/'per_frame_metrics.csv').open() as f:records=list(csv.DictReader(f))
        if [r['name'] for r in records]!=bundle.metadata['input_names']:bundle.bulk_raw_unavailable=True;return None
        with path.open('rb') as f:
            header=[]
            while True:
                line=f.readline()
                if not line:bundle.bulk_raw_unavailable=True;return None
                header.append(line.decode('ascii').strip())
                if header[-1]=='end_header':break
            dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
            properties=[s for s in header if s.startswith('property ')]
            if properties!=['property float x','property float y','property float z','property uchar red','property uchar green','property uchar blue'] or 'format binary_little_endian 1.0' not in header or f'element vertex {count}' not in header or path.stat().st_size-f.tell()!=count*dtype.itemsize:
                bundle.bulk_raw_unavailable=True;return None
            print(f'  {bundle.path.name}: sequentially reading complete raw PLY ({count} original pixels)',flush=True)
            data=np.fromfile(f,dtype=dtype,count=count)
        bundle.bulk_raw=(data,np.cumsum([0]+areas))
        # Independent source-map and RGB probe guards the exporter convention.
        t=transforms[0];w,h=t['input_size_wh'];l,top,r,b=t['padding_ltrb'];rw=w-l-r;rh=h-top-b
        positions=np.asarray([[0,0],[rh//2,rw//2],[rh-1,rw-1]])
        indices=positions[:,0]*rw+positions[:,1]
        packed_xyz=np.column_stack([data[k][indices] for k in ('x','y','z')]);packed_rgb=np.column_stack([data[k][indices] for k in ('red','green','blue')])
        first=Path(bundle.metadata['input_names'][0]).stem
        with saved_npz(bundle.path/'raw_predictions'/f'{first}.npz') as p:original=p['world_points'].reshape(h,w,3)
        if original.dtype!=np.float32:
            del bundle.bulk_raw;bundle.bulk_raw_unavailable=True;return None
        source_rgb=np.asarray(Image.open(bundle.path/'rgb'/f'{first}.png'))
        source_ij=positions+np.asarray([top,l])
        exact_xyz=np.array_equal(packed_xyz,original[source_ij[:,0],source_ij[:,1]])
        exact_rgb=np.array_equal(packed_rgb,source_rgb[source_ij[:,0],source_ij[:,1]])
        if not exact_xyz or not exact_rgb:raise RuntimeError('Raw PLY reconstruction failed original XYZ/RGB probe; no derived results accepted')
        json_write(bundle.path/'visualizations'/'bulk_raw_reconstruction_audit.json',dict(source=str(path),vertices=count,
            original_frame_order_verified=True,finite_unpadded_pixel_count_proof=True,original_xyz_dtype='float32',first_frame_three_pixel_xyz_exact=exact_xyz,
            first_frame_three_pixel_rgb_exact=exact_rgb,source_pixel_coordinates_yx=source_ij,
            interpretation='Exact original row-major unpadded XYZ/RGB recovery; confidence and masks remain original files; padding invalid; no geometry changes'))
    i=bundle.metadata['input_names'].index(name);t=bundle.pixel_transforms[i];w,h=t['input_size_wh'];l,top,r,b=t['padding_ltrb']
    data,offsets=bundle.bulk_raw;part=data[offsets[i]:offsets[i+1]];rh=h-top-b;rw=w-l-r
    xyz=np.full((h,w,3),np.nan,np.float32);rgb=np.zeros((h,w,3),np.uint8)
    xyz[top:h-b,l:w-r]=np.column_stack([part[k] for k in ('x','y','z')]).reshape(rh,rw,3)
    rgb[top:h-b,l:w-r]=np.column_stack([part[k] for k in ('red','green','blue')]).reshape(rh,rw,3)
    with saved_npz(bundle.path/'confidence'/f'{Path(name).stem}.npz') as p:conf=p['world_points_conf'].reshape(h,w)
    mask=np.asarray(Image.open(bundle.path/'mask'/name))>0;finite=np.isfinite(xyz).all(-1)&np.isfinite(conf)
    return dict(name=name,pred={'world_points':xyz,'world_points_conf':conf},xyz=xyz,conf=conf,mask=mask,rgb=rgb,valid=finite&(conf>1),finite=finite,
        c2w=bundle.cameras['c2w'][i],k=bundle.cameras['intrinsics_model'][i],index=i)

def to_camera(xyz,c2w):
    if c2w is None or not np.isfinite(c2w).all() or abs(np.linalg.det(c2w[:3,:3]))<1e-12:
        return np.full(np.asarray(xyz).shape,np.nan,dtype=float)
    w2c=np.linalg.inv(c2w)
    return xyz@w2c[:3,:3].T+w2c[:3,3]

def valid_camera_indices(bundle):
    poses=np.asarray(bundle.cameras['c2w'])
    finite=np.isfinite(poses).all(axis=(1,2))
    indices=np.flatnonzero(finite)
    return np.array([i for i in indices if abs(np.linalg.det(poses[i,:3,:3]))>1e-12],dtype=int)

def common_registered_names(bundles):
    """Retained unmapped names are not valid common registration controls."""
    return set.intersection(*(set(b.metadata['input_names'][int(i)] for i in valid_camera_indices(b)) for b in bundles.values())) if bundles else set()

def anchor_pose(bundle):
    if not hasattr(bundle,'display_anchor_index'):
        indices=valid_camera_indices(bundle);bundle.display_anchor_index=int(indices[0]) if len(indices) else None
    return None if bundle.display_anchor_index is None else bundle.cameras['c2w'][bundle.display_anchor_index]

def to_display_reference(xyz,bundle):
    pose=anchor_pose(bundle)
    return np.asarray(xyz) if pose is None else to_camera(xyz,pose)

def projected(xyz,c2w,k):
    cam=to_camera(xyz,c2w);xy=cam[...,:2]/np.maximum(cam[...,2:3],1e-12)
    return xy@k[:2,:2].T+k[:2,2],cam[...,2]

def sample_pixels(frame,target=6000,randomized=False):
    valid=frame['mask']&frame['valid']
    coords=np.argwhere(valid)
    if randomized and len(coords)>target:
        seed=int(Path(frame['name']).stem.split('_')[-1])
        return coords[np.random.default_rng(seed).choice(len(coords),size=target,replace=False)]
    stride=max(1,math.ceil(len(coords)/target))
    return coords[::stride]

def valid_points(frame,target=6000):
    ij=sample_pixels(frame,target)
    return frame['xyz'][ij[:,0],ij[:,1]],ij

def scene_normalization(bundle):
    clouds=[]
    # Every available frame contributes; fixed deterministic display/metric sample.
    for name in bundle.metadata['input_names']:
        frame=load_frame(bundle,name);pts,_=valid_points(frame,1000);clouds.append(pts)
        if len(clouds)%32==0:print(f'  {bundle.path.name}: cached {len(clouds)}/{len(bundle.metadata["input_names"])} original maps',flush=True)
    allpoints=np.concatenate(clouds) if clouds else np.empty((0,3))
    if not len(allpoints):return np.zeros(3),1.,allpoints
    centered=to_display_reference(allpoints,bundle)
    center=np.median(centered,axis=0)
    lo,hi=np.quantile(centered,[.025,.975],axis=0)
    scale=float(np.max(hi-lo));scale=max(scale,1e-12)
    return center,scale,allpoints

def safe_quantile(values,q):
    a=np.asarray(values);a=a[np.isfinite(a)]
    return float(np.quantile(a,q)) if a.size else None

def safe_mean(values):
    a=np.asarray([v for v in values if v is not None],dtype=float);a=a[np.isfinite(a)]
    return float(a.mean()) if a.size else None

def edge_discontinuity(frame):
    cam=to_camera(frame['xyz'],frame['c2w']);z=cam[...,2]
    m=frame['mask']&frame['valid']&(z>0)
    discontinuities=[]
    for axis in (0,1):
        a=np.take(z,np.arange(z.shape[axis]-1),axis=axis);b=np.take(z,np.arange(1,z.shape[axis]),axis=axis)
        ma=np.take(m,np.arange(z.shape[axis]-1),axis=axis);mb=np.take(m,np.arange(1,z.shape[axis]),axis=axis)
        sel=ma&mb;discontinuities.append((np.abs(a-b)/np.maximum(np.minimum(a,b),1e-12))[sel])
    values=np.concatenate(discontinuities)
    return float((values>.05).mean()) if len(values) else None

def plane_roughness(frame,scale):
    points,_=valid_points(frame,1500)
    if len(points)<12:return None
    tree=cKDTree(points);_,index=tree.query(points,k=min(12,len(points)),workers=1)
    neighborhoods=points[index];centered=neighborhoods-neighborhoods.mean(axis=1,keepdims=True)
    cov=np.einsum('nki,nkj->nij',centered,centered)/index.shape[1]
    eigenvalues=np.linalg.eigvalsh(cov)
    return float(np.median(np.sqrt(np.maximum(eigenvalues[:,0],0)))/scale)

def adjacent_consistency(source,target,scale):
    pts,ij=valid_points(source,6000)
    if not len(pts):return dict(projected=0,overlap=0,visible=0)
    uv,z=projected(pts,target['c2w'],target['k']);finite=np.isfinite(uv).all(-1)&(z>0)
    uv=np.rint(np.nan_to_num(uv,nan=-1e8,posinf=-1e8,neginf=-1e8)).astype(np.int64)
    h,w=target['mask'].shape
    inside=finite&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    index=np.flatnonzero(inside);uvinside=uv[index]
    overlap=target['mask'][uvinside[:,1],uvinside[:,0]]&target['valid'][uvinside[:,1],uvinside[:,0]]
    index=index[overlap];uvinside=uv[index]
    predicted_target=target['xyz'][uvinside[:,1],uvinside[:,0]]
    target_z=to_camera(predicted_target,target['c2w'])[:,2]
    # Exclude points hidden behind the target-predicted foreground surface.
    visible=(target_z>0)&(z[index]<=target_z*1.05)
    index=index[visible];target_z=target_z[visible];predicted_target=predicted_target[visible]
    distance=np.linalg.norm(pts[index]-predicted_target,axis=-1)/scale
    relative_depth=np.abs(z[index]-target_z)/np.maximum(target_z,1e-12)
    return dict(projected=len(pts),overlap=int(overlap.sum()),visible=len(index),
        distance_p50=safe_quantile(distance,.5),distance_p90=safe_quantile(distance,.9),
        depth_p50=safe_quantile(relative_depth,.5),depth_agreement_fraction=float((relative_depth<=.05).mean()) if len(index) else None)

def world_nn_points(a,b,scale):
    if not len(a) or not len(b):return dict(world_nn_p50=None,world_nn_p90=None,source_sample_points=len(a),target_sample_points=len(b))
    distances=np.concatenate((cKDTree(b).query(a,k=1,workers=1)[0],cKDTree(a).query(b,k=1,workers=1)[0]))/scale
    return dict(world_nn_p50=safe_quantile(distances,.5),world_nn_p90=safe_quantile(distances,.9),source_sample_points=len(a),target_sample_points=len(b))

def frame_nn_points(frame):
    ij=sample_pixels(frame,6000,randomized=True)
    return frame['xyz'][ij[:,0],ij[:,1]]

def augment_world_nn(bundle,summary):
    """Pose-independent supplementary proxy, using identical saved point samples."""
    if 'adjacent_world_nn_p50_normalized' in summary:return summary
    samples={}
    for name in bundle.metadata['input_names']:
        if name in getattr(bundle,'frame_cache',{}):samples[name]=frame_nn_points(bundle.frame_cache[name])
        else:
            packed=sample_saved_concat_cloud(bundle,{name},6000)
            samples[name]=packed[0] if packed is not None else frame_nn_points(load_frame(bundle,name))
    rows=[];names=bundle.metadata['input_names'];scale=summary['normalization_extent']
    for a,b in zip(names[:-1],names[1:]):rows.append(dict(source=a,target=b,**world_nn_points(samples[a],samples[b],scale)))
    summary['adjacent_world_nn_p50_normalized']=safe_quantile([r['world_nn_p50'] for r in rows if r['world_nn_p50'] is not None],.5)
    summary['adjacent_world_nn_p90_normalized']=safe_quantile([r['world_nn_p90'] for r in rows if r['world_nn_p90'] is not None],.5)
    summary['adjacent_world_nn_available_pairs']=sum(r['world_nn_p50'] is not None for r in rows)
    summary['world_nn_semantics']='Symmetric neighboring world-cloud NN; at most6000 source-frame-seeded raw foreground points/view, same extent. No known surface correspondences/visibility; density, occlusion and collapsed planes affect this proxy. Not GT accuracy or a stand-alone winner criterion.'
    table_write(bundle.path/'geometry_world_nn_pairs.csv',rows)
    return summary

def per_frame_metrics(frame,scale,transform):
    m=frame['mask'];edge=m&~binary_erosion(m);valid=frame['valid'];xyz=frame['xyz'];conf=frame['conf']
    camera=to_camera(xyz,frame['c2w']);foreground=m&valid;positive=foreground&(camera[...,2]>0)
    pixels=np.indices(m.shape);uv,_=projected(xyz[positive],frame['c2w'],frame['k'])
    expected=np.stack((pixels[1][positive],pixels[0][positive]),axis=-1)
    error=np.linalg.norm(uv-expected,axis=-1)
    def mean_conf(sel):return float(np.mean(conf[sel&frame['finite']])) if (sel&frame['finite']).any() else None
    q85=safe_quantile(conf[frame['finite']],.85)
    valid_q85=frame['valid']&(conf>=q85) if q85 is not None else np.zeros_like(m)
    row=dict(name=frame['name'],foreground_pixels=int(m.sum()),foreground_valid=int(foreground.sum()),edge_pixels=int(edge.sum()),edge_valid=int((edge&valid).sum()),
        finite_foreground_pixels=int((m&frame['finite']).sum()),
        foreground_q85_valid=int((m&valid_q85).sum()),edge_q85_valid=int((edge&valid_q85).sum()),confidence_q85=q85,
        confidence_floor_fg_fraction=float(((conf<=1)&m&frame['finite']).sum()/max((m&frame['finite']).sum(),1)),
        confidence_q85_tie_fraction=float(((conf==q85)&frame['finite']).sum()/max(frame['finite'].sum(),1)),
        confidence_q85_retained_valid_fraction=float(valid_q85.sum()/max(valid.sum(),1)),
        foreground_coverage=float(foreground.sum()/max(m.sum(),1)),edge_coverage=float((edge&valid).sum()/max(edge.sum(),1)),
        positive_depth_fraction=float(positive.sum()/max(foreground.sum(),1)),fg_conf_mean=mean_conf(m),bg_conf_mean=mean_conf(~m),edge_conf_mean=mean_conf(edge),
        confidence_fg_p10=safe_quantile(conf[m],.1),confidence_fg_p50=safe_quantile(conf[m],.5),confidence_fg_p90=safe_quantile(conf[m],.9),
        self_reprojection_p50_px=safe_quantile(error,.5),self_reprojection_p90_px=safe_quantile(error,.9),
        depth_neighbor_jump_ratio=edge_discontinuity(frame),local_plane_residual_normalized=plane_roughness(frame,scale))
    if frame['name']==ROI_FRAME:
        a=np.asarray(transform['pixel_transform']);inv=np.linalg.inv(a)
        y,x=np.indices(m.shape);original=np.stack((x,y,np.ones_like(x)),axis=-1)@inv.T
        for name,box in ROIS.items():
            l,t,r,b=box;roi=(original[...,0]>=l)&(original[...,0]<r)&(original[...,1]>=t)&(original[...,1]<b)
            selected=roi&m;selected_edge=roi&edge
            row.update({f'{name}_roi_pixels':int(selected.sum()),f'{name}_roi_valid':int((selected&valid).sum()),
                f'{name}_roi_edge_pixels':int(selected_edge.sum()),f'{name}_roi_edge_valid':int((selected_edge&valid).sum()),
                f'{name}_roi_q85_coverage':float((selected&valid_q85).sum()/max(selected.sum(),1)),
                f'{name}_roi_q85_edge_coverage':float((selected_edge&valid_q85).sum()/max(selected_edge.sum(),1)),
                f'{name}_roi_coverage':float((selected&valid).sum()/max(selected.sum(),1)),
                f'{name}_roi_edge_coverage':float((selected_edge&valid).sum()/max(selected_edge.sum(),1))})
    return row

def pose_statistics(bundle,scale):
    pose=np.asarray(bundle.cameras['c2w']);translation=np.linalg.norm(np.diff(pose[:,:3,3],axis=0),axis=-1)/scale
    relative=np.einsum('nij,njk->nik',pose[:-1,:3,:3].transpose(0,2,1),pose[1:,:3,:3])
    rotation=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    status_file=bundle.path/'pose_status.json';status=json.loads(status_file.read_text()) if status_file.exists() else None
    success=None;eligible=[];unknown=len(bundle.metadata['input_names']);scope='not applicable: pose-head predictions or SfM registration'
    kind=bundle.metadata.get('method',bundle.metadata.get('metadata',{}).get('method',bundle.path.name))
    if kind.startswith('Fast3R') and 'SfM' not in kind and isinstance(status,list):
        eligible=[s.get('status') for s in status if s.get('status') in ('success','failed','failure','error','failed_identity_fallback')]
        unknown=len(status)-len(eligible);scope='official direct Fast3R camera solver status; not registration quality'
    elif kind=='Fast3R_AMB3R_SfM' and isinstance(status,list):
        eligible=[s.get('last_local_pnp_status') for s in status if s.get('accepted_local_call_ids') and s.get('last_local_pnp_status') in ('success','failed','failure','error','failed_identity_fallback')]
        unknown=len(status)-len(eligible);scope='last accepted local Fast3R PnP contributor per frame; selected contributors only; not final fused-pose quality or all-local-call success rate'
    if eligible:success=sum(s=='success' for s in eligible)/len(eligible)
    valid_indices=valid_camera_indices(bundle)
    deterror=np.abs(np.linalg.det(pose[valid_indices,:3,:3])-1) if len(valid_indices) else np.array([])
    orthogonality=np.linalg.norm(np.einsum('nij,njk->nik',pose[valid_indices,:3,:3].transpose(0,2,1),pose[valid_indices,:3,:3])-np.eye(3),axis=(1,2)) if len(valid_indices) else np.array([])
    return dict(translation_step_p50_normalized=safe_quantile(translation,.5),translation_step_p90_normalized=safe_quantile(translation,.9),
        rotation_step_p50_deg=safe_quantile(rotation,.5),rotation_step_p90_deg=safe_quantile(rotation,.9),pnp_success_rate=success,
        pnp_success_rate_scope=scope,pnp_eligible_status_count=len(eligible),pnp_unknown_or_inapplicable_frame_count=unknown,
        finite_camera_count=len(valid_indices),pose_rotation_det_error_max=float(deterror.max()) if len(deterror) else None,
        pose_rotation_orthogonality_max=float(orthogonality.max()) if len(orthogonality) else None)

def table_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def enrich_summary(summary,bundle):
    """Refresh run provenance independently of reusable geometry metrics."""
    meta=json.loads((bundle.path/'bundle.json').read_text()).get('metadata',{})
    for key in ('inference_seconds','inference_time_scope','total_frontend_seconds','total_frontend_time_scope','peak_vram_gib','registered_frames','unmapped_frames','scene_metric_scale_predicted','world_points_metric_derived_gauge_validated'):
        summary[key]=meta.get(key)
    summary['registered_frames']=meta.get('registered_frames',meta.get('mapped_frames'))
    states=json.loads((bundle.path/'pose_status.json').read_text())
    kind=bundle.metadata.get('method',meta.get('method',bundle.path.name))
    if 'SfM' in kind and isinstance(states,list):
        observed=sum(s.get('status')=='registered' for s in states)
        summary['registration_status_registered_count']=observed
        summary['registration_count_matches_metadata']=summary['registered_frames']==observed
        keyframes_path=bundle.path/'keyframes.json'
        if keyframes_path.exists():
            keyframes=json.loads(keyframes_path.read_text()).get('indices',[])
            summary['keyframe_count']=len(keyframes)
            summary['keyframe_count_scope']='final saved official SfMemory keyframes.json indices; coarse-stage temporary counts excluded'
    if not summary.get('inference_time_scope'):summary['inference_time_scope']=meta.get('timing_scope',{}).get('pipeline')
    for key in ('local_model_seconds','local_raw_export_seconds','local_calls','local_call_count','none_return_count','backend_fallback_count','peak_vram_reserved_gib'):
        summary[key]=meta.get(key)
    summary['local_calls']=meta.get('local_calls',meta.get('local_prediction_calls'))
    summary['none_return_count']=meta.get('none_return_count',meta.get('official_benchmark_early_stop_calls'))
    summary['backend_fallback_count']=meta.get('backend_fallback_count',meta.get('backend_frontend_fallback_count'))
    summary['peak_vram_reserved_gib']=meta.get('peak_vram_reserved_gib',meta.get('peak_reserved_vram_gib'))
    for key in ('bundle_export_seconds','post_pipeline_seconds','elapsed_seconds_including_model_load','elapsed_seconds_through_bundle_export'):
        summary[key]=meta.get(key)
    if summary.get('total_frontend_seconds') is None and meta.get('elapsed_seconds_through_bundle_export') is not None:
        summary['total_frontend_seconds']=meta['elapsed_seconds_through_bundle_export']
        summary['total_frontend_time_scope']='recorded runner wall time from model loading/weight hashing through CPU bundle completion; excludes earlier shared input preparation, later persistent copy/verification and evaluation'
    counts=meta.get('observed_pnp_status_counts_over_all_local_calls')
    observation_missing=meta.get('pnp_instrumentation_incomplete_views')
    if counts is None and kind=='Fast3R_AMB3R_SfM' and (bundle.path/'local_call_index.json').exists():
        index_path=bundle.path/'local_call_index.json';index_bytes=index_path.read_bytes();calls=json.loads(index_bytes)
        counts={};observation_missing=0
        for call in calls:
            statuses=call.get('pnp_status',[])
            for record in statuses:
                state=record.get('status','unknown');counts[state]=counts.get(state,0)+1
            observed=call.get('geometry_validation',{}).get('pnp_observation',{})
            if statuses or call.get('status')=='predicted':
                expected=observed.get('expected_views',len(call.get('frame_indices',[])))
                recorded=observed.get('recorded_views',len(statuses))
                observation_missing+=max(0,expected-recorded)
        audit=dict(source=str(index_path.resolve()),source_sha256=hashlib.sha256(index_bytes).hexdigest(),
            local_calls=len(calls),observed_status_counts=counts,missing_observation_views=observation_missing,
            interpretation='Observed original local-call solver records, including returning predictions not accepted by final SfMemory; registration and final fused-pose accuracy are separate.')
        json_write(bundle.path/'pnp_all_local_call_audit.json',audit)
    if isinstance(counts,dict):
        successes=counts.get('success',0);failures=counts.get('failed_identity_fallback',0);known=successes+failures
        summary['all_local_call_pnp_success_rate']=successes/known if known else None
        summary['all_local_call_pnp_eligible_count']=known
        summary['all_local_call_pnp_failed_identity_fallback_count']=failures
        summary['all_local_call_pnp_observed_incomplete_count']=counts.get('not_called',0)+counts.get('started',0)
        summary['all_local_call_pnp_missing_observation_views']=observation_missing
        summary['all_local_call_pnp_scope']='all returning local model calls, including calls not accepted in final memory; eligible=success+failed_identity_fallback; incompleteness separate; not global fused-pose quality'
    summary.update(pose_statistics(bundle,summary['normalization_extent']))
    if not summary.get('inference_time_scope'):
        summary['inference_time_scope']='official forward only; export and evaluation excluded' if bundle.path.name.startswith('AMB3R_Base') else 'not recorded; refer to original run report'
    if not summary.get('total_frontend_time_scope'):
        summary['total_frontend_time_scope']='not measured as a combined frontend time'
    summary['registered_frame_fraction']=summary['registered_frames']/max(summary['views'],1) if isinstance(summary.get('registered_frames'),int) else None
    diagnostic=bundle.path/'head_consistency_diagnostic.json'
    if diagnostic.exists():
        head=json.loads(diagnostic.read_text()).get('all_frames_view_medians',{})
        summary['pointhead_camera_reprojection_all_foreground_p50_px']=head.get('direct_pointhead_reprojection_median_px')
        summary['depth_unprojection_self_reprojection_p50_px']=head.get('unprojection_reprojection_median_px')
        summary['pointhead_over_unprojection_worldnorm_p50']=head.get('median_worldnorm_pointwise_ratio')
    json_write(bundle.path/'geometry_metrics.json',summary)
    return summary

def evaluate_bundle(bundle):
    bundle.enable_bulk_raw=True
    center,scale,allpoints=scene_normalization(bundle)
    transforms=json.loads((bundle.path/'pixel_transforms.json').read_text())
    by_name={t['name']:t for t in transforms};rows=[];pair_rows=[];previous=None
    for name in bundle.metadata['input_names']:
        frame=load_frame(bundle,name);rows.append(per_frame_metrics(frame,scale,by_name[name]))
        if previous is not None:
            for src,tgt in ((previous,frame),(frame,previous)):
                pair_rows.append(dict(source=src['name'],target=tgt['name'],**adjacent_consistency(src,tgt,scale)))
        previous=frame
        if len(rows)%32==0:print(f'  {bundle.path.name}: {len(rows)}/{len(bundle.metadata["input_names"])} views',flush=True)
    table_write(bundle.path/'geometry_per_frame.csv',rows);table_write(bundle.path/'geometry_adjacent_pairs.csv',pair_rows)
    fg=sum(r['foreground_pixels'] for r in rows);edge=sum(r['edge_pixels'] for r in rows)
    summary=dict(method=bundle.path.name,status='complete',views=len(rows),full_sequence=bundle.metadata.get('metadata',{}).get('full_sequence',False),
        scale_semantics=bundle.metadata['scale'],foreground_pixels=fg,foreground_points=sum(r['foreground_valid'] for r in rows),
        foreground_coverage=sum(r['foreground_valid'] for r in rows)/max(fg,1),edge_coverage=sum(r['edge_valid'] for r in rows)/max(edge,1),
        foreground_q85_coverage=sum(r['foreground_q85_valid'] for r in rows)/max(fg,1),edge_q85_coverage=sum(r['edge_q85_valid'] for r in rows)/max(edge,1),
        foreground_points_per_foreground_pixel=sum(r['foreground_valid'] for r in rows)/max(fg,1),
        raw_global_points=bundle.metadata['cloud_counts']['raw'],
        foreground_finite_ratio=sum(r['finite_foreground_pixels'] for r in rows)/max(fg,1),
        fg_conf_mean=safe_mean([r['fg_conf_mean'] for r in rows]),bg_conf_mean=safe_mean([r['bg_conf_mean'] for r in rows]),edge_conf_mean=safe_mean([r['edge_conf_mean'] for r in rows]),
        confidence_fg_p50=safe_quantile([r['confidence_fg_p50'] for r in rows if r['confidence_fg_p50'] is not None],.5),
        low_confidence_frames_ratio=safe_mean([(1-1/r['confidence_fg_p50'])<.2 for r in rows if r['confidence_fg_p50'] is not None]),
        confidence_baseline_keeps_almost_all_frames_ratio=float(np.mean([r['foreground_coverage']>=.999 for r in rows])),
        confidence_floor_fg_fraction=float(np.mean([r['confidence_floor_fg_fraction'] for r in rows])),
        confidence_q85_tie_fraction=float(np.mean([r['confidence_q85_tie_fraction'] for r in rows])),
        confidence_q85_retained_valid_fraction=float(np.mean([r['confidence_q85_retained_valid_fraction'] for r in rows])),
        self_reprojection_p50_px=safe_quantile([r['self_reprojection_p50_px'] for r in rows if r['self_reprojection_p50_px'] is not None],.5),
        depth_neighbor_jump_ratio=safe_quantile([r['depth_neighbor_jump_ratio'] for r in rows if r['depth_neighbor_jump_ratio'] is not None],.5),
        local_plane_residual_normalized=safe_quantile([r['local_plane_residual_normalized'] for r in rows if r['local_plane_residual_normalized'] is not None],.5),
        adjacent_point_distance_p50_normalized=safe_quantile([p['distance_p50'] for p in pair_rows if p.get('distance_p50') is not None],.5),
        adjacent_point_distance_p90_normalized=safe_quantile([p['distance_p90'] for p in pair_rows if p.get('distance_p90') is not None],.5),
        adjacent_depth_agreement_fraction=safe_quantile([p['depth_agreement_fraction'] for p in pair_rows if p.get('depth_agreement_fraction') is not None],.5),
        adjacent_projection_overlap_fraction=sum(p['overlap'] for p in pair_rows)/max(sum(p['projected'] for p in pair_rows),1),
        normalization_extent=scale,**pose_statistics(bundle,scale))
    if len(allpoints)>1:
        spacing=cKDTree(allpoints).query(allpoints,k=2,workers=1)[0][:,1]/scale
        summary['sample_nearest_spacing_p50_normalized']=float(np.median(spacing))
        summary['sample_close_point_ratio_001extent']=float((spacing<.001).mean())
        median=float(np.median(spacing));mad=float(np.median(np.abs(spacing-median)))
        cutoff=median+6*max(mad,1e-12)
        summary['sample_spacing_outlier_ratio_6mad']=float((spacing>cutoff).mean())
        summary['sample_spacing_outlier_cutoff_normalized']=cutoff
    for row in rows:
        if row['name']==ROI_FRAME:
            for k,v in row.items():
                if '_roi_' in k:summary[k]=v
    meta=bundle.metadata.get('metadata',{})
    summary['inference_seconds']=meta.get('inference_seconds');summary['peak_vram_gib']=meta.get('peak_vram_gib')
    summary['registered_frames']=meta.get('registered_frames');summary['unmapped_frames']=meta.get('unmapped_frames')
    summary['scene_metric_scale_predicted']=meta.get('scene_metric_scale_predicted')
    frame_scales=meta.get('frame_metric_scales_predicted')
    if frame_scales is not None:
        summary['frame_metric_scale_p10']=safe_quantile(frame_scales,.1)
        summary['frame_metric_scale_p50']=safe_quantile(frame_scales,.5)
        summary['frame_metric_scale_p90']=safe_quantile(frame_scales,.9)
    json_write(bundle.path/'geometry_metrics.json',summary)
    anchor=anchor_pose(bundle);anchor_name=bundle.metadata['input_names'][bundle.display_anchor_index] if bundle.display_anchor_index is not None else None
    normalization=dict(first_camera_w2c=np.linalg.inv(anchor) if anchor is not None else None,display_anchor_source_name=anchor_name,center_first_camera=center,extent=scale,
        display_anchor_note='common valid source camera selected across methods; raw world coordinates only if no valid camera exists',
        use='display and dimensionless predicted-geometry metrics only; no exported coordinate changes and no GT/metric scale fitting',
        extent_rule='max axis extent of 2.5–97.5% foreground coordinates; each frame contributes up to1000 original predicted points')
    json_write(bundle.path/'visualizations'/'normalization.json',normalization)
    return summary,rows,(center,scale)

def cached_metrics(bundle):
    """Reuse computed diagnostics of an immutable completed bundle on request."""
    paths=[bundle.path/'geometry_metrics.json',bundle.path/'geometry_per_frame.csv',bundle.path/'visualizations'/'normalization.json']
    if not all(p.exists() for p in paths):return None
    summary=json.loads(paths[0].read_text())
    if 'edge_q85_coverage' not in summary:return None
    with open(paths[1]) as f:raw=list(csv.DictReader(f))
    if [r['name'] for r in raw]!=bundle.metadata['input_names']:return None
    rows=[]
    for rawrow in raw:
        row={}
        for key,value in rawrow.items():
            if key=='name':row[key]=value
            elif not value:row[key]=None
            else:
                try:row[key]=int(value)
                except ValueError:row[key]=float(value)
        rows.append(row)
    normalization=json.loads(paths[2].read_text())
    anchor=anchor_pose(bundle);anchor_name=bundle.metadata['input_names'][bundle.display_anchor_index] if bundle.display_anchor_index is not None else None
    old_anchor=normalization.get('display_anchor_source_name')
    if old_anchor is None and anchor_name==bundle.metadata['input_names'][0] and normalization.get('first_camera_w2c') is not None:
        normalization['display_anchor_source_name']=anchor_name;json_write(paths[2],normalization)
    elif old_anchor!=anchor_name:
        if sample_saved_concat_cloud(bundle,{bundle.metadata['input_names'][0]},1) is None:return None
        data,counts=bundle.packed_cloud;clouds=[];offset=0
        for count in counts:
            stride=max(1,math.ceil(int(count)/1000));index=offset+np.arange(0,count,stride,dtype=int)
            clouds.append(np.column_stack([data[k][index] for k in ('x','y','z')]));offset+=count
        world=np.concatenate(clouds);reference=to_display_reference(world,bundle)
        center=np.median(reference,axis=0) if len(reference) else np.zeros(3)
        extent=max(float(np.ptp(np.quantile(reference,[.025,.975],axis=0),axis=0).max()),1e-12) if len(reference) else 1.
        factor=float(normalization['extent'])/extent
        for key,value in list(summary.items()):
            if key.endswith('_normalized') and isinstance(value,(int,float)):summary[key]=value*factor
        for row in rows:
            for key,value in list(row.items()):
                if key.endswith('_normalized') and isinstance(value,(int,float)):row[key]=value*factor
        for filename,fields in [('geometry_adjacent_pairs.csv',('distance_p50','distance_p90')),('geometry_world_nn_pairs.csv',('world_nn_p50','world_nn_p90'))]:
            p=bundle.path/filename
            if not p.exists():continue
            with p.open() as f:pair_rows=list(csv.DictReader(f))
            for r in pair_rows:
                for field in fields:
                    if r.get(field):r[field]=float(r[field])*factor
            table_write(p,pair_rows)
        if len(world)>1:
            spacing=cKDTree(world).query(world,k=2,workers=1)[0][:,1]/extent
            median=float(np.median(spacing));mad=float(np.median(np.abs(spacing-median)));cutoff=median+6*max(mad,1e-12)
            summary.update(sample_nearest_spacing_p50_normalized=median,sample_close_point_ratio_001extent=float((spacing<.001).mean()),sample_spacing_outlier_ratio_6mad=float((spacing>cutoff).mean()),sample_spacing_outlier_cutoff_normalized=cutoff)
        summary['normalization_extent']=extent
        summary['display_anchor_source_name']=anchor_name
        summary['normalization_rebase_note']='Exact original per-frame1000-point row-stride samples recovered from complete concat PLY; new camera anchor/center/extent; dimensionless distances rescaled, spacing proxies recomputed; raw predictions/cameras unchanged.'
        backup=paths[2].with_name(f'normalization_before_anchor_{Path(old_anchor).stem if old_anchor else "none"}.json')
        if not backup.exists():backup.write_bytes(paths[2].read_bytes())
        normalization.update(first_camera_w2c=np.linalg.inv(anchor) if anchor is not None else None,display_anchor_source_name=anchor_name,center_first_camera=center,extent=extent,
            rebase_from_source_name=old_anchor,dimensionless_metric_scale_factor=factor)
        json_write(paths[2],normalization);table_write(paths[1],rows);json_write(paths[0],summary)
    if 'confidence_floor_fg_fraction' not in summary or 'foreground_finite_ratio' not in summary:
        transforms=json.loads((bundle.path/'pixel_transforms.json').read_text())
        expected_raw=sum((t['input_size_wh'][0]-t['padding_ltrb'][0]-t['padding_ltrb'][2])*(t['input_size_wh'][1]-t['padding_ltrb'][1]-t['padding_ltrb'][3]) for t in transforms)
        # Safe confidence-only augmentation when the saved raw count proves
        # every unpadded image pixel has finite XYZ and confidence already.
        if bundle.metadata['cloud_counts']['raw']!=expected_raw:return None
        if bundle.metadata['cloud_counts']['confidence']==expected_raw or summary.get('foreground_coverage')==1:
            # The raw count proves finite unpadded predictions. Complete
            # foreground baseline coverage proves no FG confidence floor and
            # unchanged FG/edge q85 controls; BG tie statistics remain unknown.
            summary['confidence_floor_fg_fraction']=0.
            summary['foreground_finite_ratio']=1.
            summary['confidence_q85_tie_fraction']=None
            summary['confidence_q85_retained_valid_fraction']=None
            summary['confidence_tie_statistics']='legacy cached q85 retained; exact tie fraction not recomputed'
            for row in rows:row['finite_foreground_pixels']=row['foreground_pixels'];row['confidence_floor_fg_fraction']=0.
            table_write(paths[1],rows);json_write(paths[0],summary)
            return summary,rows,(np.asarray(normalization['center_first_camera']),float(normalization['extent']))
        for row,t in zip(rows,transforms):
            stem=Path(row['name']).stem
            with saved_npz(bundle.path/'confidence'/f'{stem}.npz') as f:conf=f['world_points_conf']
            mask=np.asarray(Image.open(bundle.path/'mask'/f'{stem}.png'))>0;conf=conf.reshape(mask.shape)
            left,top,right,bottom=t['padding_ltrb'];valid_pixels=np.zeros(mask.shape,bool);valid_pixels[top:mask.shape[0]-bottom,left:mask.shape[1]-right]=True
            valid=valid_pixels&(conf>1);q=row['confidence_q85'];retained=valid&(conf>=q)
            row['confidence_floor_fg_fraction']=float(((conf<=1)&mask&valid_pixels).sum()/max((mask&valid_pixels).sum(),1))
            row['finite_foreground_pixels']=int((mask&valid_pixels).sum())
            row['confidence_q85_tie_fraction']=float(((conf==q)&valid_pixels).sum()/max(valid_pixels.sum(),1))
            row['confidence_q85_retained_valid_fraction']=float(retained.sum()/max(valid.sum(),1))
            edge=mask&~binary_erosion(mask)
            row['foreground_q85_valid']=int((mask&retained).sum());row['edge_q85_valid']=int((edge&retained).sum())
            if row['name']==ROI_FRAME:
                y,x=np.indices(mask.shape);original=np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(t['pixel_transform']).T
                for roi,box in ROIS.items():
                    l,top,r,bottom=box;selected=(original[...,0]>=l)&(original[...,0]<r)&(original[...,1]>=top)&(original[...,1]<bottom)&mask
                    selected_edge=selected&edge
                    row[f'{roi}_roi_q85_coverage']=float((selected&retained).sum()/max(selected.sum(),1))
                    row[f'{roi}_roi_q85_edge_coverage']=float((selected_edge&retained).sum()/max(selected_edge.sum(),1))
                    summary[f'{roi}_roi_q85_coverage']=row[f'{roi}_roi_q85_coverage'];summary[f'{roi}_roi_q85_edge_coverage']=row[f'{roi}_roi_q85_edge_coverage']
        for key in ('confidence_floor_fg_fraction','confidence_q85_tie_fraction','confidence_q85_retained_valid_fraction'):
            summary[key]=float(np.mean([r[key] for r in rows]))
        summary['foreground_q85_coverage']=sum(r['foreground_q85_valid'] for r in rows)/max(sum(r['foreground_pixels'] for r in rows),1)
        summary['edge_q85_coverage']=sum(r['edge_q85_valid'] for r in rows)/max(sum(r['edge_pixels'] for r in rows),1)
        summary['foreground_finite_ratio']=sum(r['finite_foreground_pixels'] for r in rows)/max(sum(r['foreground_pixels'] for r in rows),1)
        table_write(paths[1],rows);json_write(paths[0],summary)
    return summary,rows,(np.asarray(normalization['center_first_camera']),float(normalization['extent']))

def display_points(bundle,norm,names=None,per_frame=None):
    center,scale=norm;clouds=[];colors=[]
    packed=sample_saved_concat_cloud(bundle,names,per_frame)
    if packed is not None:
        xyz,col=packed
        return (to_display_reference(xyz,bundle)-center)/scale,col
    if per_frame is None:
        count=len(bundle.metadata['input_names']) if names is None else len(names)
        per_frame=max(1,250000//max(count,1))
    for name in bundle.metadata['input_names']:
        if names is not None and name not in names:continue
        frame=load_frame(bundle,name);ij=sample_pixels(frame,per_frame,randomized=True);pts=frame['xyz'][ij[:,0],ij[:,1]]
        clouds.append((to_display_reference(pts,bundle)-center)/scale);colors.append(frame['rgb'][ij[:,0],ij[:,1]]/255.)
    return (np.concatenate(clouds),np.concatenate(colors)) if clouds else (np.empty((0,3)),np.empty((0,3)))

def sample_saved_concat_cloud(bundle,names=None,per_frame=None):
    """Use the complete unfused Bundle PLY after validating frame offsets.

    The legacy fused cloud and capped visualization PLY are never used.
    Each packed point equals the row-major confidence+mask selected source point.
    """
    path=bundle.path/'pointcloud'/'global_mask_confidence.ply';records_path=bundle.path/'per_frame_metrics.csv'
    if not path.exists() or not records_path.exists():return None
    if not hasattr(bundle,'packed_cloud'):
        with records_path.open() as f:records=list(csv.DictReader(f))
        if [r['name'] for r in records]!=bundle.metadata['input_names']:return None
        counts=np.asarray([int(r['valid_foreground_points']) for r in records]);count=int(counts.sum())
        if count!=bundle.metadata['cloud_counts']['mask_confidence']:return None
        with path.open('rb') as f:
            lines=[]
            while True:
                line=f.readline()
                if not line:return None
                lines.append(line.decode('ascii').strip())
                if lines[-1]=='end_header':break
            properties=[s for s in lines if s.startswith('property ')]
            required=['property float x','property float y','property float z','property uchar red','property uchar green','property uchar blue']
            if 'format binary_little_endian 1.0' not in lines or f'element vertex {count}' not in lines or properties!=required:return None
            dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
            if path.stat().st_size-f.tell()!=count*dtype.itemsize:return None
            data=np.fromfile(f,dtype=dtype,count=count)
        bundle.packed_cloud=(data,counts)
        json_write(bundle.path/'visualizations'/'packed_cloud_sampling.json',dict(source=str(path),vertices=count,
            source_order_verified=True,per_frame_counts_verified=True,record_bytes=15,source_provenance='complete original per-frame concatenation; not fused or capped',sampling='random source-frame-seeded per-view sampling matching raw-mask row-major order; display only'))
    data,counts=bundle.packed_cloud
    selected=[];offset=0;quota=per_frame or max(1,250000//max(len(names) if names is not None else len(counts),1))
    for name,count in zip(bundle.metadata['input_names'],counts):
        if names is None or name in names:
            seed=int(Path(name).stem.split('_')[-1]);index=np.random.default_rng(seed).choice(int(count),size=quota,replace=False) if count>quota else np.arange(count)
            selected.append(index+offset)
        offset+=count
    index=np.concatenate(selected).astype(int) if selected else np.empty(0,dtype=int)
    points=np.column_stack([data[k][index] for k in ('x','y','z')]);col=np.column_stack([data[k][index] for k in ('red','green','blue')])/255.
    return points,col

def manual_gt_figures(bundles,norms,output):
    """Manual2D GT versus global raw foreground clouds in5source cameras."""
    import matplotlib.pyplot as plt
    methods=list(bundles);available=[i for i in GT_INDICES if (GT_ROOT/f'gt_{i:04d}.png').exists() and (GT_ROOT/f'{i:04d}.jpg').exists()]
    if not available:return
    world_clouds={}
    for name,b in bundles.items():
        packed=sample_saved_concat_cloud(b)
        if packed is not None:
            world_clouds[name]=packed
            continue
        coords=[];colors=[];limit=max(1,250000//max(b.metadata['frame_count'],1))
        for frame_name in b.metadata['input_names']:
            frame=load_frame(b,frame_name);ij=sample_pixels(frame,limit,randomized=True)
            coords.append(frame['xyz'][ij[:,0],ij[:,1]]);colors.append(frame['rgb'][ij[:,0],ij[:,1]]/255.)
        world_clouds[name]=(np.concatenate(coords),np.concatenate(colors))
    fig,axes=plt.subplots(len(available),len(methods)+1,figsize=(2.5*(len(methods)+1),4.1*len(available)),squeeze=False)
    references=[]
    for i,source_index in enumerate(available):
        gt=np.asarray(Image.open(GT_ROOT/f'gt_{source_index:04d}.png'))>0
        photo=np.asarray(Image.open(GT_ROOT/f'{source_index:04d}.jpg').convert('RGB'));assert photo.shape[:2]==gt.shape
        masked=photo.copy();masked[~gt]=0
        axes[i,0].imshow(masked);axes[i,0].contour(gt,levels=[.5],colors='white',linewidths=.3)
        axes[i,0].set_title(f'Manual2D GT / photo\nsource{source_index:04d}',fontsize=9);axes[i,0].axis('off')
        name=f'crop_{source_index:04d}.png'
        references.append(dict(name=name,gt_mask=str(GT_ROOT/f'gt_{source_index:04d}.png'),photo=str(GT_ROOT/f'{source_index:04d}.jpg'),manual_annotation=str(GT_ROOT/f'{source_index:04d}.json'),size_wh=[gt.shape[1],gt.shape[0]],gt_3d=False))
        for j,method in enumerate(methods,1):
            b=bundles[method];ax=axes[i,j];ax.set_facecolor('black');ax.set_xlim(0,gt.shape[1]);ax.set_ylim(gt.shape[0],0);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{method}\nsource{source_index:04d}',fontsize=9)
            if name not in b.metadata['input_names']:
                ax.text(.5,.5,'Source view not present',color='white',ha='center',transform=ax.transAxes);continue
            idx=b.metadata['input_names'].index(name);pose=b.cameras['c2w'][idx]
            if not np.isfinite(pose).all():
                ax.text(.5,.5,'Unmapped source view',color='white',ha='center',transform=ax.transAxes);continue
            xyz,col=world_clouds[method];uv,z=projected(xyz,pose,b.cameras['intrinsics_original'][idx])
            valid=np.isfinite(uv).all(-1)&(z>0)&(uv[:,0]>=0)&(uv[:,0]<gt.shape[1])&(uv[:,1]>=0)&(uv[:,1]<gt.shape[0])
            # Farther points are drawn first with identical parameters; this is
            # a pointcloud projection, not a trained renderer orGT alignment.
            idxpoints=np.flatnonzero(valid);idxpoints=idxpoints[np.argsort(z[idxpoints])[::-1]]
            ax.scatter(uv[idxpoints,0],uv[idxpoints,1],c=col[idxpoints],s=PLOT_POINT_SIZE,marker='.',linewidths=0,rasterized=True)
            ax.contour(gt,levels=[.5],colors='white',linewidths=.3)
            if not len(idxpoints):ax.text(.5,.5,'No visible valid points',color='white',ha='center',transform=ax.transAxes)
    fig.suptitle('Five manually annotated2D views | global foreground pointclouds projected with each model camera\nWhite contour is manualGT; identical colors/point size/background; no measured3D GT or pose alignment',fontsize=10)
    fig.tight_layout(rect=[0,0,1,.975]);fig.savefig(Path(output)/'manual_GT_five_view_geometry_comparison.png',dpi=150);fig.savefig(Path(output)/'manual_GT_five_view_geometry_comparison.pdf');plt.close(fig)
    json_write(Path(output)/'manual_GT_reference_manifest.json',dict(references=references,rendering='same sampled whole foreground cloud projected through each source model C2W/K_original; confidence>1+RAPmask;250000points/methodmaximum',point_size=PLOT_POINT_SIZE,ground_truth='manual2D masks and matching original photos only; no3D scan orGT camera registration'))

def point_plot(ax,points,colors,view):
    if view=='front':x,y=points[:,0],-points[:,1]
    elif view=='side':x,y=points[:,2],-points[:,1]
    else:x,y=points[:,0],points[:,2]
    ax.scatter(x,y,c=colors,s=PLOT_POINT_SIZE,marker='.',linewidths=0,rasterized=True)
    ax.set_xlim(-.9,.9);ax.set_ylim(-.9,.9);ax.set_aspect('equal');ax.set_facecolor('black');ax.set_xticks([]);ax.set_yticks([])

def figures(bundles,norms,output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'savefig.facecolor':'white'})
    methods=list(bundles);n=len(methods)
    if not n:return
    clouds={name:display_points(bundle,norms[name]) for name,bundle in bundles.items()}
    fig,axes=plt.subplots(3,n,figsize=(3*n,8),squeeze=False)
    for j,name in enumerate(methods):
        bundle=bundles[name];pts,col=clouds[name]
        for i,view in enumerate(('front','side','top')):
            point_plot(axes[i,j],pts,col,view)
            if i==0:axes[i,j].set_title(f'{name}\n{bundle.metadata["frame_count"]} views')
            if j==0:axes[i,j].set_ylabel(view)
        single,sax=plt.subplots(1,3,figsize=(9,3))
        for ax,view in zip(sax,('front','side','top')):point_plot(ax,pts,col,view);ax.set_title(view)
        single.suptitle(f'{name}: display-only normalization; identical point size/background')
        single.tight_layout();single.savefig(bundle.path/'visualizations'/'whole_plant_three_views.png',dpi=160);plt.close(single)
    fig.suptitle('Foreground outputs: Mask + confidence>1\nDisplay-only normalization; identical rendering; no GT alignment',fontsize=10)
    fig.tight_layout();fig.savefig(output/'whole_plant_comparison.png',dpi=180);fig.savefig(output/'whole_plant_comparison.pdf');plt.close(fig)
    shared=common_registered_names(bundles)
    json_write(output/'render_parameters.json',dict(point_size=PLOT_POINT_SIZE,background='black',color='original RGB',limits=[-.9,.9],views=['front x,-y','side z,-y','top x,z'],
        sampling='up to250000 original predicted points per method, equally divided among available views; uniform sampling seeded by original frame number; display only',matched_input_names=sorted(shared),normalization='independent display extent, fixed first input camera axes'))
    if shared:
        fig,axes=plt.subplots(1,n,figsize=(3*n,3),squeeze=False)
        for j,name in enumerate(methods):
            pts,col=display_points(bundles[name],norms[name],shared);point_plot(axes[0,j],pts,col,'front');axes[0,j].set_title(name)
        fig.suptitle(f'Matched source frames only: {len(shared)} common views; same display rules')
        fig.tight_layout();fig.savefig(output/'matched_views_front.png',dpi=180);plt.close(fig)
    # Identical source-pixel ROIs on each method's pointmaps; no 3D crop registration.
    roi_methods=[name for name in methods if ROI_FRAME in bundles[name].metadata['input_names']]
    if roi_methods:
        source=Image.open('/data/fj/F2DMAS/03-SAM/KongQueZhuYu/'+ROI_FRAME).convert('RGB')
        fig,axes=plt.subplots(2,len(roi_methods)+1,figsize=(3*(len(roi_methods)+1),6),squeeze=False)
        for i,(roi,box) in enumerate(ROIS.items()):
            axes[i,0].imshow(source.crop(box));axes[i,0].set_title(f'Source RGB: {roi}');axes[i,0].axis('off')
            for j,name in enumerate(roi_methods,1):
                b=bundles[name];frame=load_frame(b,ROI_FRAME);t=json.loads((b.path/'pixel_transforms.json').read_text())[frame['index']];inv=np.linalg.inv(t['pixel_transform'])
                y,x=np.indices(frame['mask'].shape);original=np.stack((x,y,np.ones_like(x)),axis=-1)@inv.T;l,top,r,bottom=box
                sel=(original[...,0]>=l)&(original[...,0]<r)&(original[...,1]>=top)&(original[...,1]<bottom)&frame['mask']&frame['valid']
                axes[i,j].scatter(original[...,0][sel],-original[...,1][sel],c=frame['rgb'][sel]/255.,s=1.,marker='.',linewidths=0)
                axes[i,j].set_xlim(l,r);axes[i,j].set_ylim(-bottom,-top);axes[i,j].set_aspect('equal');axes[i,j].set_facecolor('black');axes[i,j].set_title(f'{name}: point support');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        fig.suptitle('Same original-pixel leaf ROIs: foreground point support, not a registered 3D accuracy test')
        fig.tight_layout();fig.savefig(output/'leaf_and_edge_original_pixel_rois.png',dpi=180);plt.close(fig)
        # Inspect real leaf geometry in front and side projections, selecting the
        # same original-pixel window for every method without fitting surfaces.
        fig,axes=plt.subplots(4,len(roi_methods),figsize=(3*len(roi_methods),10),squeeze=False)
        for j,name in enumerate(roi_methods):
            bundle=bundles[name];frame=load_frame(bundle,ROI_FRAME)
            transform=json.loads((bundle.path/'pixel_transforms.json').read_text())[frame['index']]
            inv=np.linalg.inv(transform['pixel_transform']);y,x=np.indices(frame['mask'].shape)
            original=np.stack((x,y,np.ones_like(x)),axis=-1)@inv.T
            for i,(roi,box) in enumerate(ROIS.items()):
                l,top,r,bottom=box
                sel=(original[...,0]>=l)&(original[...,0]<r)&(original[...,1]>=top)&(original[...,1]<bottom)&frame['mask']&frame['valid']
                pts=to_camera(frame['xyz'][sel],frame['c2w']);col=frame['rgb'][sel]/255.
                center=np.median(pts,axis=0) if len(pts) else np.zeros(3);pts=(pts-center)/norms[name][1]
                radius=.22 if roi=='leaf' else .12
                for k,view in enumerate(('front','side')):
                    ax=axes[2*i+k,j];point_plot(ax,pts,col,view);ax.set_xlim(-radius,radius);ax.set_ylim(-radius,radius)
                    ax.set_title(f'{name}: {roi} {view}')
                    if not len(pts):ax.text(.5,.5,'No points pass\nMask + conf>1',color='white',ha='center',va='center',transform=ax.transAxes)
        fig.suptitle('Leaf geometry from identical source-pixel ROIs; ROI median centered for display only; common whole-scene extent fractions')
        fig.tight_layout();fig.savefig(output/'leaf_and_edge_geometry.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(4,len(roi_methods),figsize=(3*len(roi_methods),10),squeeze=False)
        for j,name in enumerate(roi_methods):
            bundle=bundles[name];frame=load_frame(bundle,ROI_FRAME);transform=bundle.pixel_transforms[frame['index']]
            y,x=np.indices(frame['mask'].shape);original=np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(transform['pixel_transform']).T
            for i,(roi,box) in enumerate(ROIS.items()):
                l,top,r,bottom=box;sel=(original[...,0]>=l)&(original[...,0]<r)&(original[...,1]>=top)&(original[...,1]<bottom)&frame['mask']&frame['valid']
                pts=to_camera(frame['xyz'][sel],frame['c2w']);center=np.median(pts,axis=0) if len(pts) else np.zeros(3);pts=(pts-center)/norms[name][1]
                gray=np.full((len(pts),3),.82);radius=.22 if roi=='leaf' else .12
                for k,view in enumerate(('front','side')):
                    ax=axes[2*i+k,j];point_plot(ax,pts,gray,view);ax.set_xlim(-radius,radius);ax.set_ylim(-radius,radius);ax.set_title(f'{name}: {roi} {view}',fontsize=9)
                    if not len(pts):ax.text(.5,.5,'No points pass\nMask + conf>1',color='white',ha='center',va='center',transform=ax.transAxes)
        fig.suptitle('Geometry-only leaf/edge diagnostic: identical constant gray, point size and ROI ranges\nMask + confidence>1; original RGB predictions are unchanged',fontsize=10)
        fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(output/'leaf_and_edge_geometry_uniform_gray.png',dpi=180);fig.savefig(output/'leaf_and_edge_geometry_uniform_gray.pdf');plt.close(fig)
        # Shared confidence color range in sigmoid-like representation c -> (c-1)/c.
        fig,axes=plt.subplots(1,len(roi_methods),figsize=(3*len(roi_methods),5),squeeze=False)
        for j,name in enumerate(roi_methods):
            frame=load_frame(bundles[name],ROI_FRAME);sig=1-1/np.maximum(frame['conf'],1e-12)
            im=axes[0,j].imshow(sig,cmap='viridis',vmin=0,vmax=1);axes[0,j].contour(frame['mask'],levels=[.5],colors='white',linewidths=.3);axes[0,j].set_title(name);axes[0,j].axis('off')
            single,ax=plt.subplots(figsize=(3,5));h=ax.imshow(sig,cmap='viridis',vmin=0,vmax=1);ax.contour(frame['mask'],levels=[.5],colors='white',linewidths=.3);ax.axis('off');ax.set_title(f'{name}: confidence (c-1)/c');single.colorbar(h,ax=ax,fraction=.04,pad=.02);single.tight_layout();single.savefig(bundles[name].path/'visualizations'/'confidence_heatmap.png',dpi=180);plt.close(single)
        fig.suptitle('Confidence (c-1)/c; common [0,1] scale; numerical scores are not cross-model calibrated')
        fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.02);fig.savefig(output/'confidence_heatmap_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,n,figsize=(3*n,3),squeeze=False)
    for j,name in enumerate(methods):
        b=bundles[name];center,scale=norms[name];centers=b.cameras['c2w'][:,:3,3];pts=(to_display_reference(centers,b)-center)/scale
        axes[0,j].plot(pts[:,0],pts[:,2],'-',lw=.8);axes[0,j].scatter(pts[:,0],pts[:,2],c=np.arange(len(pts)),cmap='plasma',s=5);axes[0,j].set_title(name);axes[0,j].set_aspect('equal');axes[0,j].set_xlim(-1.5,1.5);axes[0,j].set_ylim(-1.5,1.5);axes[0,j].set_xlabel('x / display extent');axes[0,j].set_ylabel('z / display extent')
    fig.suptitle('Camera centers; sequence color; dimensionless display normalization')
    fig.tight_layout();fig.savefig(output/'camera_trajectory_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for name,bundle in bundles.items():
        with open(bundle.path/'geometry_per_frame.csv') as f:rows=list(csv.DictReader(f))
        index=np.array([int(Path(r['name']).stem.split('_')[-1]) for r in rows])
        confidence=np.array([float(r['confidence_fg_p50']) if r['confidence_fg_p50'] else np.nan for r in rows])
        axes[0].plot(index,1-1/np.maximum(confidence,1e-12),label=name,lw=.9)
        pose=bundle.cameras['c2w'];relative=np.einsum('nij,njk->nik',pose[:-1,:3,:3].transpose(0,2,1),pose[1:,:3,:3])
        rotation=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
        axes[1].plot(index[1:],rotation,label=name,lw=.9)
    axes[0].axhline(.2,color='gray',ls='--',lw=.8);axes[0].set_ylim(0,1);axes[0].set_ylabel('Foreground median (c-1)/c');axes[0].legend(fontsize=8)
    axes[1].set_ylim(0,180);axes[1].set_ylabel('Successive-view rotation (degrees)');axes[1].set_xlabel('Original source frame number')
    fig.suptitle('Confidence and camera continuity over actual input order; sparse subsets have larger frame gaps')
    fig.tight_layout();fig.savefig(output/'confidence_and_rotation_over_sequence.png',dpi=180);plt.close(fig)
    # The requested GT column can only show image evidence without registered 3D GT.
    fig,axes=plt.subplots(1,n+1,figsize=(3*(n+1),3),squeeze=False)
    source=Image.open('/data/fj/F2DMAS/03-SAM/KongQueZhuYu/'+ROI_FRAME).convert('RGB');axes[0,0].imshow(source);axes[0,0].set_title('Source RGB / silhouette reference\nNo registered 3D GT');axes[0,0].axis('off')
    for j,name in enumerate(methods,1):point_plot(axes[0,j],*clouds[name],'front');axes[0,j].set_title(name)
    fig.suptitle('Source image reference | predicted geometry\nNo registered ground-truth pointcloud',fontsize=10)
    fig.tight_layout();fig.savefig(output/'source_reference_and_frontends.png',dpi=180);plt.close(fig)
    # Local vs global diagnostic uses same predicted source view. Local head scales
    # are normalized independently only in this clearly labeled diagnostic panel.
    for name,bundle in bundles.items():
        indices=valid_camera_indices(bundle)
        if not len(indices):continue
        frame=load_frame(bundle,bundle.metadata['input_names'][int(indices[0])],include_local=True);local=frame['pred'].get('local_points')
        if local is None:continue
        local=np.asarray(local).reshape(frame['xyz'].shape);sel=frame['mask']&frame['valid']&np.isfinite(local).all(-1)
        if not sel.any():continue
        col=frame['rgb'][sel]/255.;global_pts=to_camera(frame['xyz'][sel],frame['c2w']);local_pts=local[sel]
        fig,axes=plt.subplots(1,2,figsize=(6,3))
        for ax,pts,label in zip(axes,(global_pts,local_pts),('global map in viewing camera','local camera map')):
            center=np.median(pts,axis=0);lo,hi=np.quantile(pts,[.025,.975],axis=0);extent=max(float(np.max(hi-lo)),1e-12)
            stride=max(1,len(pts)//30000);point_plot(ax,(pts[::stride]-center)/extent,col[::stride],'side');ax.set_title(label)
        fig.suptitle('Individual maps: independent diagnostic display scales; no map alignment or fusion')
        fig.tight_layout();fig.savefig(bundle.path/'visualizations'/'global_local_side.png',dpi=180);plt.close(fig)
    manual_gt_figures(bundles,norms,output)

def fmt(value):
    if value is None:return '—'
    if isinstance(value,float):return f'{value:.4g}'
    return str(value)

def paired_geometry_controls(off,on,output):
    """Isolate changed geometry from the OFF/ON confidence-selection change."""
    names=off.metadata['input_names']
    if names!=on.metadata['input_names']:return []
    off.enable_bulk_raw=True;on.enable_bulk_raw=True
    original={off.path.name:[load_frame(off,n) for n in names],on.path.name:[load_frame(on,n) for n in names]}
    results=[]
    for control in ('shared_baseline_valid','mask_finite_without_confidence'):
        selected=[]
        for a,b in zip(original[off.path.name],original[on.path.name]):
            gate=a['finite']&b['finite']
            if control=='shared_baseline_valid':gate&=a['valid']&b['valid']
            selected.append(gate)
        for bundle in (off,on):
            frames=[dict(f,valid=g) for f,g in zip(original[bundle.path.name],selected)]
            samples=[]
            for f in frames:
                xyz,_=valid_points(f,1000);samples.append(xyz)
            points=np.concatenate(samples);reference=to_display_reference(points,bundle)
            scale=max(float(np.ptp(np.quantile(reference,[.025,.975],axis=0),axis=0).max()),1e-12) if len(reference) else 1.
            pairs=[];nn_pairs=[];previous=None;discontinuities=[]
            for f in frames:
                discontinuities.append(edge_discontinuity(f))
                if previous is not None:
                    for src,tgt in ((previous,f),(f,previous)):pairs.append(dict(source=src['name'],target=tgt['name'],**adjacent_consistency(src,tgt,scale)))
                    nn_pairs.append(dict(source=previous['name'],target=f['name'],**world_nn_points(frame_nn_points(previous),frame_nn_points(f),scale)))
                previous=f
            row=dict(method=bundle.path.name,control=control,views=len(frames),selection_identical_between_off_on=True,
                selected_foreground_fraction=sum((f['mask']&f['valid']).sum() for f in frames)/max(sum(f['mask'].sum() for f in frames),1),
                normalization_extent=scale,adjacent_depth_agreement_fraction=safe_quantile([p['depth_agreement_fraction'] for p in pairs if p.get('depth_agreement_fraction') is not None],.5),
                adjacent_point_distance_p50_normalized=safe_quantile([p['distance_p50'] for p in pairs if p.get('distance_p50') is not None],.5),
                projection_overlap_fraction=sum(p['overlap'] for p in pairs)/max(sum(p['projected'] for p in pairs),1),
                adjacent_world_nn_p50_normalized=safe_quantile([p['world_nn_p50'] for p in nn_pairs if p['world_nn_p50'] is not None],.5),
                adjacent_world_nn_p90_normalized=safe_quantile([p['world_nn_p90'] for p in nn_pairs if p['world_nn_p90'] is not None],.5),
                depth_neighbor_jump_ratio=safe_quantile([v for v in discontinuities if v is not None],.5),
                interpretation='Point-head geometry only; identical per-frame pixels, original masks/poses/maps; display extent per method; no point averaging or alignment; not measured3D accuracy')
            results.append(row);table_write(bundle.path/f'geometry_pair_control_{control}_pairs.csv',pairs)
            table_write(bundle.path/f'geometry_pair_control_{control}_world_nn.csv',nn_pairs)
    table_write(Path(output)/'amb3r_off_on_geometry_controls.csv',results)
    json_write(Path(output)/'amb3r_off_on_geometry_controls.json',results)
    return results

def pair_control_signature(off,on):
    """Cheap identity guard for an already completed immutable bundle evaluation."""
    result=[]
    for bundle in (off,on):
        identities=[]
        for name in bundle.metadata['input_names']:
            for directory,suffix in (('raw_predictions','.npz'),('rgb','.png'),('mask','.png')):
                path=bundle.path/directory/(Path(name).stem+suffix)
                stat=path.stat()
                identities.append([str(path.resolve()),stat.st_size,stat.st_mtime_ns])
        result.append(dict(names=bundle.metadata['input_names'],seed=bundle.metadata.get('metadata',{}).get('seed'),
            raw_rgb_mask_identities_sha256=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),
            camera_sha256=hashlib.sha256((bundle.path/'cameras'/'cameras.npz').read_bytes()).hexdigest(),
            transforms_sha256=hashlib.sha256((bundle.path/'pixel_transforms.json').read_bytes()).hexdigest()))
    return result

def reports(output,summaries,bundles,frame_rows):
    output=Path(output);table_write(output/'comparison.csv',summaries)
    fields=['method','status','views','foreground_coverage','edge_coverage','edge_q85_coverage','adjacent_depth_agreement_fraction','adjacent_point_distance_p50_normalized','rotation_step_p90_deg','pnp_success_rate','inference_seconds','total_frontend_seconds','peak_vram_gib']
    text=['# KongQueZhuYu geometry comparison','',
        'These are diagnostics of predicted geometry. No scanned GT has been registered, no true scale correction has been applied, and no 3DGS training or PSNR evaluation has been run.',
        '', '| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']
    for row in summaries:text.append('| '+' | '.join(fmt(row.get(k)) for k in fields)+' |')
    text += ['', 'Resolution and scope differ between the existing VGGT-Ω artifact, full-sequence frontends and any fixed-frame AMB3R OFF/ON test. Point counts alone are not a quality ranking. The confidence >1 baseline can retain almost every pixel because both confidence heads use 1+exp; geometry consistency and leaf diagnostics carry more information.',
        '', '`edge_q85_coverage` is the boundary of the complete foreground mask, including plant, pot and blue object. It is not a measured leaf-edge score. The separate leaf and leaf-edge-neighborhood ROIs of source0000 are shared rectangular windows; their q85 coverage measures retained foreground pixels, not annotated3D leaf-edge accuracy.',
        '', 'Self-reprojection is a pointmap/camera agreement diagnostic. It is algebraically zero for depth-unprojected VGGT-Ω points and must not be compared as a learned geometry accuracy score.',
        '', 'See `visualizations/render_parameters.json` and `docs/EVALUATION_PROTOCOL.md` for fixed plotting parameters, confidence semantics, source-pixel ROIs, visibility filtering and metric limitations.']
    (output/'comparison.md').write_text('\n'.join(text))
    if bundles:
        shared=common_registered_names(bundles)
        matched=[]
        for name,rows in frame_rows.items():
            subset=[r for r in rows if r['name'] in shared];fg=sum(r['foreground_pixels'] for r in subset);edge=sum(r['edge_pixels'] for r in subset)
            pairs=[];prior=None;bundle=bundles[name]
            scale=next(r['normalization_extent'] for r in summaries if r['method']==name)
            all_pairs=bundle.path/'geometry_adjacent_pairs.csv'
            if shared==set(bundle.metadata['input_names']) and all_pairs.exists():
                with open(all_pairs) as f:
                    for raw in csv.DictReader(f):
                        pair={}
                        for key,value in raw.items():
                            pair[key]=value if key in ('source','target') else (float(value) if value else None)
                        pairs.append(pair)
            else:
                for frame_name in bundle.metadata['input_names']:
                    if frame_name not in shared:continue
                    frame=load_frame(bundle,frame_name)
                    if prior is not None:
                        for src,tgt in ((prior,frame),(frame,prior)):
                            pairs.append(dict(source=src['name'],target=tgt['name'],**adjacent_consistency(src,tgt,scale)))
                    prior=frame
            if pairs:table_write(bundle.path/'geometry_matched_pairs.csv',pairs)
            matched.append(dict(method=name,common_frame_count=len(subset),common_registered_frame_count=len(shared),comparison_status='available' if shared else 'unavailable_no_common_registered_frames',foreground_coverage=sum(r['foreground_valid'] for r in subset)/max(fg,1),edge_coverage=sum(r['edge_valid'] for r in subset)/max(edge,1),
                foreground_q85_coverage=sum(r['foreground_q85_valid'] for r in subset)/max(fg,1),edge_q85_coverage=sum(r['edge_q85_valid'] for r in subset)/max(edge,1),
                depth_neighbor_jump_ratio=safe_quantile([r['depth_neighbor_jump_ratio'] for r in subset if r['depth_neighbor_jump_ratio'] is not None],.5),
                adjacent_depth_agreement_fraction=safe_quantile([p['depth_agreement_fraction'] for p in pairs if p.get('depth_agreement_fraction') is not None],.5),
                adjacent_point_distance_p50_normalized=safe_quantile([p['distance_p50'] for p in pairs if p.get('distance_p50') is not None],.5),
                projection_overlap_fraction=sum(p['overlap'] for p in pairs)/max(sum(p['projected'] for p in pairs),1)))
        table_write(output/'comparison_matched_frames.csv',matched);json_write(output/'matched_frame_names.json',sorted(shared))
        matched_fields=['method','common_frame_count','edge_q85_coverage','adjacent_depth_agreement_fraction','adjacent_point_distance_p50_normalized','projection_overlap_fraction']
        text += ['', '## Same source frames and same evaluation pairs', '',
            'The control uses only source cameras registered and finite in every method and recomputes identical bidirectional pairs in original order. Source-index gaps remain explicit in pair CSV. Primary full208 coverage retains unmapped frames as failures. An empty common registered intersection makes this control unavailable. Full-sequence and subset inference remain distinct scopes.', '',
            '| '+' | '.join(matched_fields)+' |','| '+' | '.join(['---']*len(matched_fields))+' |']
        text += ['', 'Timing is scoped in comparison.csv. Fast3R total includes model+PnP and excludes export; AMB3R Base records forward only; SfM pipeline includes registration/local serialization when recorded. Shared disk contention affects wall time. Manual GT columns are original-pixel2D footprint support, not measured3D accuracy.']
        text.extend('| '+' | '.join(fmt(r.get(k)) for k in matched_fields)+' |' for r in matched)
        (output/'comparison.md').write_text('\n'.join(text))
    for name,bundle in bundles.items():
        row=next(r for r in summaries if r['method']==name);meta=bundle.metadata.get('metadata',{})
        report=[f'# RUN REPORT: {name}','',f'- Saved views: {row["views"]}; full sequence: {row.get("full_sequence")}',
            f'- Scale: {bundle.metadata["scale"]}',f'- Geometry export: {bundle.metadata["pointcloud_processing"]}',
            f'- Weight SHA256: {meta.get("weight_sha256", "not recorded by prior artifact")}',f'- Source commit: {meta.get("git_commit", "not recorded")}',
            f'- Inference seconds: {fmt(row.get("inference_seconds"))}; peak GPU memory GiB: {fmt(row.get("peak_vram_gib"))}',
            f'- Inference timing scope: {row.get("inference_time_scope")}',
            f'- Total frontend seconds: {fmt(row.get("total_frontend_seconds"))}; scope: {row.get("total_frontend_time_scope")}',
            f'- Recorded bundle export / post-pipeline seconds: {fmt(row.get("bundle_export_seconds"))} / {fmt(row.get("post_pipeline_seconds"))}; pipeline-local model / raw serialization seconds: {fmt(row.get("local_model_seconds"))} / {fmt(row.get("local_raw_export_seconds"))}. These intervals have different scopes and are not summed as independent algorithm timings.',
            '- Full model parameters and original model report: `run_parameters.json` and `MODEL_RUN_REPORT.md` when saved. Raw model outputs and original cameras are unchanged.',
            '', '## Geometry diagnostics','']
        report.extend(f'- {key}: {fmt(row.get(key))}' for key in fields[3:])
        report += [f'- Whole-foreground q85 boundary coverage includes the plant, pot and blue object; shared rectangular leaf/edge ROI retention is reported separately.',
            f'- Finite foreground fraction: {fmt(row.get("foreground_finite_ratio"))}; camera count: {fmt(row.get("finite_camera_count"))}.',
            f'- Registered / unmapped frames: {fmt(row.get("registered_frames"))} / {fmt(row.get("unmapped_frames"))}; final keyframes: {fmt(row.get("keyframe_count"))}.',
            f'- PnP status scope: {row.get("pnp_success_rate_scope")}; eligible observations: {fmt(row.get("pnp_eligible_status_count"))}.',
            f'- All returning local-call PnP rate: {fmt(row.get("all_local_call_pnp_success_rate"))}; eligible observations: {fmt(row.get("all_local_call_pnp_eligible_count"))}; missing observations: {fmt(row.get("all_local_call_pnp_missing_observation_views"))}.',
            '- Solver success and successful mapping do not establish accuracy of final fused cameras or thin-leaf geometry.',
            f'- Adjacent world-cloud NN, median / p90 normalized: {fmt(row.get("adjacent_world_nn_p50_normalized"))} / {fmt(row.get("adjacent_world_nn_p90_normalized"))}. This proximity proxy is affected by density, occlusion and collapsed surfaces; it is not GT accuracy.']
        if row.get('scene_metric_scale_predicted') is not None:
            report += ['', '## Learned metric-scale prediction', '',
                f'- Normalized-to-metric factor predicted by the metric head: {fmt(row["scene_metric_scale_predicted"])}',
                f'- Frame factors p10 / median / p90: {fmt(row.get("frame_metric_scale_p10"))} / {fmt(row.get("frame_metric_scale_p50"))} / {fmt(row.get("frame_metric_scale_p90"))}',
                '- Official depth_metric is a learned metre prediction without measured verification. Direct point-head, depth and pose gauges are not demonstrated equal. Multiplying direct pointmaps by a depth-head factor is a conditional derivative, not validated metric geometry. Raw global pointmaps/camera translations retain their original normalized scale.']
        if row.get('pointhead_camera_reprojection_all_foreground_p50_px') is not None:
            report += ['', '## Point-head / depth-head agreement', '',
                f'- All-foreground point-head/camera reprojection: {fmt(row["pointhead_camera_reprojection_all_foreground_p50_px"])}px; depth-unprojection algebraic self-reprojection: {fmt(row.get("depth_unprojection_self_reprojection_p50_px"))}px.',
                '- These are consistency checks between saved outputs. They do not isolate camera error or establish accuracy against real 3D geometry. See `head_consistency_diagnostic.json`.']
        if (bundle.path/'depth_unprojection_diagnostic.json').exists():
            report += ['', '## Original depth branch control', '',
                '- See `depth_unprojection_diagnostic.json`, `depth_diagnostic_native_depth_confidence_pairs.csv` and the experiment-level `amb3r_depth_channel_comparison.csv` for the actual full208 OFF/ON depth-unprojection comparison.',
                '- Native depth-confidence selection evaluates that original branch. The separate world-confidence selection is a geometry-control experiment, not a claim that the point-head confidence describes all AMB3R geometry.',
                '- Depth-unprojected points are already world-frame points and are not a local point head. Algebraic self-reprojection does not validate real camera/depth accuracy. Official depth_metric metre predictions remain unverified by external length measurements.']
        if row.get('gt_reference_frames'):
            report += ['', '## Existing manual 2D GT support', '',
                f'- Original-view GT silhouette support recall: {fmt(row.get("manual_gt_2d_support_recall"))}; inner-edge recall: {fmt(row.get("manual_gt_2d_edge_support_recall"))}.',
                '- This measures the original-pixel footprint of valid RAP foreground geometry. The GT masks are manual 2D references and are not measured 3D surfaces.']
        report += ['', 'All per-frame metrics are in `geometry_per_frame.csv`, neighboring-view metrics in `geometry_adjacent_pairs.csv`, and aggregate values in `geometry_metrics.json`.',
            '', 'Coverage measures saved valid foreground pixels, not unseen surfaces. Local plane residual and depth jumps mix noise, curvature, thin surfaces and occlusion boundaries; they are not absolute reconstruction error. Confidence is not calibrated between models. This bundle has no claim of validated GGGS/COLMAP tracks and no training was started.']
        (bundle.path/'RUN_REPORT.md').write_text('\n'.join(report))
    by={r['method']:r for r in summaries};required=[m for m in METHODS if by.get(m,{}).get('status')!='complete']
    analysis=['# ForeSplat frontend decision: KongQueZhuYu','',
        'Decisions use saved real predictions only. Missing or failed workflows remain missing; another model does not substitute for them.',
        '',f'Incomplete/failed required workflows: {", ".join(required) if required else "none"}.',
        '', '## Questions and evidence','',
        '1. **Fast3R versus VGGT-Ω:** compare foreground and leaf-edge support, neighboring-view depth agreement, normalized map distance, trajectory and the common-frame figures. The legacy VGGT-Ω artifact is depth-based and has higher input resolution. A score advantage is only a predicted-geometry consistency advantage; it is not proof of GT accuracy.',
        '2. **AMB3R backend OFF versus ON:** the OFF/ON pair must have exactly identical frame names, transforms and RGB/masks. Evaluate pointmap changes, leaf-edge support, depth jumps and adjacent consistency. A shared full-input OFF/ON artifact is required before declaring a backend gain.',
        '3. **AMB3R-SfM global registration:** use registered/unmapped frame counts and official registration logs. Full global mapping success means organization/registration completed; it does not itself validate every thin leaf surface.',
        '4. **Fast3R + AMB3R-SfM:** compare mapping failures and matched-frame consistency against one-pass Fast3R. Each cluster reference is local until official registration; concatenation of raw cluster outputs is not accepted as global geometry.',
        '5. **Leaf-edge retention:** the original-pixel rectangular leaf/edge ROIs of `crop_0000.png` are shared across methods. They are evaluation ROIs chosen by directly viewing the source RGB, not leaf instance annotations. Edge coverage does not determine which sheet is reconstructed correctly.',
        '6. **3DGS initialization:** choose only after valid camera convention, sufficient foreground support, acceptable registration rate, adjacent geometry and thin-leaf diagnostics agree. No 3DGS training was started; current bundles expose camera/pointmap inputs for subsequent validation.',
        '7. **Metric measurement:** AMB3R metric head provides learned metric-scale predictions when saved. There is no external length measurement in this evaluation. Fast3R and VGGT-Ω scales are ambiguous. Metric output is a measurement candidate, not a demonstrated measurement accuracy.',
        '', '## Current decision','']
    if required:analysis.append('Keep the frontend decision provisional until all required workflows have either produced real outputs or have a documented concrete failure. Do not start EO exposure optimization yet.')
    else:analysis.append('All required outputs are available for review. A main-frontend decision still requires inspecting the leaf/edge figures alongside the numeric diagnostics; no external 3D GT or metric accuracy has been established. EO remains a separate next-stage decision.')
    if by.get('Fast3R',{}).get('status')=='complete' and by.get('VGGT_Omega_Existing',{}).get('status')=='complete':
        fast=by['Fast3R'];vggt=by['VGGT_Omega_Existing']
        analysis += ['', '## Actual Fast3R / existing VGGT-Ω evidence','',
            '| Diagnostic | Existing VGGT-Ω depth reference | Fast3R single global pass |','| --- | --- | --- |']
        for key in ('views','edge_q85_coverage','adjacent_depth_agreement_fraction','adjacent_point_distance_p50_normalized','rotation_step_p90_deg','low_confidence_frames_ratio'):
            analysis.append(f'| {key} | {fmt(vggt.get(key))} | {fmt(fast.get(key))} |')
        fast_consistency=fast.get('adjacent_depth_agreement_fraction');vggt_consistency=vggt.get('adjacent_depth_agreement_fraction')
        fast_distance=fast.get('adjacent_point_distance_p50_normalized');vggt_distance=vggt.get('adjacent_point_distance_p50_normalized')
        if None not in (fast_consistency,vggt_consistency,fast_distance,vggt_distance) and fast_consistency<vggt_consistency and fast_distance>vggt_distance:
            analysis += ['', '**Current answer: Fast3R does not improve on the existing VGGT-Ω reference in these full-sequence consistency diagnostics.** Its lower adjacent depth agreement and larger normalized pointmap discrepancy point in the same direction. This comparison retains the existing VGGT input resolution and the prescribed Fast3R settings; it is not a matched-resolution GT accuracy claim.',
                '', f'Fast3R PnP solver success is {fmt(fast.get("pnp_success_rate"))}, but foreground map/camera reprojection median is {fmt(fast.get("self_reprojection_p50_px"))}px and rotation-step p90 is {fmt(fast.get("rotation_step_p90_deg"))}degrees. Solver success does not establish reliable camera geometry. Keep VGGT-Ω as the reference until another frontend demonstrates a useful geometric gain.']
        else:
            analysis += ['', 'The available diagnostics do not give a unanimous Fast3R-versus-VGGT-Ω verdict. Inspect confidence-gated leaf geometry and trajectory before choosing a main frontend.']
        analysis += ['', 'A very small Fast3R depth-jump ratio must be read alongside map discrepancy, reprojection and confidence: excessive smoothing or collapsed sheets can also be continuous. No claim of superior thin-leaf geometry follows from continuity alone.']
        diagnostic_path=output/'Fast3R_foreground_PnP_diagnostic'/'summary.json'
        if diagnostic_path.exists():
            diagnostic=json.loads(diagnostic_path.read_text())
            analysis += ['', '## Foreground-validity camera control','',
                'The official PnP camera procedure was rerun on CPU with unchanged raw world points and exact foreground confidence, marking only background pixels invalid in a temporary confidence copy. The prescribed primary cameras were retained.',
                '', '| Diagnostic | Primary all-pixel PnP | Foreground-validity control |','| --- | --- | --- |']
            for diagnostic_key,primary_key in [('foreground_reprojection_p50_px','self_reprojection_p50_px'),('rotation_step_p90_deg','rotation_step_p90_deg'),('adjacent_point_distance_p50_normalized','adjacent_point_distance_p50_normalized'),('adjacent_depth_agreement_fraction','adjacent_depth_agreement_fraction')]:
                analysis.append(f'| {diagnostic_key} | {fmt(fast.get(primary_key))} | {fmt(diagnostic.get(diagnostic_key))} |')
            analysis += ['', 'This control diagnoses sensitivity to foreground validity. It does not replace the original experiment. Partial improvement in reprojection does not establish reliable geometry when rotation continuity and adjacent-depth agreement remain poor. Raw pointmaps, and therefore raw side/top geometric banding, are unchanged.']
    if 'AMB3R_Base_OFF' in bundles and 'AMB3R_Base_ON' in bundles:
        off=bundles['AMB3R_Base_OFF'];on=bundles['AMB3R_Base_ON']
        same_names=off.metadata['input_names']==on.metadata['input_names'];same_transforms=(off.path/'pixel_transforms.json').read_bytes()==(on.path/'pixel_transforms.json').read_bytes()
        same_seed=off.metadata.get('metadata',{}).get('seed')==on.metadata.get('metadata',{}).get('seed')
        pair_signature=pair_control_signature(off,on)
        pair_cache=output/'amb3r_off_on_pair_cache.json'
        cached_pair=json.loads(pair_cache.read_text()) if pair_cache.exists() else {}
        reuse_pair=cached_pair.get('signature')==pair_signature and (output/'amb3r_off_on_geometry_controls.json').exists()
        if reuse_pair:
            verification=cached_pair['input_verification']
            same_rgb=verification['model_rgb_pixels_identical'];same_mask=verification['model_mask_pixels_identical']
            print('  Reusing completed OFF/ON identical-pixel controls: raw/image/mask file identities and camera/transform digests unchanged',flush=True)
        else:
            same_rgb=same_names and all(np.array_equal(load_frame(off,name)['rgb'],load_frame(on,name)['rgb']) for name in off.metadata['input_names'])
            same_mask=same_names and all(np.array_equal(load_frame(off,name)['mask'],load_frame(on,name)['mask']) for name in off.metadata['input_names'])
        json_write(output/'amb3r_off_on_input_verification.json',dict(frame_order_identical=same_names,pixel_transforms_identical=same_transforms,model_rgb_pixels_identical=same_rgb,model_mask_pixels_identical=same_mask,seed_identical=same_seed))
        analysis += ['',f'OFF/ON frame order identical: {same_names}; pixel transforms identical: {same_transforms}; RGB pixels identical: {same_rgb}; mask pixels identical: {same_mask}; seed identical: {same_seed}.']
        if same_names:
            analysis += ['', '| Diagnostic | OFF | ON |','| --- | --- | --- |']
            for key in ('foreground_coverage','edge_coverage','edge_q85_coverage','confidence_floor_fg_fraction','depth_neighbor_jump_ratio','adjacent_depth_agreement_fraction','adjacent_point_distance_p50_normalized','pointhead_camera_reprojection_all_foreground_p50_px','depth_unprojection_self_reprojection_p50_px','pointhead_over_unprojection_worldnorm_p50'):
                analysis.append(f'| {key} | {fmt(by["AMB3R_Base_OFF"].get(key))} | {fmt(by["AMB3R_Base_ON"].get(key))} |')
            offrow=by['AMB3R_Base_OFF'];onrow=by['AMB3R_Base_ON']
            offdist=offrow.get('adjacent_point_distance_p50_normalized');ondist=onrow.get('adjacent_point_distance_p50_normalized')
            offagree=offrow.get('adjacent_depth_agreement_fraction');onagree=onrow.get('adjacent_depth_agreement_fraction')
            if same_transforms and same_rgb and same_mask and same_seed and None not in (offdist,ondist,offagree,onagree):
                direction='improves' if ondist<offdist and onagree>offagree else 'does not consistently improve'
                analysis += ['',f'On the identical OFF/ON input set, the backend **{direction} both adjacent diagnostics**. Leaf-edge and depth-continuity figures are still required to decide whether this numerical change benefits thin leaves rather than smoothing or moving them.']
            analysis += ['', 'The higher ON confidence-valid coverage is a real change in the saved confidence output, strongly affected by OFF rounding at1. It is not calibrated confidence or a measured geometric accuracy gain. Large all-foreground point-head/camera disagreement remains after the backend; depth-unprojection self-reprojection is algebraic. Separate prediction gauges and point/depth inconsistency prevent assigning this discrepancy entirely to camera error.',
                '', 'The official depth_metric field retains its learned metre prediction, but transferring its factor to the direct point-head is gauge-unvalidated. Frame-to-frame metric factors and head mismatch limit current measurement claims; no measured length or scan-scale correction was used.']
            if same_transforms and same_rgb and same_mask and same_seed:
                controls=json.loads((output/'amb3r_off_on_geometry_controls.json').read_text()) if reuse_pair else paired_geometry_controls(off,on,output)
                json_write(pair_cache,dict(signature=pair_signature,input_verification=json.loads((output/'amb3r_off_on_input_verification.json').read_text()),semantics='Reuse of an actual completed full-pair evaluation only when source raw/RGB/mask path-size-mtime identities, camera/transform SHA256, input order and seeds match. No new predictions or confidence remapping.'))
                analysis += ['', '## OFF/ON identical-pixel geometry controls', '',
                    'Primary confidence-valid pixels differ between OFF and ON. These two controls use exactly the same selected pixels in both methods: the intersection of baseline-valid geometry, and all mask+finite geometry without a confidence threshold. They separate geometry changes from confidence-selection effects.', '',
                    '| Method | Control | Selected FG | Adjacent depth agreement | Normalized map distance | Projection overlap | World NN |','| --- | --- | --- | --- | --- | --- | --- |']
                for r in controls:analysis.append('| '+' | '.join(fmt(r.get(k)) for k in ('method','control','selected_foreground_fraction','adjacent_depth_agreement_fraction','adjacent_point_distance_p50_normalized','projection_overlap_fraction','adjacent_world_nn_p50_normalized'))+' |')
                analysis += ['', 'World NN is a pose-independent proximity proxy. Occlusions, density and collapsed surfaces can yield a low distance without correct geometry; it is never a sole accuracy or winner criterion.']
    if by.get('Fast3R_FP32',{}).get('status')=='complete' and by.get('Fast3R',{}).get('status')=='complete':
        analysis += ['', '## Full-sequence precision control', '',
            'The official precision string32 disables outer autocast; official internal FlashAttention still uses BF16. Both runs use all208 identical original frames and unchanged weights. The BF16 primary result is preserved.',
            '', '| Diagnostic | BF16 primary | Outer FP32 control |','| --- | --- | --- |']
        for key in ('self_reprojection_p50_px','edge_q85_coverage','adjacent_depth_agreement_fraction','adjacent_point_distance_p50_normalized','rotation_step_p90_deg','total_frontend_seconds'):
            analysis.append(f'| {key} | {fmt(by["Fast3R"].get(key))} | {fmt(by["Fast3R_FP32"].get(key))} |')
        analysis += ['', 'Inspect `visualizations/precision_control` together with this table. Outer precision is a tested control; the result does not isolate the official internal BF16 attention. Neither a similar reprojection result nor numeric continuity alone establishes reliable thin-leaf reconstruction.']
    if any(r.get('gt_reference_frames') for r in summaries):
        analysis += ['', '## Existing five-view manual2D GT', '',
            'Actual manual masks and photos from03-GT are used for source0000,0025,0050,0075,0100. The GT column and model projections share original-view coordinates. The mask also includes the pot and separate blue object; it is not a leaf-only instance annotation. Support recall measures valid RAP foreground geometry at manual-GT pixels. No registered3D scan or real-scale accuracy is asserted.',
            '', '| Method | Manual2D GT support | Manual2D GT edge support |','| --- | --- | --- |']
        for r in summaries:
            if r.get('gt_reference_frames'):analysis.append(f'| {r["method"]} | {fmt(r.get("manual_gt_2d_support_recall"))} | {fmt(r.get("manual_gt_2d_edge_support_recall"))} |')
    # Human/team decisions are preserved when measurements are refreshed.
    (output/'evaluation_analysis.md').write_text('\n'.join(analysis))
    if not (output/'final_analysis.md').exists():
        (output/'final_analysis.md').write_text('\n'.join(analysis))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,default=EXPERIMENT);parser.add_argument('--output',type=Path,default=None);parser.add_argument('--methods',nargs='*',default=METHODS);parser.add_argument('--no-figures',action='store_true');parser.add_argument('--reuse-metrics',action='store_true');parser.add_argument('--figures-only',action='store_true')
    args=parser.parse_args();output=args.output or args.experiment;output.mkdir(parents=True,exist_ok=True);summaries=[];bundles={};norms={};frame_rows={}
    if args.figures_only:args.reuse_metrics=True
    pair_selection_path=args.experiment/'shared_config'/'amb3r_base_pair.json'
    pair_selection=json.loads(pair_selection_path.read_text()) if pair_selection_path.exists() else {}
    selected_paths={'AMB3R_Base_OFF':pair_selection.get('off_bundle_dir'),'AMB3R_Base_ON':pair_selection.get('on_bundle_dir')}
    method_list=list(args.methods)
    if (args.experiment/'Fast3R_FP32'/'bundle.json').exists() and 'Fast3R_FP32' not in method_list:method_list.append('Fast3R_FP32')
    preloaded={}
    registry_path=args.experiment/'run_registry.json'
    registry={r['method']:r for r in json.loads(registry_path.read_text()).get('records',[])} if registry_path.exists() else {}
    for method in method_list:
        candidate=args.experiment/(selected_paths.get(method) or method)
        if (candidate/'bundle.json').exists():preloaded[method]=FrontendBundle(candidate)
    eligible=[b for b in preloaded.values() if len(valid_camera_indices(b))]
    common_valid=set.intersection(*(set(b.metadata['input_names'][int(i)] for i in valid_camera_indices(b)) for b in eligible)) if eligible else set()
    common_anchor=next((n for n in eligible[0].metadata['input_names'] if n in common_valid),None) if eligible else None
    anchor_record={}
    for method,b in preloaded.items():
        valid=valid_camera_indices(b);available={b.metadata['input_names'][int(i)]:int(i) for i in valid}
        b.display_anchor_index=available.get(common_anchor,int(valid[0]) if len(valid) else None)
        anchor_record[method]=b.metadata['input_names'][b.display_anchor_index] if b.display_anchor_index is not None else None
    json_write(output/'display_anchors.json',dict(common_anchor_source_name=common_anchor,methods=anchor_record,policy='earliest source camera finite in every method with any finite cameras; empty methods have no camera anchor'))
    for method in method_list:
        path=args.experiment/(selected_paths.get(method) or method)
        if method not in preloaded:
            status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
            record=registry.get(method,{})
            state=status.get('status',record.get('state','pending'))
            if (path/'bundle.json').exists():state='complete_pending_evaluation'
            summaries.append(dict(method=method,status=state,views=status.get('input_views',status.get('views',record.get('input_views',0))),error=status.get('error'),attempt_path=record.get('attempt_path')));continue
        print(f'evaluating {method}',flush=True)
        bundle=preloaded[method];cached=cached_metrics(bundle) if args.reuse_metrics else None
        if args.figures_only and cached is None:raise RuntimeError(f'{method}: figures-only requires saved metrics with the same display anchor')
        summary,rows,norm=cached if cached is not None else evaluate_bundle(bundle)
        if not args.figures_only:summary=augment_world_nn(bundle,summary)
        summary.update(evaluate_manual_gt(bundle,reuse=args.reuse_metrics))
        summary=enrich_summary(summary,bundle)
        summary=dict(summary,method=method,bundle_directory=str(path.relative_to(args.experiment)))
        summaries.append(summary);bundles[method]=bundle;norms[method]=norm;frame_rows[method]=rows
    if not args.figures_only:reports(output,summaries,bundles,frame_rows)
    if not args.no_figures:
        primary={k:b for k,b in bundles.items() if k!='Fast3R_FP32'}
        figures(primary,{k:norms[k] for k in primary},output/'visualizations')
        if 'Fast3R' in bundles and 'Fast3R_FP32' in bundles:
            precision={k:bundles[k] for k in ('Fast3R','Fast3R_FP32')}
            figures(precision,{k:norms[k] for k in precision},output/'visualizations'/'precision_control')
    print('evaluation complete',flush=True)

if __name__=='__main__':main()
