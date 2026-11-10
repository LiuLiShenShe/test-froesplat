"""P2 conservative multiview geometric cleaning; no color/opacity deletion."""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import json,time
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from plyfile import PlyData,PlyElement
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection';DUAL=OUT.parent/'geometry_rgb_dual_frontend'
dst=OUT/'P2_GGGS_Cleaning';dst.mkdir(parents=True,exist_ok=True)
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
if (dst/'cleaning_complete.json').exists():raise SystemExit(0)
start=time.time();source=DUAL/'E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply'
data=PlyData.read(str(source));v=data['vertex'].data
xyz=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64)
sigma=np.exp(np.column_stack([v[f'scale_{i}'] for i in range(3)]).astype(np.float64))
filter3d=np.asarray(v['filter_3D'],np.float64)[:,None];filtered=np.hypot(sigma,filter3d)
maxsig=filtered.max(1);minsig=filtered.min(1);anisotropy=maxsig/np.maximum(minsig,1e-12)
n=len(v);counts={k:np.zeros(n,np.uint16) for k in ['frustum','visible','fg_support','edge_support','bg_visible','depth_consistent','foreground_inconsistent','occluded']}
scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text());diag=scale['diagonal'];voxel=scale['tsdf_voxel']
index=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text());assert len(index)==208
for row in index:
    c=row['camera'];E=np.asarray(c['world_view_transform'],np.float64).T
    pc=xyz@E[:3,:3].T+E[:3,3];z=pc[:,2]
    uv=pc[:,:2]/np.maximum(z[:,None],1e-12)*[c['Fx'],c['Fy']]+[c['Cx'],c['Cy']]
    inview=(z>0)&(uv[:,0]>=0)&(uv[:,0]<c['width'])&(uv[:,1]>=0)&(uv[:,1]<c['height'])
    ids=np.flatnonzero(inview);counts['frustum'][ids]+=1
    ix=np.clip(np.rint(uv[ids,0]).astype(int),0,c['width']-1);iy=np.clip(np.rint(uv[ids,1]).astype(int),0,c['height']-1)
    with np.load(row['path']) as a:mask=a['foreground_mask'];depth=a['depth_surface'];alpha=a['alpha']
    outside_dist=ndimage.distance_transform_edt(~mask);inside_dist=ndimage.distance_transform_edt(mask)
    d=depth[iy,ix];reliable=np.isfinite(d)&(d>0)&(alpha[iy,ix]>.1)
    tol=np.maximum(np.maximum(.01*np.abs(d),2*minsig[ids]),2*voxel)
    visible=(~reliable)|(z[ids]<=d+tol);counts['visible'][ids[visible]]+=1;counts['occluded'][ids[~visible]]+=1
    pad=np.clip(2*maxsig[ids]*max(c['Fx'],c['Fy'])/np.maximum(z[ids],1e-12),2,12)
    foreground=mask[iy,ix];edge=(outside_dist[iy,ix]<=pad)&(~foreground)
    consistent=reliable&(np.abs(z[ids]-d)<=tol)
    counts['depth_consistent'][ids[consistent]]+=1
    fg=visible&foreground
    counts['fg_support'][ids[fg]]+=1
    counts['edge_support'][ids[visible&edge&consistent]]+=1
    counts['foreground_inconsistent'][ids[foreground&visible&(~consistent)]]+=1
    counts['bg_visible'][ids[visible&(~foreground)&(~edge)]]+=1
    print('P2 support',row['source_id'],flush=True)
# Neighborhood support uses spatial neighbors with foreground-supported evidence.
tree=cKDTree(xyz);dist,neighbor=tree.query(xyz,k=9,workers=8)
supported=(counts['fg_support']>0)|(counts['edge_support']>0)
neighbor_supported=supported[neighbor[:,1:]].sum(1)
nearest_supported=cKDTree(xyz[supported]).query(xyz,workers=8)[0] if supported.any() else np.full(n,np.inf)
bg=counts['bg_visible'];fg=counts['fg_support'];edge=counts['edge_support'];visible=counts['visible']
no_foreground=(fg==0)&(edge==0)
ratio=bg/np.maximum(visible,1)
# Three mutually logged causes, with explicit thin-leaf/edge and spatial protection.
background=no_foreground&(bg>=3)&(ratio>=.8)&((neighbor_supported<=2)|(nearest_supported>4*voxel))
isolated=no_foreground&(bg>=8)&(ratio>=.5)&(nearest_supported>10*voxel)
abnormal=(anisotropy>100)|(maxsig>.05*diag)
spike=abnormal&(bg>=15)&(fg<=1)&(edge==0)&(counts['foreground_inconsistent']>=fg)&(counts['depth_consistent']<2)&(nearest_supported>8*voxel)
remove=background|isolated|spike
# Any consistent leaf-edge evidence stays. Gaussian color/opacity never used.
protect=(edge>0)|((fg>0)&(counts['depth_consistent']>0))|((neighbor_supported>=5)&(nearest_supported<2*voxel))
remove &= ~protect
np.savez_compressed(dst/'cleaning_evidence.npz',**counts,xyz=xyz,max_sigma=maxsig,min_sigma=minsig,anisotropy=anisotropy,neighbor_supported=neighbor_supported,nearest_supported_distance=nearest_supported,background_cause=background,isolated_cause=isolated,spike_cause=spike,protected=protect,removed=remove)
# Original links to actual immutable historical native model; independent filtered files.
original=dst/'original_gaussians.ply'
if not original.exists():original.symlink_to(source.resolve())
for name,sel in [('cleaned_gaussians.ply',~remove),('removed_gaussians.ply',remove)]:
    PlyData([PlyElement.describe(v[sel].copy(),'vertex')],text=False,byte_order='<').write(str(dst/name))
write(dst/'cleaning_complete.json',dict(status='native_cleaned_model_exported_pending_render_and_mesh',source_checkpoint=str(source.resolve()),original_gaussians=n,cleaned_gaussians=int((~remove).sum()),removed_gaussians=int(remove.sum()),removed_fraction=float(remove.mean()),background_cause=int((background&remove).sum()),isolated_cause=int((isolated&remove).sum()),spike_cause=int((spike&remove).sum()),protected_gaussians=int(protect.sum()),views=208,wall_seconds=time.time()-start,opacity_or_color_deletion=False,algorithm='Projected native centers with capped2sigma footprints; depth front/occlusion tolerance max1%depth,2minsigma,2voxels; multiview background evidence, native scale anisotropy and KDtree foreground-neighbor protection',parameters=dict(background_views_min=3,background_visible_ratio=.8,neighborhood_k=8,max_edge_padding_pixels=12,anisotropy_threshold=100,largest_scale_over_diagonal=.05,voxel=voxel,diag=diag)))
print('P2 exported',int(remove.sum()),'/',n,flush=True)

