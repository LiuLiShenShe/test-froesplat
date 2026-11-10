"""Stratify existing area-uniform PCA queries by actual208-view visibility/RAP.

This is a surface proxy, never a semantic leaf mask or scan-distance metric.
It reads existing meshes' raycasts and samples without resampling or changing
any Gaussian/Mesh. Local neighboring sheets still contribute to the PCA fit.
"""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import argparse,csv,fcntl,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage
from queue_io import read_status,write_status

OUT=Path(__file__).resolve().parents[1]
OLD=OUT.parent/'geometry_mesh_final_selection'
DEST=OUT/'comparison/leaf_visible_surface'
RULE='actual208_ray_depth_RAP_upper60_interior3_min2_views_within2_shared_voxels_v1'

def read(path):return list(csv.DictReader(path.open())) if path.is_file() else []
def csvwrite(path,rows):
 if not rows:return
 temporary=path.with_suffix('.csv.tmp')
 with temporary.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 temporary.replace(path)
def values(a):
 finite=np.asarray(a);finite=finite[np.isfinite(finite)]
 return dict(mean=float(finite.mean()),median=float(np.median(finite)),p90=float(np.quantile(finite,.9)),p95=float(np.quantile(finite,.95))) if len(finite) else dict(mean='',median='',p90='',p95='')
def mask_lookup(packed,pixels):
 return ((packed[pixels//8]>>(7-pixels%8))&1).astype(bool)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='+');args=parser.parse_args()
 DEST.mkdir(exist_ok=True)
 lock=(DEST/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.time()
 scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text())
 diagonal=float(scale['diagonal']);voxel=diagonal/512;tolerance=2*voxel
 truth=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text())
 source_ids={r['source_id'] for r in truth}
 assert len(truth)==208 and len(source_ids)==208
 id_bins={sid:min(index//26,7) for index,sid in enumerate(sorted(source_ids))}
 cameras={'standard':{r['source_id']:r['camera'] for r in truth}}
 omega=json.loads((OLD/'surface_inputs/e8_gggs_surface/render_index.json').read_text())
 cameras['omega']={r['source_id']:r['camera'] for r in omega}
 masks={}
 for item in truth:
  with np.load(item['path']) as z:fg=z['foreground_mask'].copy()
  assert hashlib.sha256(fg.tobytes()).hexdigest()==item['mask_sha256']
  y=np.nonzero(fg)[0];cut=int(y.min()+.6*(y.max()-y.min()))
  interior=ndimage.binary_erosion(fg,iterations=3);interior[cut:]=False
  masks[item['source_id']]=(np.packbits(fg.ravel()),np.packbits(interior.ravel()),fg.shape,cut)
 routes=read(OUT/'comparison/ALL_MESH_EXPERIMENTS.csv')
 roughness=read(OUT/'comparison/surface_geometry_metrics.csv')
 allrows=[];completed=[]
 for route in routes:
  method,stage=route['mesh_id'],route['mesh_stage']
  if args.methods and method not in args.methods:continue
  source=[r for r in roughness if r['mesh_id']==method and r['mesh_stage']==stage]
  if not source:continue
  sample_path=Path(source[0]['metric_source']).parent/'surface_samples_and_residuals.npz'
  original=read(Path(route['metric_source']))
  assert len(original)==208 and {int(r['source_id']) for r in original}==source_ids
  key=method+'_'+stage;folder=DEST/key;folder.mkdir(exist_ok=True)
  marker=folder/'complete.json';old=read_status(marker)
  sample_digest=hashlib.sha256(sample_path.read_bytes()).hexdigest()
  metric_digest=hashlib.sha256(Path(route['metric_source']).read_bytes()).hexdigest()
  if old.get('status')=='complete' and old.get('protocol')==RULE and old.get('sample_sha256')==sample_digest and old.get('metrics_sha256')==metric_digest:
   allrows+=read(folder/'metrics.csv');completed.append(old);continue
  with np.load(sample_path) as archive:
   xyz=archive['query_xyz'].copy();face_id=archive['query_face_id'].copy()
   assert np.isfinite(xyz).all() and len(xyz)==int(source[0]['queries'])
   n=len(xyz);foreground=np.zeros(n,np.int16);leaf=np.zeros(n,np.int16);outside=np.zeros(n,np.int16);visible_count=np.zeros(n,np.int16)
   leaf_bins=np.zeros((n,8),bool);view_rows=[]
   camera_set=cameras['omega' if method.startswith('M1_') else 'standard']
   for item in original:
    sid=int(item['source_id']);c=camera_set[sid];packed,interior,shape,cut=masks[sid]
    E=np.asarray(c['world_view_transform'],np.float64).T
    pc=xyz@E[:3,:3].T+E[:3,3]
    assert shape==(c['height'],c['width'])
    with np.errstate(divide='ignore',invalid='ignore'):
     uv=pc[:,:2]/pc[:,2,None]*[c['Fx'],c['Fy']]+[c['Cx'],c['Cy']]
    valid=np.isfinite(uv).all(1)&(pc[:,2]>0)
    ix=np.zeros((n,2),np.int64);ix[valid]=np.rint(uv[valid]).astype(np.int64)
    valid&=(ix[:,0]>=0)&(ix[:,0]<c['width'])&(ix[:,1]>=0)&(ix[:,1]<c['height'])
    index=np.flatnonzero(valid);pixels=ix[index,1]*c['width']+ix[index,0]
    measured=Path(route['metric_source'])
    cache=Path(item['source_native_raycast']) if item.get('source_native_raycast') else measured.parent/'views'/f'{stage}_{sid:04d}.npz'
    with np.load(cache) as z:depth=z['mesh_depth']
    assert depth.shape==shape
    surface=depth[ix[index,1],ix[index,0]]
    visible=np.isfinite(surface)&(surface>0)&(np.abs(surface-pc[index,2])<=tolerance)
    index=index[visible];pixels=pixels[visible]
    fg=mask_lookup(packed,pixels);upper=mask_lookup(interior,pixels)
    visible_count[index]+=1;foreground[index[fg]]+=1;leaf[index[upper]]+=1;outside[index[~fg]]+=1
    leaf_bins[index[upper],id_bins[sid]]=True
    view_rows.append(dict(mesh_id=method,mesh_stage=stage,source_id=sid,in_image_queries=int(valid.sum()),visible_queries=len(index),RAP_foreground_visible=int(fg.sum()),upper_interior_visible=int(upper.sum()),outside_RAP_visible=int((~fg).sum()),source_native_raycast=str(cache.resolve())))
    del depth
   regions=dict(all_surface=np.ones(n,bool),RAP_foreground_supported=foreground>=2,
    upper_leaf_interior_supported=leaf>=2,upper_leaf_supported_across2_id_bins=leaf_bins.sum(1)>=2,
    outside_only_supported=(outside>=2)&(foreground==0),mixed_foreground_background=(foreground>=2)&(outside>=2),
    no_visible_RAP_foreground=foreground==0)
   result=[]
   for row in source:
    tag='r'+row['radius_divisor']+'_'+row['neighborhood']
    rms=archive[tag+'_local_plane_rms'];distance=archive[tag+'_query_plane_distance']
    neighbors=archive[tag+'_neighbor_count']
    for region,selected in regions.items():
     count=int(selected.sum());finite=np.isfinite(rms)&selected
     stats=dict(mesh_id=method,mesh_stage=stage,region=region,radius_divisor=int(row['radius_divisor']),radius=float(row['radius']),neighborhood=row['neighborhood'],total_surface_queries=n,
      selected_queries=count,selected_fraction=count/n,valid_PCA_queries=int(finite.sum()),valid_PCA_selected_fraction=int(finite.sum())/count if count else '',
      effective_support_spacing=float(row['effective_spacing']),support_capped=row['spacing_capped'],
      selected_visible_views_median=float(np.median(visible_count[selected])) if count else '',selected_RAP_views_median=float(np.median(foreground[selected])) if count else '',
      selected_leaf_views_median=float(np.median(leaf[selected])) if count else '',selected_leaf_bins_median=float(np.median(leaf_bins.sum(1)[selected])) if count else '',
      neighbor_count_median=float(np.median(neighbors[selected])) if count else '',source_mesh_path=row['mesh_path'],source_samples=str(sample_path.resolve()),source_metrics=route['metric_source'],protocol=RULE)
     for name,array in [('local_plane_rms_normalized',rms/diagonal),('query_plane_distance_normalized',distance/diagonal)]:
      stats.update({name+'_'+k:v for k,v in values(array[selected]).items()})
     result.append(stats)
   np.savez_compressed(folder/'query_visibility.npz',query_xyz=xyz,query_native_face_id=face_id,visible_view_count=visible_count,RAP_foreground_view_count=foreground,upper_interior_view_count=leaf,outside_RAP_view_count=outside,upper_interior_source_id_bins=leaf_bins,**regions)
  csvwrite(folder/'metrics.csv',result);csvwrite(folder/'visibility_per_view.csv',view_rows)
  record=dict(status='complete',mesh_id=method,mesh_stage=stage,views=208,surface_queries=n,protocol=RULE,
   sample_sha256=sample_digest,metrics_sha256=metric_digest,source_samples=str(sample_path.resolve()),source_mesh=source[0]['mesh_path'],
   selected_upper_leaf_queries=int(regions['upper_leaf_interior_supported'].sum()),upper_leaf_across2_id_bins=int(regions['upper_leaf_supported_across2_id_bins'].sum()),
   depth_visibility_tolerance=tolerance,source_camera_gauge='omega native saved W2C' if method.startswith('M1_') else 'same frozen standard native saved W2C',original_artifacts_modified=False)
  write_status(marker,record);allrows+=result;completed.append(record)
  print(key,'208 visible leaf-supported PCA queries',record['selected_upper_leaf_queries'],'/',n,flush=True)
 csvwrite(DEST/'ALL_REGION_SURFACE_METRICS.csv',allrows)
 write_status(DEST/'PROTOCOL.json',dict(status='complete_actual_available',protocol=RULE,routes=len(completed),requested_methods=args.methods or 'all currently evaluated routes',records=completed,elapsed_seconds=time.time()-started,views_per_route=208,shared_diagonal=diagonal,shared_voxel=voxel,
  interpretation='Existing independent area-uniform surface queries are stratified by their actual mesh ray-depth visibility and frozen RAP projections. Upper60pct bounding-box interior excludes3px rim; neither semantic leaf segmentation nor true scan geometry.',
  caveats='Each method samples its own mesh area, so anatomical samples differ. Local PCA still includes curvature and nearby sheets; support density can be capped. Native face winding matters in same-oriented-normal neighborhoods. Source-ID bins partition208 frames into8 contiguous groups, without asserting calibrated baseline or statistical independence. Visibility tolerates2shared voxels and nearest integer pixels; it can mix surfaces closer than that tolerance. Omega uses its own predicted gauge and W2C with the standardD reference, so fixed-radius lengths across frontends are not a calibrated absolute-length comparison.',
  no_new_inference=True,no_Gaussian_or_Mesh_changes=True,CPU_only=True))
 print('Actual leaf-visible surface metrics',len(completed),'routes',len(allrows),'rows',flush=True)

if __name__=='__main__':main()
