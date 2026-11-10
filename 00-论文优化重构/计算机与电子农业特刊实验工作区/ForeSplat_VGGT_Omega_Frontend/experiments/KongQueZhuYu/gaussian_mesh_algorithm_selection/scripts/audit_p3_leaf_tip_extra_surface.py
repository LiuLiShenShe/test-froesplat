"""Locate actual P3 extra projected geometry and its native mesh components."""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy import ndimage
from PIL import Image
from queue_io import write_status

OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection'
DEST=OUT/'comparison/P3_leaf_tip_spatial'

def main():
 DEST.mkdir(exist_ok=True);started=time.time();sid=157
 route=next(r for r in csv.DictReader((OUT/'comparison/ALL_MESH_EXPERIMENTS.csv').open()) if r['mesh_id']=='P3_shared' and r['mesh_stage']=='post')
 mesh_path=Path(route['post_mesh_path'])
 item=next(r for r in json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text()) if r['source_id']==sid)
 bounds=next(r['ROIs']['leaf_edge'] for r in json.loads((OUT/'comparison/visualizations/mesh_comparison/PROTOCOL.json').read_text())['records'] if r['source_id']==sid)
 x1,y1,x2,y2=bounds
 metric=next(r for r in csv.DictReader(Path(route['metric_source']).open()) if int(r['source_id'])==sid)
 cache=Path(metric['source_native_raycast']) if metric.get('source_native_raycast') else Path(route['metric_source']).parent/'views'/f'post_{sid:04d}.npz'
 with np.load(cache) as z:depth=z['mesh_depth'].copy();pred=z['mesh_silhouette'].copy()
 with np.load(item['path']) as z:fg=z['foreground_mask'].copy();gt=z['gt_rgb'].copy()
 raw_source=next(r for r in csv.DictReader((OUT.parent/'geometry_rgb_dual_frontend/shared_manifest.csv').open()) if int(r['source_index'])==sid)
 raw_path=Path(raw_source['raw_rgb_path']).resolve(strict=True)
 with Image.open(raw_path) as im:
  assert im.size==(2160,3840)
  raw=np.asarray(im.convert('RGB').resize((1080,1920)),np.float32)/255
 Image.fromarray(np.rint(raw[y1:y2,x1:x2]*255).astype(np.uint8)).save(DEST/'source157_original_RGB_context.png')
 assert hashlib.sha256(fg.tobytes()).hexdigest()==item['mask_sha256']
 distance=ndimage.distance_transform_edt(~fg)
 extra=pred[y1:y2,x1:x2]&(~fg[y1:y2,x1:x2])&(distance[y1:y2,x1:x2]>3)
 labels,count=ndimage.label(extra);sizes=np.bincount(labels.ravel())
 # Exclude cropped continuations along leaf/background boundaries. This
 # isolates the visually observed separate fragment within the fixed ROI.
 borders=np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])
 eligible=[i for i in range(1,count+1) if sizes[i]>=9 and i not in borders]
 assert eligible,'No enclosed extra-surface candidate in the actually inspected fixed leaf-tip ROI'
 selected=max(eligible,key=lambda i:sizes[i]);yy,xx=np.nonzero(labels==selected);yy+=y1;xx+=x1
 mesh=o3d.io.read_triangle_mesh(str(mesh_path));v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles)
 clusters,component_faces,component_areas=mesh.cluster_connected_triangles()
 clusters=np.asarray(clusters);component_faces=np.asarray(component_faces);component_areas=np.asarray(component_areas)
 scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
 c=item['camera'];c2w=np.linalg.inv(np.asarray(c['world_view_transform']).T)
 direction=np.column_stack([(xx-c['Cx'])/c['Fx'],(yy-c['Cy'])/c['Fy'],np.ones(len(xx))])@c2w[:3,:3].T
 rays=np.column_stack([np.broadcast_to(c2w[:3,3].astype(np.float32),direction.shape),direction]).astype(np.float32)
 hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8);z=hit['t_hit'].numpy();triangle=hit['primitive_ids'].numpy()
 assert np.isfinite(z).all() and (z>0).all() and (triangle<len(f)).all()
 error=np.abs(z-depth[yy,xx]);assert error.max()<1e-5,('Actual cached depth versus same corrected-camera raycast',float(error.max()))
 world=rays[:,:3]+z[:,None]*rays[:,3:]
 records=[]
 for i in range(len(xx)):
  comp=int(clusters[triangle[i]])
  records.append(dict(source_id=sid,pixel_u=int(xx[i]),pixel_v=int(yy[i]),depth_camera_z=float(z[i]),world_x=float(world[i,0]),world_y=float(world[i,1]),world_z=float(world[i,2]),native_triangle_id=int(triangle[i]),native_mesh_component=comp,component_faces=int(component_faces[comp]),GT_RAP_foreground=bool(fg[yy[i],xx[i]]),distance_from_RAP_px=float(distance[yy[i],xx[i]]),training_GT_luminance=float(gt[yy[i],xx[i]]@[.2126,.7152,.0722]),original_RGB_luminance=float(raw[yy[i],xx[i]]@[.2126,.7152,.0722])))
 with (DEST/'source157_extra_surface_world_triangles.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 comps=np.unique(clusters[triangle]);largest=int(component_faces.argmax())
 component_records=[]
 for comp in comps:
  face_ids=np.flatnonzero(clusters==comp);ids=np.unique(f[face_ids].ravel());points=v[ids]
  component_records.append(dict(native_component_id=int(comp),faces=int(component_faces[comp]),vertices=len(ids),area_predicted_gauge_squared=float(component_areas[comp]),bbox_min=points.min(0).tolist(),bbox_max=points.max(0).tolist(),is_largest_mesh_component=int(comp)==largest,pixels_in_selected_candidate=int((clusters[triangle]==comp).sum())))
 result=dict(status='complete',source_id=sid,bounds_native=bounds,candidate_extra_pixels=len(xx),actual_source_mesh=str(mesh_path.resolve()),source_cache=str(cache.resolve()),
  native_mesh_vertices=len(v),native_mesh_faces=len(f),native_mesh_components=len(component_faces),component_records=component_records,
  candidate_components_all_disconnected_from_largest=all(int(comp)!=largest for comp in comps),GT_foreground_overlap_pixels=int(fg[yy,xx].sum()),training_GT_luminance_median=float(np.median([r['training_GT_luminance'] for r in records])),original_RGB_luminance_median=float(np.median([r['original_RGB_luminance'] for r in records])),
  original_RGB_source_path=str(raw_path),original_RGB_source_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),original_RGB_context_crop=str((DEST/'source157_original_RGB_context.png').resolve()),
  exact_camera_cached_depth_max_difference=float(error.max()),world_triangle_csv=str((DEST/'source157_extra_surface_world_triangles.csv').resolve()),elapsed_seconds=time.time()-started,
  selection='Largest >=9px extra-geometry component fully inside the previously frozen192px leaf-edge ROI, >3px from original RAP foreground. Selected after direct inspection; diagnostic sample, not an unbiased all-view frequency.',
  interpretation='Actual mesh-connected-component and world-coordinate evidence. Outside RAP is a mask consistency proxy; absent scan truth and potentially imperfect masks do not certify all components erroneous. Mesh unchanged; a floating fragment in this inspected view is not evidence that all GS2Mesh surfaces float.')
 write_status(DEST/'complete.json',result)
 print('Actual P3 leaf-tip extra surface',len(xx),'pixels; connected components',[(r['native_component_id'],r['faces'],r['is_largest_mesh_component']) for r in component_records],flush=True)

if __name__=='__main__':main()
