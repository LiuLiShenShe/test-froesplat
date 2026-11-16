"""Actual mesh-only provisional phenotypes, with explicit scale/landmark limits."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from common import *
import csv
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from measurement_geometry import *
sys.path.insert(0,str(ROOT/'00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十F_植物表型与尺度恢复/scripts'))
import phenotype_core as core

def csvwrite(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)

def main():
 import open3d as o3d
 meter=Meter();source=read_mesh(meter);vertices=np.asarray(source.vertices);faces=np.asarray(source.triangles)
 calibration=read_json(OUT/'calibration/scale_calibration.json');assert calibration['status']=='CALIBRATED_MULTI_VIEW_CUBE';scale=calibration['scale_cm_per_unit'];frame=read_json(OUT/'calibration/reference_frame.json');vertical=np.array(frame['vertical_unit_vector']);axis_a=np.array(frame['horizontal_axis_a']);axis_b=np.array(frame['horizontal_axis_b'])
 sem=np.load(io.BytesIO(meter.read(OUT/'plant_only/triangle_semantics.npy')));plant=vertices[np.unique(faces[sem==1])]
 # Agent-proposed visible stem/soil contact in original0000 RGB. It is kept
 # distinct from a human-confirmed botanical landmark.
 name='0000';pixel=np.array([267.,452.])
 with np.load(io.BytesIO(meter.read(OUT/'plant_only/projection_cache/0000.npz'))) as d:ids=d['triangle_id'];uv=d['barycentric_uv']
 y,x=np.rint(pixel).astype(int)[::-1];face=int(ids[y,x])
 if face<0:raise RuntimeError('Visiblebase proposal does not intersect sourceMesh')
 bary=np.r_[1-uv[y,x].sum(),uv[y,x]].astype(float);base=(vertices[faces[face]]*bary[:,None]).sum(0)
 # Existing annotation corrections, when present, take precedence.
 userfile=OUT/'annotations/plant_base.json';base_status='AGENT_PROPOSED_VISIBLE_STEM_SOIL_CONTACT_REQUIRES_HUMAN_CONFIRMATION'
 if userfile.exists():
  user=read_json(userfile)
  if user.get('confirmed') and user.get('xyz_units'):
   base=np.array(user['xyz_units']);base_status='HUMAN_CONFIRMED';name=user.get('view',name);pixel=np.array(user.get('pixel_xy',pixel));face=user.get('original_face_id',face)
 transform=np.stack([axis_a,axis_b,vertical]);frame.update(base_point=base.tolist(),base_point_cm=[0,0,0],base_status=base_status,base_view=name,base_pixel_xy=pixel.tolist(),base_original_face_id=face,base_landmark_author='Agent visual inspection of actualRGB, not historicalhumanmeasurements',scale_cm_per_unit=scale,measurement_rotation_rows=transform.tolist(),reference_plane_point_cm=((np.array(frame['reference_plane']['point_units'])-base)@transform.T*scale).tolist())
 write(OUT/'calibration/reference_frame.json',frame)
 h=core.metric_length(core.height_along_axis(plant,base,vertical),scale,'blue_cube20mm multi-view cube fit')
 span_a,span_b,_=core.canopy_spans(plant,vertical,axis_a,axis_b)
 # Maximumhorizontal Feret diameter is distinct from max of two fixed axes.
 xy=plant@np.stack([axis_a,axis_b]).T
 from scipy.spatial import ConvexHull,distance
 hull=xy[ConvexHull(xy).vertices];maximum=float(distance.pdist(hull).max())
 leaves=read_json(OUT/'leaf_instances/leaf_instances.json')['instances'];rows=[];visualizations=[]
 edits=read_json(OUT/'annotations/user_edits.json')['records'] if (OUT/'annotations/user_edits.json').exists() else {}
 for record in leaves:
  leaf_id=record['leaf_id'];folder=OUT/'leaf_instances'/leaf_id
  with np.load(io.BytesIO(meter.read(folder/'source_indices.npz'))) as d:source_faces=d['original_face_ids'];source_vertices=d['original_vertex_ids']
  mesh,vertex_indices=subset_mesh(source,source_faces);v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles)
  complete_views=[]
  for candidate in record['coverage_by_view']:
   xy,z=project(v,candidate['view']);inframe=(z>0)&(xy[:,0]>3)&(xy[:,0]<527)&(xy[:,1]>3)&(xy[:,1]<945)
   if inframe.mean()>.995:complete_views.append(candidate)
  best_view=max(complete_views or record['coverage_by_view'],key=lambda r:r['mask_pixels'])['view'];camera=CAMERAS[best_view]['c2w'][:3,3]
  observed_area,total_area,side,normal=one_side_area(v,f,camera);side_faces=f[side]
  # Work on one actual oriented surface; otherwise graph paths can cross a
  # reconstructed leaf thickness and falsely shorten the curved-leaf length.
  graph=surface_graph(v,side_faces);ncomp,components=connected_components(graph,directed=False)
  anchors,pairs,axes=centerline_anchors(v,side_faces)
  # Candidate central path must stay on one connected observed patch.
  active=np.unique(side_faces);labels,counts=np.unique(components[active],return_counts=True);component=int(labels[np.argmax(counts)])
  user_endpoints=None;landmark_review='Automatic landmarks; no human confirmation'
  for edit in edits.values():
   lifted=edit.get('landmarks_3d',{})
   if edit.get('leaf_id')!=leaf_id or not edit.get('human_confirmed') or not lifted.get('base') or not lifted.get('tip'):continue
   ends=[lifted[k][0] for k in ['base','tip']]
   if not all(e.get('xyz_units') and e.get('original_face_id') in source_faces for e in ends):
    landmark_review='Human endpoints hit an occluding or differently assigned surface; correction retained, not used';continue
   nearest=[int(active[np.argmin(np.linalg.norm(v[active]-e['xyz_units'],axis=1))]) for e in ends]
   if components[nearest[0]]!=components[nearest[1]]:
    landmark_review='Human endpoints lie on disconnected observed patches; full leaf length unavailable';continue
   user_endpoints=nearest;component=int(components[nearest[0]]);landmark_review='Human-confirmed endpoints snapped to source vertices on one connected observed side';break
  usable=np.flatnonzero(components==component);local=np.full(len(v),-1,int);local[usable]=np.arange(len(usable))
  allowed=np.all(np.isin(side_faces,usable),axis=1);part=side_faces[allowed]
  part_local=local[part];sub_v=v[usable];subgraph=surface_graph(sub_v,part_local)
  subanchors,width_pairs,axes=centerline_anchors(sub_v,part_local)
  # Base is the endpoint nearer the actual plantbase; preserve both candidate
  # endpoint coordinates and whether they have been human-confirmed.
  origin=v[user_endpoints[0]] if user_endpoints is not None else base
  if len(subanchors)>1 and np.linalg.norm(sub_v[subanchors[0]]-origin)>np.linalg.norm(sub_v[subanchors[-1]]-origin):subanchors=subanchors[::-1]
  if user_endpoints is not None:subanchors=np.r_[local[user_endpoints[0]],subanchors[1:-1],local[user_endpoints[1]]]
  path_ids,length=surface_path(sub_v,subgraph,subanchors);widths=[];width_results=[]
  for left,right in width_pairs:
   ids_w,value=surface_path(sub_v,subgraph,[left,right]);straight=float(np.linalg.norm(sub_v[left]-sub_v[right]))
   if value is not None and value<=straight*1.8:
    widths.append(value);width_results.append(dict(left_vertex=int(usable[left]),right_vertex=int(usable[right]),length_units=value,path_source_vertices=source_vertices[usable[ids_w]].tolist()))
  width=max(widths) if widths else None
  straight=core.leaf_length(leaf_id,sub_v[subanchors[0]],sub_v[subanchors[-1]]) if len(subanchors)>1 else None
  simple_width=core.leaf_width(leaf_id,v,axes[1])
  valid_path=length is not None and straight is not None and length<=straight*1.8 and length>=straight*.99
  largest_fraction=float(len(usable)/max(len(v),1))
  status='PROVISIONAL_OBSERVED_SURFACE_AUTO_LANDMARKS'
  if largest_fraction<.8 or record['components']>5:status='PARTIAL_FRAGMENTED_SURFACE_AUTO_LANDMARKS'
  if not valid_path:status='LENGTH_PATH_UNRELIABLE_PARTIAL_SURFACE'
  landmarks=dict(leaf_id=leaf_id,base_xyz_units=sub_v[subanchors[0]].tolist() if len(subanchors) else None,tip_xyz_units=sub_v[subanchors[-1]].tolist() if len(subanchors) else None,landmark_author='Automatic3D principal-axis slab centreline; endpoint nearer plantbase proposed as leafbase',human_confirmed=False,centerline_method='Actual one-side mesh graph through25 surface-centre slab anchors; Dijkstra segment distances, no Euclidean replacement',centerline_xyz_units=sub_v[path_ids].tolist() if path_ids is not None else [],centerline_source_vertex_ids=source_vertices[usable[path_ids]].tolist() if path_ids is not None else [],width_cross_sections=width_results,boundary_definition='Mesh boundary of assigned observed single-side triangles; includes holes and label seams',one_side_normal=normal.tolist(),one_side_source_face_ids=source_faces[side].tolist(),area_policy='Sum actual triangles whose oriented normal faces the reference side of leafPCA plane; opposing side excluded, no divide-by2 assumption',measurement_component_vertex_fraction=largest_fraction,reference_view=best_view,curved_path_length_units=length,path_valid=valid_path,source_surface_incomplete=True)
  landmarks.update(human_confirmed=user_endpoints is not None,landmark_review=landmark_review,area_policy='Partition actual triangle normals by leafPCA reference plane and retain the side with greater observed area; opposite side excluded, no divide-by2 assumption. Folds/inconsistent normals remain a limitation.')
  edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);edges.sort(1);uniq,count=np.unique(edges,axis=0,return_counts=True);boundary=uniq[count==1];npz(folder/'leaf_boundary.npz',meter,source_vertex_edges=source_vertices[boundary])
  landmarks['boundary_edge_count']=len(boundary);write(folder/'landmarks.json',landmarks)
  row=dict(plant_id='KongQueZhuYu',species='孔雀竹芋',leaf_id=leaf_id,leaf_length_cm=float(length*scale) if valid_path else '',leaf_width_cm=float(width*scale) if width is not None else '',leaf_area_cm2=float(observed_area*scale**2),leaf_surface_coverage=record['leaf_surface_coverage'],leaf_measurement_status=status,straight_leaf_length_cm=float(straight*scale) if straight is not None else '',pca_width_cm=float(simple_width*scale),total_observed_two_side_area_cm2=float(total_area*scale**2),measurement_component_vertex_fraction=largest_fraction,scale_cm_per_unit=scale,formal_whole_leaf_valid=False,area_definition='Observed single-side surface only, partial; not complete leaf area',landmarks_human_confirmed=False)
  row['landmarks_human_confirmed']=user_endpoints is not None
  rows.append(row);visualizations.append(dict(leaf_id=leaf_id,base=landmarks['base_xyz_units'],tip=landmarks['tip_xyz_units'],path=landmarks['centerline_xyz_units'],side_normal=normal.tolist()))
  print('TRAIT',leaf_id,row['leaf_length_cm'],row['leaf_width_cm'],row['leaf_area_cm2'],status,flush=True)
 csvwrite(OUT/'phenotype/leaf_measurements.csv',rows)
 plantrow=dict(plant_id='KongQueZhuYu',plant_height_cm=h,canopy_span_a_cm=span_a*scale,canopy_span_b_cm=span_b*scale,canopy_width_cm=maximum*scale,leaf_count=12,leaf_count_definition='12selected representative instances; not all leaves of plant',total_valid_leaf_area_cm2='',total_observed_selected_leaf_area_cm2=sum(r['leaf_area_cm2'] for r in rows),scale_cm_per_unit=scale,measurement_status='PROVISIONAL_BASE_AND_VERTICAL_REVIEW_REQUIRED',base_status=base_status,valid_whole_leaf_count=0)
 csvwrite(OUT/'phenotype/plant_measurements.csv',[plantrow]);write(OUT/'phenotype/measurement_geometry.json',dict(plant=plantrow,leaf_paths=visualizations,io=meter.stats(),height_definition='max plant vertex along vertical relative visible stemsoil base',canopy_definition='Fixed orthonormal axes + maximum convexhull Feret diameter',area_limit='No complete leaf accuracy or totalplant leafarea claim without complete surfaces and confirmed landmarks'))
 # Matched physical data cannot be recovered from workbook row order.
 historical=ROOT/'00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十F_植物表型与尺度恢复/HISTORICAL_PHENOTYPE_MATCHING.csv'
 write(OUT/'phenotype/manual_reference_audit.json',dict(status='NO_CONFIRMED_SAME_SPECIMEN_SAME_TIME_LEAF_PAIRS',source=str(historical),reason='Workbooks contain species-level rows and ordinal leaf slots, without same-specimen time or stableLeafID correspondence. Historicalvirtual values not used.',calibrating_height_from_manual=False))
 manual=OUT/'phenotype/manual_reference.csv'
 if not manual.exists():
  template=[dict(plant_id='KongQueZhuYu',leaf_id=r['leaf_id'],leaf_length_cm='',leaf_width_cm='',leaf_area_cm2='',plant_height_cm='',canopy_span_a_cm='',canopy_span_b_cm='',capture_session='',measured_at='',measurement_tool='',operator='',pair_confirmed=False,notes='Match illustratedLeafID; midrib length, maximumwidth, scanner or area-meter single-side area') for r in rows]
  csvwrite(manual,template)
 csvwrite(OUT/'phenotype/phenotype_errors.csv',[dict(trait=t,n_pairs=0,MAE='',RMSE='',Bias='',relative_error='',R2='',status='NO_REAL_PAIRED_MANUAL_REFERENCE') for t in ['leaf_length_cm','leaf_width_cm','leaf_area_cm2','plant_height_cm','canopy_span_a_cm','canopy_span_b_cm']])

if __name__=='__main__':main()
