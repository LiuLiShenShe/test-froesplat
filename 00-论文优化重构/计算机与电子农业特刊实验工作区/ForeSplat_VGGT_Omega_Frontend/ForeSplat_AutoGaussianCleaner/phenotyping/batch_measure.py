"""Metric-only traits: all mesh lengths already cm, all areas already cm²."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from batch_common import *
from measurement_geometry import surface_graph,surface_path,one_side_area,centerline_anchors
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull,distance
import csv
sys.path.insert(0,str(FRONT.parent/'阶段二十F_植物表型与尺度恢复/scripts'))
import phenotype_core as core

def csvwrite(path,rows,fields=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=fields or list(dict.fromkeys(k for row in rows for k in row))
 temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with temporary.open('w',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 temporary.replace(path)
def csvread(path):
 with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))

def reference_frame(ctx,mesh,metric):
 data,cameras=ctx.cameras(metric);up=np.mean([-np.array(c['R'])[1] for c in cameras.values()],axis=0);up/=np.linalg.norm(up);cal=read(ctx.out/'calibration/scale_calibration.json')
 if cal.get('cube_orientation'):
  axes=np.array(cal['cube_orientation']);axis=int(np.argmax(abs(up@axes)));vertical=axes[:,axis];vertical*=np.sign(vertical@up);origin='Cube orthogonal axis closest to mean image-up; support/gravity unconfirmed'
 else:
  positions=np.array([c['center'] for c in cameras.values()]);_,_,axes=np.linalg.svd(positions-positions.mean(0),full_matrices=False);vertical=axes[-1];vertical*=np.sign(vertical@up);origin='Camera-orbit plane normal candidate; unconfirmed gravity'
 horizontal=np.array([1.,0,0]);horizontal-=vertical*(horizontal@vertical)
 if np.linalg.norm(horizontal)<.1:horizontal=np.cross(vertical,[0,1,0])
 horizontal/=np.linalg.norm(horizontal);other=np.cross(vertical,horizontal)
 v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);sem=np.load(ctx.out/'plant_only/triangle_semantics.npy');plant=v[np.unique(f[sem==1])];pot=v[np.unique(f[sem==2])]
 if not len(plant):raise ValueError('No supported plant triangles for phenotyping')
 if len(pot):
  level=np.quantile(pot@vertical,.95);center=np.median(pot,axis=0);base=center+vertical*(level-center@vertical);base_source='Projected pot95% height plane at potmedian; stemsoil contact not identified'
 else:
  level=np.quantile(plant@vertical,.01);low=plant[plant@vertical<=np.quantile(plant@vertical,.03)];base=np.median(low,axis=0);base_source='Lowest3% visibleplant median; not confirmed stemsoil contact'
 confirmed=False;user=ctx.out/'annotations/plant_base.json'
 if user.exists():
  annotation=read(user)
  if annotation.get('confirmed') and annotation.get('xyz'):base=np.array(annotation['xyz']);confirmed=True;base_source='Human confirmed visible originalMesh base'
 result=dict(vertical_unit_vector=vertical.tolist(),horizontal_axis_a=horizontal.tolist(),horizontal_axis_b=other.tolist(),base_point=base.tolist(),base_confirmed=confirmed,base_source=base_source,vertical_source=origin,vertical_confirmed=False,unit='cm' if metric else 'model_units',applied_scale_count=1 if metric else 0,trait_multiplier=1,reference_plane=dict(point=base.tolist(),normal=vertical.tolist(),status='Candidate reference plane, not surveyed soil or gravity'))
 write(ctx.out/'calibration/reference_frame.json',result);return result,plant

def main(ctx):
 marker=ctx.out/'phenotype/complete.json'
 if marker.exists():return
 ctx.status('phenotype','running');mesh=ctx.mesh();metric=read(ctx.out/'mesh/complete.json')['metric'];unit='cm' if metric else 'model_units';frame,plant=reference_frame(ctx,mesh,metric);vertical=np.array(frame['vertical_unit_vector']);a=np.array(frame['horizontal_axis_a']);b=np.array(frame['horizontal_axis_b']);base=np.array(frame['base_point']);height=float(np.ptp(plant@vertical)) if getattr(ctx,'use_visible_height',False) else core.height_along_axis(plant,base,vertical);span_a,span_b,_=core.canopy_spans(plant,vertical,a,b);xy=plant@np.stack([a,b]).T
 try:hull=xy[ConvexHull(xy).vertices];maximum=float(distance.pdist(hull).max())
 except Exception:maximum=max(span_a,span_b)
 catalog=read(ctx.out/'leaf_instances/leaf_instances.json');vsource=np.asarray(mesh.vertices);fsource=np.asarray(mesh.triangles);records=[];editpath=ctx.out/'annotations/user_edits.json';edits=read(editpath).get('records',{}) if editpath.exists() else {}
 for inst in catalog['instances']:
  leaf=inst['leaf_id'];folder=ctx.out/'leaf_instances'/leaf
  with np.load(io.BytesIO(ctx.meter.read(folder/'source_indices.npz'))) as d:fi=d['original_face_ids'];vi=d['original_vertex_ids']
  sub,_=subset(mesh,fi);v=np.asarray(sub.vertices);f=np.asarray(sub.triangles);camera=np.array(ctx.cameras(metric)[1][inst['reference_view']]['center']);area,total_area,side,normal=one_side_area(v,f,camera);sf=f[side]
  if not len(sf):continue
  graph=surface_graph(v,sf);nc,components=connected_components(graph,directed=False);active=np.unique(sf);ll,counts=np.unique(components[active],return_counts=True);component=int(ll[counts.argmax()]);selected=np.flatnonzero(components==component);local=np.full(len(v),-1,int);local[selected]=np.arange(len(selected));part=sf[np.all(np.isin(sf,selected),axis=1)];pv=v[selected];pf=local[part];anchors,cross,axes=centerline_anchors(pv,pf)
  if len(anchors)>1 and np.linalg.norm(pv[anchors[0]]-base)>np.linalg.norm(pv[anchors[-1]]-base):anchors=anchors[::-1]
  confirmed_records=[r for r in edits.values() if r.get('leaf_id')==leaf and r.get('human_confirmed')];endpoint_points={};endpoint_sources={}
  for kind in ['base','tip']:
   points=[(r['view'],p) for r in confirmed_records for p in r.get('landmarks_3d',{}).get(kind,[]) if p.get('xyz_units') is not None and p.get('original_face_id') in set(fi.tolist()) and p.get('unit')==unit]
   if points:endpoint_points[kind]=np.median([p['xyz_units'] for _,p in points],axis=0);endpoint_sources[kind]=[view for view,_ in points]
  anatomical_confirmed=all(kind in endpoint_points for kind in ['base','tip'])
  if anatomical_confirmed:anchors=np.array([int(np.argmin(np.linalg.norm(pv-endpoint_points[kind],axis=1))) for kind in ['base','tip']])
  path,length=surface_path(pv,surface_graph(pv,pf),anchors);straight=float(np.linalg.norm(pv[anchors[0]]-pv[anchors[-1]])) if len(anchors)>1 else None;valid=length is not None and straight is not None and .99*straight<=length<=1.8*straight;widths=[];widthpaths=[]
  g=surface_graph(pv,pf)
  for left,right in cross:
   ip,ww=surface_path(pv,g,[left,right]);ss=float(np.linalg.norm(pv[left]-pv[right]))
   if ww is not None and ss>0 and ww<=ss*1.8:widths.append(ww);widthpaths.append(dict(length=ww,source_vertex_ids=vi[selected[ip]].tolist()))
  width=max(widths) if widths else None;whole_surface=any(r.get('whole_leaf_surface_complete_confirmed') is True for r in confirmed_records);whole_valid=bool(metric and valid and anatomical_confirmed and whole_surface and len(selected)/len(v)>=.95 and width is not None);status='HUMAN_CONFIRMED_WHOLE_OBSERVED_SINGLE_SIDE' if whole_valid else 'PARTIAL_OBSERVED_SURFACE_CONFIRMED_ENDPOINTS' if anatomical_confirmed and valid else 'PARTIAL_OBSERVED_SURFACE_AUTO_LANDMARKS' if valid else 'LENGTH_PATH_UNRELIABLE_PARTIAL_SURFACE';provenance='Explicit human confirmed endpoints on this original leaf surface' if anatomical_confirmed else 'Candidate endpoint nearer estimated plantbase; not confirmed anatomical leafbase'
  landmarks=dict(leaf_id=leaf,unit=unit,human_confirmed=anatomical_confirmed,endpoint_reference_views=endpoint_sources,whole_leaf_surface_complete_confirmed=whole_surface,base_xyz=pv[anchors[0]].tolist() if len(anchors) else None,tip_xyz=pv[anchors[-1]].tolist() if len(anchors) else None,centerline_source_vertex_ids=vi[selected[path]].tolist() if path is not None else [],centerline_xyz=pv[path].tolist() if path is not None else [],curved_path_length=length,path_valid=valid,width_paths=widthpaths,one_side_source_face_ids=fi[side].tolist(),one_side_normal=normal.tolist(),area_policy='Greater observed area of two PCA-normal partitions; opposite orientation excluded. Folded/inconsistent/overlapping geometry remains provisional.',measurement_component_vertex_fraction=len(selected)/len(v),original_geometry_preserved=True,anatomical_endpoints_provenance=provenance)
  edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);edges.sort(1);unique,count=np.unique(edges,axis=0,return_counts=True);npz(folder/'boundary.npz',ctx.meter,original_vertex_edges=vi[unique[count==1]]);write(folder/'landmarks.json',landmarks)
  row=dict(plant_id=ctx.plant,group=ctx.record['group'],species_label=ctx.plant,leaf_id=leaf,leaf_length_cm=length if metric and valid else '',leaf_width_cm=width if metric and width is not None else '',leaf_area_cm2=area if metric else '',length_model_units=length if not metric and valid else '',width_model_units=width if not metric and width is not None else '',area_model_units2=area if not metric else '',leaf_surface_coverage='',leaf_surface_coverage_definition='Anatomical surface completeness unmeasured; not inferred from2D visibility',leaf_measurement_status=status if metric else 'UNCALIBRATED_'+status,metric_calibrated=metric,formal_whole_leaf_valid=whole_valid,anatomical_landmarks_confirmed=anatomical_confirmed,straight_leaf_length_cm=straight if metric and straight is not None else '',measurement_component_vertex_fraction=len(selected)/len(v),observed_components=inst['components'],unit=unit,trait_multiplier=1);records.append(row);print('PHENOTYPE',ctx.plant,leaf,row['leaf_length_cm'],row['leaf_area_cm2'],flush=True)
 plantrow=dict(plant_id=ctx.plant,group=ctx.record['group'],plant_height_cm=height if metric else '',canopy_span_a_cm=span_a if metric else '',canopy_span_b_cm=span_b if metric else '',canopy_width_cm=maximum if metric else '',plant_height_model_units=height if not metric else '',leaf_count=len(records),leaf_count_definition='Stable recognized visible instances; not certified totalbiological leafcount',total_valid_leaf_area_cm2='',observed_selected_leaf_area_cm2=sum(r['leaf_area_cm2'] for r in records) if metric else '',scale_cm_per_unit=read(ctx.out/'calibration/scale_calibration.json').get('scale_cm_per_unit'),metric_calibrated=metric,valid_whole_leaf_count=sum(r['formal_whole_leaf_valid'] for r in records),measurement_status='PROVISIONAL_BASE_VERTICAL_AND_SURFACE_REVIEW' if metric else 'UNCALIBRATED_NO_CM_VALUES',base_confirmed=frame['base_confirmed'],vertical_confirmed=False,trait_multiplier=1)
 if getattr(ctx,'use_visible_height',False):plantrow.update(height_definition='Visible plant mesh extent along estimated vertical: max(z)-min(z)',measurement_status='VISIBLE_EXTENT_ALONG_ESTIMATED_VERTICAL' if metric else 'UNCALIBRATED_NO_CM_VALUES')
 leaf_fields=['plant_id','group','species_label','leaf_id','leaf_length_cm','leaf_width_cm','leaf_area_cm2','leaf_measurement_status','metric_calibrated','formal_whole_leaf_valid'];csvwrite(ctx.out/'phenotype/leaf_measurements.csv',records,fields=None if records else leaf_fields);csvwrite(ctx.out/'phenotype/plant_measurements.csv',[plantrow])
 manual=ctx.out/'phenotype/manual_reference.csv'
 if not manual.exists():csvwrite(manual,[dict(plant_id=ctx.plant,leaf_id=r['leaf_id'],leaf_length_cm='',leaf_width_cm='',leaf_area_cm2='',plant_height_cm='',canopy_span_a_cm='',canopy_span_b_cm='',canopy_width_cm='',capture_session='',measured_at='',measurement_tool='',operator='',pair_confirmed=False,notes='Confirm same plant,leafID and capturetime; centerline length,maxwidth,single-side area') for r in records] or [dict(plant_id=ctx.plant,leaf_id='',leaf_length_cm='',leaf_width_cm='',leaf_area_cm2='',plant_height_cm='',canopy_span_a_cm='',canopy_span_b_cm='',canopy_width_cm='',capture_session='',measured_at='',measurement_tool='',operator='',pair_confirmed=False,notes='No stableleaf; planttraits only')])
 write(marker,dict(status='complete',metric=metric,leaf_count=len(records),whole_leaf_valid=sum(r['formal_whole_leaf_valid'] for r in records),length_candidates=sum(bool(r['leaf_length_cm']) for r in records),unit=unit,trait_multiplier=1,manual_truth='No inferred historical pairs; explicit samecapture confirmation required',io=ctx.meter.stats()));ctx.status('phenotype','complete',metric=metric,leaf_count=len(records),whole_leaf_valid=sum(r['formal_whole_leaf_valid'] for r in records))

if __name__=='__main__':main(Context(sys.argv[1]))
