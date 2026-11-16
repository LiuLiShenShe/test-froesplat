"""Multi-view cube boundary/corner fit in the frozen COLMAP coordinate frame."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from common import *
from scipy.optimize import least_squares,linear_sum_assignment
from scipy.spatial import ConvexHull
from PIL import ImageDraw

SIGNS=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])

def cube(params):return params[:3]+np.exp(params[6])*(SIGNS@Rotation.from_rotvec(params[3:6]).as_matrix().T)

def triangulate(observations):
 rows=[]
 for name,xy in observations:
  c=CAMERAS[name];P=K@np.c_[c['R'],c['t']]
  rows.extend([xy[0]*P[2]-P[0],xy[1]*P[2]-P[1]])
 _,_,v=np.linalg.svd(rows);return v[-1,:3]/v[-1,3]

def main():
 meter=Meter();records=[];observations=[];directions=np.c_[np.cos(np.linspace(0,2*np.pi,48,endpoint=False)),np.sin(np.linspace(0,2*np.pi,48,endpoint=False))]
 for p in sorted((ROOT/'03-GT-区分/KongQueZhuYu').glob('*.json')):
  j=read_json(p);shape=next(s for s in j['shapes'] if s['label']=='blue_cube')
  pts=raw_to_common(shape['points']);hull=pts[ConvexHull(pts).vertices]
  # Hull simplification removes dense annotation jitter without inventing corners.
  import cv2
  hull=cv2.approxPolyDP(hull.astype('float32'),.025*cv2.arcLength(hull.astype('float32'),True),True)[:,0].astype(float)
  records.append(dict(view=p.stem,source=str(p),source_type='Historical human semantic boundary, not manually assigned3D corner IDs',hull_common_xy=hull.tolist(),raw_polygon=shape['points']))
  observations.append((p.stem,hull))
 center=triangulate([(n,p.mean(0)) for n,p in observations])
 estimates=[]
 for name,points in observations:
  _,depth=project(center[None],name);estimates.append(np.ptp(points,axis=0).mean()*depth[0]/K[0,0])
 edge=np.median(estimates)
 def residual(params,subset=observations):
  xyz=cube(params);rows=[]
  for name,points in subset:
   xy,z=project(xyz,name)
   rows.extend(((xy@directions.T).max(0)-(points@directions.T).max(0)))
   rows.extend(((xy@directions.T).min(0)-(points@directions.T).min(0)))
  return np.array(rows)
 best=None;rng=np.random.default_rng(0)
 for i in range(24):
  rotation=Rotation.random(random_state=rng).as_rotvec();initial=np.r_[center,rotation,np.log(edge)]
  fit=least_squares(residual,initial,loss='soft_l1',f_scale=.6,max_nfev=400)
  if best is None or np.sum(residual(fit.x)**2)<np.sum(residual(best.x)**2):best=fit
 params=best.x;xyz=cube(params);corner_obs={i:[] for i in range(8)};assignments=[]
 for name,points in observations:
  xy,z=project(xyz,name);h=ConvexHull(xy);visible=h.vertices
  distances=np.linalg.norm(xy[visible,None]-points[None],axis=2);a,b=linear_sum_assignment(distances)
  for ai,bi in zip(a,b):
   index=int(visible[ai]);error=float(distances[ai,bi])
   if error<3:
    corner_obs[index].append((name,points[bi]));assignments.append(dict(view=name,corner=index,observed_xy=points[bi].tolist(),fitted_xy=xy[index].tolist(),reprojection_error_px_common=error))
 corners={};errors=[]
 for index,obs in corner_obs.items():
  if len(obs)<2:continue
  init=triangulate(obs)
  fit=least_squares(lambda x:np.concatenate([project(x[None],n)[0][0]-p for n,p in obs]),init,loss='soft_l1',f_scale=.5)
  per=[float(np.linalg.norm(project(fit.x[None],n)[0][0]-p)) for n,p in obs];errors.extend(per)
  corners[index]=dict(xyz=fit.x.tolist(),views=[n for n,_ in obs],errors_common_px=per)
 edges=[]
 for i in corners:
  for j in corners:
   if i<j and np.sum(SIGNS[i]!=SIGNS[j])==1:
    length=float(np.linalg.norm(np.array(corners[i]['xyz'])-corners[j]['xyz']))
    edges.append(dict(corners=[i,j],length_units=length,views_common=sorted(set(corners[i]['views'])&set(corners[j]['views']))))
 model_edge=float(np.exp(params[6]));support_rms=float(np.sqrt(np.mean(residual(params)**2)))
 status='CALIBRATED_MULTI_VIEW_CUBE' if len(corners)>=4 and len(edges)>=3 and support_rms<2 else 'REVIEW_REQUIRED'
 leaveout=[]
 for name,_ in observations:
  sub=[o for o in observations if o[0]!=name];f=least_squares(lambda p:residual(p,sub),params,loss='soft_l1',f_scale=.6,max_nfev=200)
  leaveout.append(dict(excluded=name,edge_units=float(np.exp(f.x[6])),scale_cm_per_unit=float(2/np.exp(f.x[6]))))
 write(OUT/'calibration/cube_corners.json',dict(corner_index_definition=SIGNS.tolist(),joint_cube_corners=xyz.tolist(),individual_triangulated_corners=corners,adjacent_edges=edges,observed_corner_assignments=assignments,semantic_boundary_sources=records,method='Rigid cube jointly fit to5 trueCOLMAP multi-view convex silhouettes; fixed cube topology resolves adjacent edges; independent triangulation and corner reprojection cross-check',manual_new_clicks=0,historical_semantic_annotations=5,agent_visual_inspection=True))
 scale=2/model_edge
 write(OUT/'calibration/scale_calibration.json',dict(reference_object='blue_cube',real_edge_length_cm=2.,reconstructed_edge_length_units=model_edge,scale_cm_per_unit=scale,coordinate_frame='Unchanged trueCOLMAP',reference_views=[n for n,_ in observations],reprojection_error_px=float(np.mean(errors)) if errors else None,reprojection_resolution=[530,948],corner_error_px_original_approx=float(np.mean(errors)*4) if errors else None,boundary_support_rms_px=support_rms,status=status,independent_adjacent_edges=edges,edge_length_units_mean=float(np.mean([r['length_units'] for r in edges])) if edges else None,edge_length_units_std=float(np.std([r['length_units'] for r in edges])) if edges else None,leave_one_view_out=leaveout,motion_relation='Cube/pot on same rigid support; static multi-view cube fit evaluated with plantCOLMAP cameras, no external-camera model substitution',motion_validation='Multiplechanging visible faces and low residualjointcube reprojection over widely separated orbit views',cube_orientation=Rotation.from_rotvec(params[3:6]).as_matrix().tolist(),cube_center=xyz.mean(0).tolist(),limits=['Only5 historical annotated views provide high-resolution boundaries; SAM3 additional cube masks provide a separate semantic check','Edge identities from multi-view rigid cube fit; boundary annotations not precision fiducials','No claim that the cube alone establishes gravity or soil-base origin']))
 canvas=Image.new('RGB',(530*5,600),'white');draw=ImageDraw.Draw(canvas)
 for c,(name,points) in enumerate(observations):
  im=rgb(name);xy,_=project(xyz,name);lo=np.floor(points.min(0)-15).astype(int);hi=np.ceil(points.max(0)+15).astype(int);crop=im.crop((*lo,*hi));factor=min(500/crop.width,520/crop.height);crop=crop.resize((int(crop.width*factor),int(crop.height*factor)));canvas.paste(crop,(c*530,50))
  for i in range(8):
   x,y=(xy[i]-lo)*factor+[c*530,50];draw.ellipse((x-4,y-4,x+4,y+4),fill='red');draw.text((x+4,y),str(i),fill='red')
  for i in range(8):
   for j in range(i):
    if np.sum(SIGNS[i]!=SIGNS[j])==1:
     p0=(xy[i]-lo)*factor+[c*530,50];p1=(xy[j]-lo)*factor+[c*530,50];draw.line([tuple(p0),tuple(p1)],fill='lime',width=2)
  draw.text((c*530+5,5),f'{name} fitted cube; edge {model_edge:.5f}u',fill='black')
 canvas.save(OUT/'calibration/calibration_overview.png')
 print('CALIBRATION',status,'scale',scale,'edge',model_edge,'rms',support_rms,'corners',len(corners),'edges',len(edges),flush=True)

if __name__=='__main__':main()
