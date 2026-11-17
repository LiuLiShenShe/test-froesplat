"""Generic multi-view rigid cube fitting, derived from calibrate_cube.py."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from batch_common import *
from scipy.optimize import least_squares,linear_sum_assignment
from scipy.spatial import ConvexHull
from PIL import ImageDraw
SIGNS=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])

def fit(ctx,observations,assisted):
 data,cameras=ctx.cameras();K=np.array(data['K']);directions=np.c_[np.cos(np.linspace(0,2*np.pi,32,endpoint=False)),np.sin(np.linspace(0,2*np.pi,32,endpoint=False))]
 def project(xyz,name):
  c=cameras[name];camera=xyz@np.array(c['R']).T+c['tvec'];return camera[:,:2]/camera[:,2,None]*np.diag(K)[:2]+K[:2,2],camera[:,2]
 def triangulate(obs):
  rows=[]
  for name,xy in obs:
   c=cameras[name];P=K@np.c_[c['R'],c['tvec']];rows.extend([xy[0]*P[2]-P[0],xy[1]*P[2]-P[1]])
  _,_,a=np.linalg.svd(rows);return a[-1,:3]/a[-1,3]
 def cube(p):return p[:3]+np.exp(p[6])*(SIGNS@Rotation.from_rotvec(p[3:6]).as_matrix().T)
 def residual(p,subset=None):
  subset=observations if subset is None else subset
  xyz=cube(p);values=[]
  for name,points in subset:
   xy,z=project(xyz,name);values.extend((xy@directions.T).max(0)-(points@directions.T).max(0));values.extend((xy@directions.T).min(0)-(points@directions.T).min(0))
  return np.array(values)
 center=triangulate([(n,p.mean(0)) for n,p in observations]);sizes=[np.ptp(p,axis=0).mean()*abs(project(center[None],n)[1][0])/K[0,0] for n,p in observations];edge=float(np.median(sizes))
 if not np.isfinite(edge) or edge<=0:raise ValueError('Degenerate cube initialization')
 rng=np.random.default_rng(0);best=None
 for attempt in range(16):
  p=np.r_[center,Rotation.random(random_state=rng).as_rotvec(),np.log(edge)];result=least_squares(residual,p,loss='soft_l1',f_scale=.6,max_nfev=350)
  if best is None or np.mean(residual(result.x)**2)<np.mean(residual(best.x)**2):best=result
 p=best.x;initial_rms=[float(np.sqrt(np.mean(residual(p,[(name,points)])**2))) for name,points in observations];original_views=[name for name,_ in observations];accepted=np.array(initial_rms)<2.;rejected=[]
 # Occlusion or an unrelated blue object can corrupt a single silhouette.
 # Keep a strict majority with the same fixed2px support test, then refit3D.
 # No scale is accepted from a small cherry-picked subset of detections.
 if accepted.sum()>=max(3,len(observations)//2+1) and not accepted.all():
  rejected=[dict(view=n,support_rms_px=r,reason='Cube silhouette inconsistent with common rigid3D fit') for (n,_),r,ok in zip(observations,initial_rms,accepted) if not ok];observations=[item for item,ok in zip(observations,accepted) if ok];p=least_squares(residual,p,loss='soft_l1',f_scale=.6,max_nfev=350).x
 xyz=cube(p);support=float(np.sqrt(np.mean(residual(p)**2)));cornerobs={i:[] for i in range(8)};assignments=[]
 for name,points in observations:
  xy,z=project(xyz,name);visible=ConvexHull(xy).vertices;dist=np.linalg.norm(xy[visible,None]-points[None],axis=2);a,b=linear_sum_assignment(dist)
  for i,j in zip(a,b):
   if dist[i,j]<3:
    k=int(visible[i]);cornerobs[k].append((name,points[j]));assignments.append(dict(view=name,corner=k,observed_xy=points[j].tolist(),projected_xy=xy[k].tolist(),error_px=float(dist[i,j])))
 corners={};errors=[]
 for k,obs in cornerobs.items():
  if len(obs)<2:continue
  initial=triangulate(obs);refined=least_squares(lambda x:np.concatenate([project(x[None],n)[0][0]-xy for n,xy in obs]),initial,loss='soft_l1',f_scale=.5);ee=[float(np.linalg.norm(project(refined.x[None],n)[0][0]-xy)) for n,xy in obs];errors.extend(ee);corners[k]=dict(xyz=refined.x.tolist(),views=[n for n,_ in obs],errors_px=ee)
 edges=[dict(corners=[i,j],length_units=float(np.linalg.norm(np.array(corners[i]['xyz'])-corners[j]['xyz']))) for i in corners for j in corners if i<j and np.sum(SIGNS[i]!=SIGNS[j])==1]
 depths=np.array([project(xyz,n)[1].min() for n,_ in observations]);model_edge=float(np.exp(p[6]));views=len(observations);good=views>=3 and support<2 and len(corners)>=4 and len(edges)>=3 and np.all(depths>0)
 result=dict(reference_object='blue_cube',real_edge_length_cm=2.,reconstructed_edge_length_units=model_edge if good else None,scale_cm_per_unit=2/model_edge if good else None,coordinate_frame='Actual COLMAP original units; original axes unchanged',reference_views=[n for n,_ in observations],reprojection_error_px=float(np.mean(errors)) if errors else None,reprojection_resolution=[data['width'],data['height']],boundary_support_rms_px=support,status='CALIBRATED_MULTI_VIEW_CUBE' if good else 'REVIEW_REQUIRED',calibration_mode='HISTORICAL_BOUNDARY_ASSISTED' if assisted else 'AUTOMATIC_SAM3_MULTI_VIEW',historical_assisted=assisted,independent_adjacent_edges=edges,edge_length_units_mean=float(np.mean([e['length_units'] for e in edges])) if edges else None,edge_length_units_std=float(np.std([e['length_units'] for e in edges])) if edges else None,cube_orientation=Rotation.from_rotvec(p[3:6]).as_matrix().tolist(),cube_center=xyz.mean(0).tolist(),motion_check='A static rigid cube must fit the plantCOLMAP cameras across at least3 views; low residual supports common rigid frame, not external survey truth',limits=['Semantic boundary accuracy and corner visibility limit physical scale','No surveyed gravity axis or independent dimension truth beyond supplied20mm edge','No2D pixel edge used as physical3D length'])
 detail=dict(joint_cube_corners=xyz.tolist(),independent_corners=corners,assignments=assignments,adjacent_edges=edges,signs=SIGNS.tolist(),observations=[dict(view=n,hull=p.tolist()) for n,p in observations])
 result.update(initial_reference_views=original_views,initial_view_support_rms_px=dict(zip(original_views,initial_rms)),rejected_views=rejected,accepted_view_support_rms_px={n:float(np.sqrt(np.mean(residual(p,[(n,pts)])**2))) for n,pts in observations})
 return result,detail

def calibrate(ctx):
 path=ctx.out/'calibration/scale_calibration.json'
 if path.exists():return read(path)
 if ctx.plant=='KongQueZhuYu':
  source=OLD/'calibration/scale_calibration.json';previous=read(source)
  assert previous['status']=='CALIBRATED_MULTI_VIEW_CUBE'
  previous.update(calibration_mode='REUSED_HISTORICAL_BOUNDARY_ASSISTED',historical_assisted=True,reused_from=signature(source),automatic_inference=False)
  write(path,previous);ctx.status('calibration','complete',scale_cm_per_unit=previous['scale_cm_per_unit'],calibration_mode=previous['calibration_mode']);return previous
 ctx.status('calibration','running');data,cameras=ctx.cameras();sam=read(ctx.out/'semantic_masks/complete.json');automatic=[];historical=[]
 import cv2
 from scipy.ndimage import label
 for name in sam['views']:
  mask=np.array(Image.open(ctx.out/'semantic_masks'/name/'blue_cube.png'))>0
  if 12<=mask.sum()<=data['width']*data['height']*.035:
   components,num=label(mask);parts=[components==i for i in range(1,num+1)];mask=max(parts,key=lambda m:m.sum()) if parts else mask;yy,xx=np.where(mask)
   if len(xx)>=12:
    hull=cv2.convexHull(np.c_[xx+.5,yy+.5].astype('float32'));hull=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)[:,0]
    if len(hull)>=4:
     done=read(ctx.out/'semantic_masks'/name/'complete.json')
     if not done.get('cube_historical_assisted'):automatic.append((name,hull.astype(float)))
  gt=ctx.annotation/f'{name}.json'
  if gt.exists():
   record=read(gt);polygons=[s['points'] for s in record.get('shapes',[]) if s['label']=='blue_cube']
   annotated=ctx.annotation/Path(record.get('imagePath',name+'.jpg')).name
   if not annotated.exists() or digest(annotated,ctx.meter)!=digest(ctx.raw/f'{name}.jpg',ctx.meter):continue
   if polygons:
    import pycolmap
    raw=pycolmap.Camera(**data['source_camera']);K=np.array(data['K']);points=np.array(polygons[0]);rays=raw.cam_from_img(points);xy=rays*np.diag(K)[:2]+K[:2,2];hull=cv2.convexHull(xy.astype('float32'));hull=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)[:,0]
    if len(hull)>=4:historical.append((name,hull.astype(float)))
 attempts=[];chosen=None;detail=None
 for obs,assisted in [(automatic,False),(historical,True)]:
  if len(obs)<3:attempts.append(dict(historical_assisted=assisted,status='INSUFFICIENT_OBSERVATIONS',views=len(obs)));continue
  sampled=[obs[i] for i in np.unique(np.linspace(0,len(obs)-1,min(len(obs),24),dtype=int))]
  try:result,corners=fit(ctx,sampled,assisted)
  except Exception as e:attempts.append(dict(historical_assisted=assisted,status='FIT_FAILED',error=repr(e)));continue
  attempts.append(result)
  if result['status']=='CALIBRATED_MULTI_VIEW_CUBE':chosen,detail=dict(result),corners;break
 if chosen is None:chosen=dict(reference_object='blue_cube',real_edge_length_cm=2.,reconstructed_edge_length_units=None,scale_cm_per_unit=None,coordinate_frame='ActualCOLMAP',reference_views=[],reprojection_error_px=None,status='UNCALIBRATED_INSUFFICIENT_OR_INCONSISTENT_CUBE',automatic_views=len(automatic),historical_views=len(historical))
 chosen['attempts']=attempts;write(path,chosen)
 if detail:
  write(ctx.out/'calibration/cube_corners.json',detail);canvas=Image.new('RGB',(5*350,500),'white');draw=ImageDraw.Draw(canvas);xyz=np.array(detail['joint_cube_corners'])
  for i,record in enumerate(detail['observations'][:5]):
   name=record['view'];p=np.array(record['hull']);im=ctx.image(name);lo=np.floor(p.min(0)-15).astype(int);hi=np.ceil(p.max(0)+15).astype(int);crop=im.crop((*lo,*hi));factor=min(330/max(crop.width,1),440/max(crop.height,1));crop=crop.resize((max(1,int(crop.width*factor)),max(1,int(crop.height*factor))));canvas.paste(crop,(i*350,35));xy,z=ctx.xyz_project(xyz,name)
   for a in range(8):
    for b in range(a):
     if np.sum(SIGNS[a]!=SIGNS[b])==1:draw.line([tuple((xy[a]-lo)*factor+[i*350,35]),tuple((xy[b]-lo)*factor+[i*350,35])],fill='lime',width=2)
   draw.text((i*350+4,5),name+' '+chosen['calibration_mode'],fill='black')
  png(ctx.out/'calibration/calibration_overview.png',np.array(canvas),ctx.meter)
 ctx.status('calibration','complete' if chosen['scale_cm_per_unit'] else 'uncalibrated',**{k:chosen.get(k) for k in ['scale_cm_per_unit','reconstructed_edge_length_units','calibration_mode','reference_views','reprojection_error_px']});print('CUBE',ctx.plant,chosen['status'],chosen['scale_cm_per_unit'],flush=True);return chosen

if __name__=='__main__':calibrate(Context(sys.argv[1]))
