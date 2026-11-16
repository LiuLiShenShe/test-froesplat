"""Historical semantic prompts and visibility-correct projection cache."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from common import *
from PIL import ImageDraw
from scipy.ndimage import binary_fill_holes,binary_dilation
from scipy.optimize import least_squares

def main():
 meter=Meter();mesh=read_mesh(meter);scene=cast_scene(mesh);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
 historical={};pot_support=np.zeros(len(faces),int);plant_support=np.zeros(len(faces),int)
 for path in sorted((ROOT/'03-GT-区分/KongQueZhuYu').glob('*.json')):
  j=read_json(path);masks={}
  for label in ['pot','potted_plant','blue_cube']:
   mask=Image.new('L',(530,948));draw=ImageDraw.Draw(mask)
   for s in j['shapes']:
    if s['label']==label:draw.polygon([tuple(p) for p in raw_to_common(s['points'])],fill=255)
   masks[label]=np.array(mask)>0
  masks['plant']=masks['potted_plant']&~masks['pot'];historical[path.stem]=masks
  ids,z,uv=cast(scene,path.stem)
  for key,support in [('pot',pot_support),('plant',plant_support)]:
   keep=(ids>=0)&masks[key];unique,counts=np.unique(ids[keep],return_counts=True);support[unique[counts>=2]]+=1
  for key in ['plant','pot','blue_cube']:png(OUT/'annotations/historical_semantics'/path.stem/f'{key}.png',masks[key].astype('uint8')*255,meter)
 pot_faces=np.flatnonzero((pot_support>=1)&(plant_support==0));pot_points=vertices[np.unique(faces[pot_faces])]
 npz(OUT/'annotations/geometry_guidance.npz',meter,pot_face_ids=pot_faces,pot_points=pot_points,plant_support=plant_support,pot_support=pot_support)
 vertical=np.array(read_json(OUT/'calibration/scale_calibration.json')['cube_orientation'])[:,2]
 # Cube orientation is sign/permutation ambiguous. Select the axis most aligned
 # with visible image-up vectors, then verify using target-pot3D shape.
 orientation=np.array(read_json(OUT/'calibration/scale_calibration.json')['cube_orientation'])
 image_up=np.mean([c['c2w'][:3,:3]@np.array([0.,-1.,0.]) for n,c in CAMERAS.items() if int(n)<70],axis=0)
 col=np.argmax(np.abs(orientation.T@image_up));vertical=orientation[:,col];vertical*=np.sign(vertical@image_up)
 center=np.median(pot_points,axis=0);t=(pot_points-center)@vertical
 lower,upper=np.quantile(t,[.01,.99]);horizontal=orientation[:,(col+1)%3];horizontal-=vertical*(horizontal@vertical);horizontal/=np.linalg.norm(horizontal)
 axis_b=np.cross(vertical,horizontal)
 rim=center+upper*vertical
 write(OUT/'calibration/reference_frame.json',dict(vertical_unit_vector=vertical.tolist(),horizontal_axis_a=horizontal.tolist(),horizontal_axis_b=axis_b.tolist(),vertical_source='Cube axis aligned to image-up, visually confirmed cube lies on common flat support; independently check pot-rim reference in review',cube_vertical_axis_index=int(col),reference_plane=dict(point_units=rim.tolist(),normal=vertical.tolist(),source='Upper1% of independently historical pot-supported visible surface; approximate pot rim, not soil-base truth'),base_point=None,base_status='Stem-to-soil contact point requires separate landmark',pot_center_units=center.tolist(),pot_vertical_range_units=[float(lower),float(upper)],coordinate_transform='measurement_xyz_cm = scale * ((world_xyz-base) dot [horizontalA,horizontalB,vertical]); no source geometry modified'))
 views=sorted(set([f'{n:04}' for n in range(0,210,3)]+[f'{n:04}' for n in range(0,210,14)]+list(historical)))
 ledger=[]
 for name in views:
  ids,z,uv=cast(scene,name)
  # Cache only triangle ID and two barycentric coordinates, not copied3D scenes.
  npz(OUT/'plant_only/projection_cache'/f'{name}.npz',meter,triangle_id=ids.astype('int32'),barycentric_uv=uv.astype('float16'))
  guide=(ids>=0)&np.isin(ids,pot_faces)
  if guide.any():
   ys,xs=np.where(guide);box=[max(0,int(xs.min())-8),max(0,int(ys.min())-8),min(529,int(xs.max())+8),min(947,int(ys.max())+8)]
  else:box=None
  png(OUT/'annotations/pot_projection'/f'{name}.png',guide.astype('uint8')*255,meter)
  ledger.append(dict(view=name,pot_box_xyxy=box,visible_guide_pixels=int(guide.sum())))
 write(OUT/'annotations/semantic_guidance.json',dict(views=ledger,source='Five historical independent pot polygons -> visible source mesh triangles -> projected geometric prompts. Not new human masks; all algorithm-generated correction prompts logged.',historical_views=list(historical),new_user_clicks=0))
 write(OUT/'summary/preparation_resources.json',dict(io=meter.stats(),views=len(views),original_vertices=len(vertices),original_faces=len(faces)))
 print('PREPARED',len(views),'views; potfaces',len(pot_faces),'axis',vertical,flush=True)

if __name__=='__main__':main()
