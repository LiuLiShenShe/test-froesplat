"""Independent geometry checks for calibration, masks and actual video outputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from common import *
from PIL import ImageDraw

def main():
 meter=Meter();mesh=read_mesh(meter);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)
 labels=np.load(OUT/'leaf_instances/triangle_leaf_ids.npy');scene=None;records=[]
 video=read_json(OUT/'leaf_instances/video_propagation.json');canvas=Image.new('RGB',(7*265,3*494),'white');draw=ImageDraw.Draw(canvas)
 for row in video['records']:
  name=row['view'];leaf=int(row['leaf_id'].split('_')[1]);cache=OUT/'plant_only/projection_cache'/f'{name}.npz'
  if cache.exists():
   with np.load(io.BytesIO(meter.read(cache))) as d:ids=d['triangle_id']
  else:
   if scene is None:scene=cast_scene(mesh)
   ids,z,uv=cast(scene,name);npz(cache,meter,triangle_id=ids.astype('int32'),barycentric_uv=uv.astype('float16'))
  mask=np.array(Image.open(OUT/'leaf_instances'/row['leaf_id']/'video_masks'/f'{name}.png'))>0
  pixelids=np.zeros(ids.shape,np.uint16);valid=ids>=0;pixelids[valid]=labels[ids[valid]]
  target=pixelids==leaf;assigned=mask&(pixelids>0);agreement=int((mask&target).sum())/max(1,int(assigned.sum()));recall=int((mask&target).sum())/max(1,int(target.sum()))
  threshold=bool(agreement>=.7 and recall>=.5 and (mask&target).sum()>=100)
  faceids=np.unique(ids[mask&target]);centroid=vertices[faces[faceids]].reshape(-1,3).mean(0).tolist() if len(faceids) else None
  rec=dict(**row,assigned_surface_pixel_precision=agreement,visible_instance_recall=recall,geometry_identity_accepted=threshold,common_original_face_count=len(faceids),shared_surface_centroid_units=centroid,rule='First-hit source triangles; identity precision>=.7, visible recall>=.5, >=100 matching pixels',used_to_expand_3d_labels=False,reason='Tracking validation only; original8keyview geometry fusion retained')
  records.append(rec)
  im=np.array(rgb(name,meter));im[mask]=(.55*im[mask]+.45*np.array([70,210,120])).astype('uint8');im=Image.fromarray(im);d=ImageDraw.Draw(im);d.text((8,8),row['leaf_id']+' '+name,fill='white',stroke_width=1,stroke_fill='black');d.text((8,28),f'3D precision {agreement:.3f}',fill='white',stroke_width=1,stroke_fill='black')
  col=int(name)-84;r=[3,8,9].index(leaf);canvas.paste(im.resize((265,474)),(col*265,r*494+20));draw.text((col*265+4,r*494+3),'PASS' if threshold else 'REVIEW',fill='green' if threshold else 'red')
 png(OUT/'leaf_instances/video_geometry_review.png',np.array(canvas),meter)
 old_status=video['status'];complete=len(records)==21 and all((OUT/'leaf_instances'/x['leaf_id']/'video_masks'/f"{x['view']}.png").exists() for x in records)
 write(OUT/'leaf_instances/video_geometry_validation.json',dict(status='VALIDATED' if complete else 'INCOMPLETE',records=records,accepted=sum(x['geometry_identity_accepted'] for x in records),total=len(records),original_run_status=old_status,propagation_frames_complete=complete,cleanup_issue='Deprecated reset_state call occurred only after all7 frames were saved; removed from new code' if 'reset_state' in old_status else None,io=meter.stats()))
 if complete and 'reset_state' in old_status:
  video.update(original_run_status=old_status,status='PROPAGATION_OUTPUTS_COMPLETE_CLEANUP_ERROR_RECOVERED_BY_VALIDATION',geometry_validation='video_geometry_validation.json');write(OUT/'leaf_instances/video_propagation.json',video)
 # Mask agreement here is guided-concordance QA, not held-out accuracy:
 # historical annotations also generated the corrective box/point guidance.
 qa=[];cube_dir=OUT/'calibration/cube_masks';cube_dir.mkdir(exist_ok=True)
 for name in read_json(OUT/'calibration/scale_calibration.json')['reference_views']:
  source=OUT/'semantic_masks'/name/'blue_cube.png';dest=cube_dir/f'{name}.png'
  if not dest.exists():dest.symlink_to(source)
  for kind in ['plant','pot','blue_cube']:
   gt=OUT/'annotations/historical_semantics'/name/f'{kind}.png'
   truth=np.array(Image.open(gt))>0;pred=np.array(Image.open(OUT/'semantic_masks'/name/f'{kind}.png'))>0
   qa.append(dict(view=name,label=kind,IoU=float((truth&pred).sum()/max(1,(truth|pred).sum())),historical_pixels=int(truth.sum()),sam_pixels=int(pred.sum())))
 write(OUT/'semantic_masks/guided_concordance.json',dict(records=qa,held_out=False,interpretation='Agreement with reused historical guidance; not independent semantic accuracy'))
 # The cube rests on the same board in RGB; camera orbit supplies a separate
 # direction check, but neither replaces a measured spirit-level gravity axis.
 frame=read_json(OUT/'calibration/reference_frame.json');vertical=np.array(frame['vertical_unit_vector']);camera=np.array([c['c2w'][:3,3] for c in CAMERAS.values()]);_,singular,axes=np.linalg.svd(camera-camera.mean(0),full_matrices=False);orbit=axes[-1];orbit*=np.sign(orbit@vertical);angle=float(np.degrees(np.arccos(np.clip(orbit@vertical,-1,1))))
 sem=np.load(OUT/'plant_only/triangle_semantics.npy');pot=vertices[np.unique(faces[sem==2])];heights=pot@vertical;lo,hi=np.quantile(heights,[.02,.98]);rim=pot[(heights>=hi-(hi-lo)*.035)&(heights<=hi)];_,sv,rimaxes=np.linalg.svd(rim-rim.mean(0),full_matrices=False);rimnormal=rimaxes[-1];rimnormal*=np.sign(rimnormal@vertical);rimangle=float(np.degrees(np.arccos(np.clip(rimnormal@vertical,-1,1))))
 calibration=read_json(OUT/'calibration/scale_calibration.json');plane=np.array(calibration['cube_center'])-vertical*calibration['reconstructed_edge_length_units']/2
 frame['reference_plane']=dict(point_units=plane.tolist(),normal=vertical.tolist(),source='Fitted cube bottom face on common support board, identified by agent direct RGB inspection; not soil level')
 frame['vertical_source']='Normal to the cube face visually resting on the common support board; camera-orbit axis is a separate consistency check'
 frame.update(vertical_verification=dict(support='Agent direct visual inspection: cube has a face resting on common flat board; chosen edge normal to that face',orbit_plane_normal=orbit.tolist(),orbit_axis_angle_deg=angle,orbit_plane_singular_values=singular.tolist(),pot_upper_band_plane_normal=rimnormal.tolist(),pot_upper_band_angle_deg=rimangle,pot_plane_points=len(rim),pot_rim_check_status='INCONCLUSIVE_TOO_FEW_POINTS' if len(rim)<20 else 'Unconfirmed upper-band estimate',pot_plane_rms_units=float(np.sqrt(np.mean(((rim-rim.mean(0))@rimnormal)**2))),status='PROVISIONAL_SUPPORT_NORMAL; cameraorbit is a consistency check, not surveyed gravity',limits='Orbit trajectory is not planar; sparse upper-band pot fragments do not establish a rim plane'))
 write(OUT/'calibration/reference_frame.json',frame)
 print('VIDEO_GEOMETRY',sum(x['geometry_identity_accepted'] for x in records),'/',len(records),'VERTICAL_CHECK',angle,rimangle,flush=True)

if __name__=='__main__':main()
