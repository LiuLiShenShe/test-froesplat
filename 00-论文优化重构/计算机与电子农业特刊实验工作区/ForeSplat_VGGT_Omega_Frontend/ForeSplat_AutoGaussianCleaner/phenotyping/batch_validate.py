"""Data invariants: dimensional scaling, camera projections and exact leaf provenance."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from batch_common import *
from batch_measure import csvread
from auto_metric_scale import LENGTH_FIELDS,LOG_FIELDS
from plyfile import PlyData

def validate(ctx):
 checks=[];failures=[]
 def check(name,condition,detail=''):
  checks.append(dict(name=name,passed=bool(condition),detail=detail))
  if not condition:failures.append(name)
 original=read(ctx.common/'complete.json');check('actual_feature_track_SfM',original.get('sparse_points',0)>100 and (original.get('valid_feature_observations',1000)>300 or ctx.plant=='KongQueZhuYu'))
 if ctx.plant!='KongQueZhuYu':
  mapping=read(ctx.common/'image_manifest.json');check('camera_RGB_identity_all_views',all(row.get('raw_sha256') and 'SHA256' in row.get('name_match_verified','') for row in mapping))
 meshinfo=read(ctx.out/'mesh/complete.json');metric=meshinfo['metric'];cal=read(ctx.out/'calibration/scale_calibration.json');check('selected_bounded_TSDF',meshinfo['relative_scene_resolution']=='robust0.5/99.5pct scene diagonal/512;5voxel SDF truncation');check('no_repeated_trait_scaling',meshinfo['subsequent_trait_scale']==1)
 if metric:
  factor=float(cal['scale_cm_per_unit']);conversion=read(ctx.out/'gaussian/metric_conversion.json');check('Gaussian_metric_once',conversion['applied_scale_count']==1 and conversion['scale_cm_per_unit']==factor)
  _,source=scalar_ply(ctx.meter.read(ctx.out/'gaussian/original_gaussians.ply'));_,scaled=scalar_ply(ctx.meter.read(ctx.out/'gaussian/metric_gaussians.ply'))
  check('same_Gaussian_schema_count',source.dtype==scaled.dtype and len(source)==len(scaled))
  for field in source.dtype.names:
   good=np.allclose(scaled[field],source[field]*factor,rtol=2e-6,atol=1e-7,equal_nan=True) if field in LENGTH_FIELDS else np.allclose(scaled[field],source[field]+np.log(factor),rtol=2e-6,atol=1e-6,equal_nan=True) if field in LOG_FIELDS else np.array_equal(source[field],scaled[field],equal_nan=True)
   check('Gaussian_field_'+field,good)
  rawdata,raw=ctx.cameras(False);cmdata,cm=ctx.cameras(True);check('camera_intrinsics_identical',rawdata['K']==cmdata['K']);check('camera_scale_once',cmdata['applied_scale_count']==1)
  xyz=np.stack([source[a][::max(1,len(source)//128)] for a in ['x','y','z']],1).astype(float)
  for name in sorted(raw)[::max(1,len(raw)//5)]:
   xy0,z0=ctx.xyz_project(xyz,name,False);xy1,z1=ctx.xyz_project(xyz*factor,name,True);visible=(z0>.001)&np.isfinite(xy0).all(1)&np.isfinite(xy1).all(1);check('uniform_camera_projection_'+name,raw[name]['R']==cm[name]['R'] and np.allclose(xy0[visible],xy1[visible],atol=1e-6) and np.allclose(np.array(cm[name]['center']),np.array(raw[name]['center'])*factor))
  if ctx.plant=='KongQueZhuYu':check('metric_depth_units',meshinfo['metric_source']=='Scaled existing native depth/camera is exact uniform equivariance')
  else:
   units=read(ctx.out/'mesh/renderer_units.json');check('native_depth_search_range_converted',np.isclose(units['metric_depth_search_range'],.4*factor) and np.isclose(units['near_plane'],.2*factor))
 else:check('uncalibrated_has_no_metric_gaussian',not(ctx.out/'gaussian/metric_gaussians.ply').exists())
 mesh=ctx.mesh();v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);check('finite_mesh',len(f)>0 and np.isfinite(v).all());catalog=read(ctx.out/'leaf_instances/leaf_instances.json');identities=[a['leaf_id'] for a in catalog['instances']];check('unique_leaf_IDs',len(identities)==len(set(identities)));claimed=np.zeros(len(f),np.uint16)
 for inst in catalog['instances']:
  leaf=inst['leaf_id'];folder=ctx.out/'leaf_instances'/leaf
  with np.load(io.BytesIO(ctx.meter.read(folder/'source_indices.npz'))) as d:fid=d['original_face_ids'];vi=d['original_vertex_ids']
  sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import read_mesh
  sub=read_mesh(folder/('leaf_cm.ply' if metric else 'leaf_units.ply'),ctx.meter);sv=np.asarray(sub.vertices);sf=np.asarray(sub.triangles);check(leaf+'_exact_source_geometry',np.array_equal(sv,v[vi]) and np.array_equal(vi[sf],f[fid]));claimed[fid]+=1
  if mesh.has_vertex_normals():check(leaf+'_source_normals_preserved',np.array_equal(np.asarray(sub.vertex_normals),np.asarray(mesh.vertex_normals)[vi]))
  check(leaf+'_stable_multiview',len(set(inst['views']))>=3);check(leaf+'_viewer_RGB_landmarks',all((folder/a).exists() for a in ['viewer.html','reference.png','reference_rgb.png','instance.json','landmarks.json']))
 check('triangles_have_unique_leaf_identity',claimed.max(initial=0)<=1)
 for filename in ['plant_measurements.csv','leaf_measurements.csv']:
  for row in csvread(ctx.out/'phenotype'/filename):
   check(filename+'_group',row.get('group')==ctx.record['group']);check(filename+'_trait_multiplier',float(row['trait_multiplier'])==1)
   if not metric:check(filename+'_uncalibrated_cm_blank',all(not value for key,value in row.items() if key.endswith('_cm') or key.endswith('_cm2')))
 check('native30K_complete',(ctx.out/'gaussian/original/complete.json').exists());check('manual_interface_template',(ctx.out/'phenotype/manual_reference.csv').exists());sem=read(ctx.out/'plant_only/complete.json');quality=[]
 if not metric:quality.append('No accepted multi-view cube scale; centimetres omitted')
 if not sem['plant_faces']:quality.append('No supported plant-only mesh')
 if not sem['pot_faces']:quality.append('No supported independent pot mesh')
 if not identities:quality.append('No leaf has >=3view stable identity')
 quality.append('Anatomical endpoints, whole-leaf completeness, vertical and base remain unconfirmed')
 result=dict(passed=not failures,delivery_complete=not failures and sem['plant_faces']>0 and sem['pot_faces']>0 and len(identities)>0,checks=checks,failures=failures,quality_limits=quality,metric=metric,leaf_count=len(identities),io=ctx.meter.stats());write(ctx.out/'summary/validation.json',result);ctx.status('validation','complete' if not failures else 'failed',passed=not failures,failures=failures)
 if failures:raise AssertionError(', '.join(failures))
 print('VALIDATED',ctx.plant,'metric',metric,'leaves',len(identities),flush=True)

if __name__=='__main__':
 validate(Context(sys.argv[1]));sys.stdout.flush();sys.stderr.flush();os._exit(0)
