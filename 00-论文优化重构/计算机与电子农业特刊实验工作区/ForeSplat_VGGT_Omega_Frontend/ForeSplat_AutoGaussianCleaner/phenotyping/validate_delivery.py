"""Artifact/provenance checks plus known-geometry tests, not biological accuracy."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from common import *
from measurement_geometry import surface_graph,surface_path,one_side_area
from manual_errors import rows
from html.parser import HTMLParser

EXPECTED_POST='f414f1352199ffbf3f90265556bf56a0f02cdfafbe37eccd8398d0474a0d34e9'
def sha(path,meter=None):return hashlib.sha256(meter.read(path) if meter else Path(path).read_bytes()).hexdigest()

def geometry_tests():
 # Synthetic analytic fixtures only; never enter experimental trait CSVs.
 v=np.array([[0.,0,0],[2,0,0],[2,1,0],[0,1,0]]);f=np.array([[0,1,2],[0,2,3]])
 ids,length=surface_path(v,surface_graph(v,f),[0,1]);assert np.isclose(length,2)
 double=np.r_[f,f[:,::-1]];single,total,side,n=one_side_area(v,double,np.array([0,0,3]));assert np.isclose(single,2) and np.isclose(total,4)
 angle=np.linspace(0,np.pi,101);v=np.r_[np.c_[np.cos(angle),np.sin(angle),np.zeros(101)],np.c_[np.cos(angle),np.sin(angle),np.ones(101)]];f=[]
 for i in range(100):f.extend([[i,i+1,i+102],[i,i+102,i+101]])
 ids,arc=surface_path(v,surface_graph(v,np.array(f)),[0,100]);assert abs(arc-np.pi)<.001 and arc>np.linalg.norm(v[0]-v[100])
 v=np.array([[0.,0,0],[1,0,0],[0,1,0],[5,0,0],[6,0,0],[5,1,0]]);f=np.array([[0,1,2],[3,4,5]]);assert surface_path(v,surface_graph(v,f),[0,3])==(None,None)
 return dict(flat_length=length,single_side_area=single,double_side_total=total,curved_arc=arc,analytic_arc=float(np.pi),disconnected_path_rejected=True,experiment_data=False)

class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag in ['a','img','iframe']:
   for key,value in attrs:
    if key in ['href','src']:self.links.append(value)

def main():
 meter=Meter();source=read_mesh(meter);vertices=np.asarray(source.vertices);faces=np.asarray(source.triangles);source_hash=sha(mesh_path(),meter);assert source_hash==EXPECTED_POST,'Frozen recommended source mesh changed'
 assert vertices.shape==(264700,3) and faces.shape==(484124,3)
 sem=np.load(OUT/'plant_only/triangle_semantics.npy');ids=np.load(OUT/'leaf_instances/triangle_leaf_ids.npy');assert sem.shape==ids.shape==(len(faces),);assert np.all(np.isin(sem,np.arange(5))) and np.all(np.isin(ids,np.arange(13)));assert np.all(sem[ids>0]==1)
 import open3d as o3d
 sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import read_mesh as load
 audits=[]
 for folder,prefix in [(OUT/'plant_only','plant_only'),(OUT/'plant_only','pot')]+[(OUT/'leaf_instances'/f'Leaf_{i:03}','leaf') for i in range(1,13)]:
  archive=folder/('source_indices.npz' if prefix=='leaf' else f'{prefix}_source_indices.npz')
  with np.load(io.BytesIO(meter.read(archive))) as d:fi=d['original_face_ids'];vi=d['original_vertex_ids']
  mesh=load(folder/f'{prefix}.ply',meter);v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles)
  assert np.allclose(v,vertices[vi],rtol=0,atol=1e-12),'Vertex coordinates changed'
  assert np.array_equal(vi[f],faces[fi]),'Source triangle identity changed'
  normals=bool(source.has_vertex_normals())
  if normals:assert np.allclose(np.asarray(mesh.vertex_normals),np.asarray(source.vertex_normals)[vi],rtol=0,atol=1e-12),'Source normals changed'
  audits.append(dict(model=str((folder/f'{prefix}.ply').relative_to(OUT)),faces=len(fi),vertices=len(vi),source_coordinates_exact=True,source_faces_exact=True,source_normals_present=normals,source_normals_preserved=normals))
 cal=read_json(OUT/'calibration/scale_calibration.json');assert np.isclose(cal['real_edge_length_cm']/cal['reconstructed_edge_length_units'],cal['scale_cm_per_unit']);assert len(cal['reference_views'])==5 and len(cal['independent_adjacent_edges'])==12
 frame=read_json(OUT/'calibration/reference_frame.json');rotation=np.array(frame['measurement_rotation_rows']);assert np.allclose(rotation@rotation.T,np.eye(3),atol=1e-9)
 traits=rows(OUT/'phenotype/leaf_measurements.csv');assert len(traits)==12;assert len({r['leaf_id'] for r in traits})==12
 for r in traits:
  assert r['formal_whole_leaf_valid']=='False'
  for key in ['leaf_length_cm','leaf_width_cm','leaf_area_cm2']:
   if r[key]:assert np.isfinite(float(r[key])) and float(r[key])>0
  if r['leaf_length_cm']:assert float(r['leaf_length_cm'])>=.99*float(r['straight_leaf_length_cm'])
  landmarks=read_json(OUT/'leaf_instances'/r['leaf_id']/'landmarks.json');path=np.array(landmarks['centerline_source_vertex_ids'],int)
  if len(path)>1:
   actual=vertices[path];assert np.allclose(actual,landmarks['centerline_xyz_units'],atol=1e-12)
   selected=faces[landmarks['one_side_source_face_ids']];edges=np.concatenate([selected[:,[0,1]],selected[:,[1,2]],selected[:,[2,0]]]);edges.sort(1);codes=np.unique(edges[:,0].astype('int64')*len(vertices)+edges[:,1]);pairs=np.sort(np.stack([path[:-1],path[1:]],axis=1),axis=1);assert np.all(np.isin(pairs[:,0].astype('int64')*len(vertices)+pairs[:,1],codes)),'Path crosses missing surface'
   if r['leaf_length_cm']:assert np.isclose(np.linalg.norm(np.diff(actual,axis=0),axis=1).sum()*cal['scale_cm_per_unit'],float(r['leaf_length_cm']))
 errors=rows(OUT/'phenotype/phenotype_errors.csv');pairs=rows(OUT/'phenotype/paired_measurements.csv')
 for r in errors:
  if int(r['n_pairs'])==0:assert all(r[k]=='' for k in ['MAE','RMSE','Bias','relative_error','R2'])
 parser=Links();parser.feed((OUT/'summary/index.html').read_text());local=[link.split('#')[0] for link in parser.links if not link.startswith(('http:','https:','data:','#'))];assert all((OUT/'summary'/p).exists() for p in local),'Broken local summary link'
 tests=geometry_tests();ui=read_json(OUT/'summary/ui_validation.json');assert ui['sam_http']==200 and ui['mask_pixels']>0
 source_files={str(p.relative_to(CLEANER)):sha(p) for p in sorted((OUT/'scripts').resolve().glob('*')) if p.suffix in ['.py','.html']}
 # SAM3 is inside the parent repository, not a separately versioned checkout.
 # File hashes and checkpoint digest provide a truthful local version pin.
 sam=ROOT/'第三方源码/SAM3-latest';sam_files={str(p.relative_to(sam)):sha(p) for p in [sam/'sam3/model_builder.py',sam/'sam3/model/sam3_image_processor.py',sam/'sam3/model/sam3_image.py',sam/'sam3/model/sam3_tracker_predictor.py'] if p.exists()}
 manifest=dict(source_mesh=dict(path=str(mesh_path()),sha256=source_hash,expected_at_stage_start=EXPECTED_POST),cameras={str(p.relative_to(PRE)):sha(p,meter) for p in [PRE/'common_input/colmap_source.json',PRE/'common_input/camera_trajectory.json']},sam3=dict(path=str(sam),upstream_commit='Not independently recorded: bundled source is tracked in parent repository',local_file_sha256=sam_files,checkpoint_sha256=read_json(OUT/'summary/sam_pilot_resources.json')['weight_digest']['sha256'],official_repository='https://github.com/facebookresearch/sam3'),implementation_files=source_files)
 write(OUT/'summary/source_manifest.json',manifest)
 validation=dict(status='PASS_ARTIFACTS_WITH_PROVISIONAL_BIOLOGICAL_MEASUREMENTS',source_mesh_unchanged=True,source_mesh_sha256=source_hash,original_vertices=len(vertices),original_faces=len(faces),derived_meshes=audits,calibration_reference_views=5,leaf_instances=12,whole_leaf_formally_valid=0,real_manual_pairs=len(pairs),error_metrics_empty_when_n_zero=True,local_index_links=len(local),analytic_geometry_tests=tests,ui=ui,physical_gpu=1,io=meter.stats(),limitations=['Artifact and analytic-fixture checks do not establish plant measurement accuracy','All twelve leaves remain partial observed surfaces; anatomical endpoints unconfirmed','Support normal and stem-soil base require physical/human confirmation','No independent matched manual measurements; no formal errors or generalization accuracy'])
 write(OUT/'summary/DELIVERY_VALIDATION.json',validation);print('DELIVERY_VALIDATION_PASS',len(audits),'derived meshes',len(local),'links',flush=True)

if __name__=='__main__':main()
