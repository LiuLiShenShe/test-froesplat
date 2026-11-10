"""Validate user-required real models, native stages, full comparisons and images."""
import argparse,csv,json,math,os
from collections import Counter,defaultdict
from pathlib import Path
from queue_io import read_status,write_status

OUT=Path(__file__).resolve().parents[1]
OLD=OUT.parent/'geometry_mesh_final_selection'
EXPECTED_IDS={0,26,53,79,105,131,157,183}

def rows(path):
 return list(csv.DictReader(path.open())) if path.is_file() else []

def header(path):
 properties=[];vertices=faces=0;element=None
 with path.open('rb') as f:
  for _ in range(300):
   line=f.readline().decode('ascii','replace').strip()
   if line.startswith('element '):
    _,element,count=line.split()
    if element=='vertex':vertices=int(count)
    if element=='face':faces=int(count)
   elif line.startswith('property ') and element=='vertex':properties.append(line.split()[-1])
   if line=='end_header':return vertices,faces,properties
 raise ValueError(f'Invalid actual PLY header: {path}')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
 checks=[]
 def check(name,passed,evidence='',details=None):
  checks.append(dict(requirement=name,passed=bool(passed),evidence=str(evidence),details=details))
 for name in ['ALL_ARTIFACTS.csv','GAUSSIAN_PLY_INDEX.md','MESH_FILE_INDEX.md','HOW_TO_VIEW_RESULTS.md']:
  path=OUT/'artifact_index'/name;check('actual delivery index '+name,path.is_file(),path)
 artifacts=rows(OUT/'artifact_index/ALL_ARTIFACTS.csv')
 problems=[]
 for row in artifacts:
  p=Path(row['absolute_path'])
  if not p.is_absolute() or not p.is_file() or str(p.resolve())!=str(p):
   problems.append(dict(path=str(p),reason='Missing or unresolved absolute path'));continue
  lookup=Path('/data/fj/F2DMAS')/row['relative_path']
  if not lookup.is_file() or lookup.resolve()!=p:
   problems.append(dict(path=str(lookup),reason='Actual repository-relative lookup does not resolve to indexed absolute file'))
  if p.stat().st_size!=int(row['file_size_bytes']) and 'rolling' not in row['status']:
   problems.append(dict(path=str(p),reason='Indexed file size differs'))
  if row['artifact_type'] in ['Gaussian PLY','Gaussian Model','Point Cloud PLY','Mesh PLY'] and p.suffix=='.ply':
   n,f,props=header(p)
   native='opacity' in props and 'f_dc_0' in props and 'scale_0' in props
   if row['artifact_type']=='Point Cloud PLY' and (native or f):problems.append(dict(path=str(p),reason='Point cloud misclassified'))
   if row['artifact_type'] in ['Gaussian PLY','Gaussian Model'] and (not native or f):problems.append(dict(path=str(p),reason='Native primitive misclassified'))
   if row['artifact_type']=='Mesh PLY' and not f:problems.append(dict(path=str(p),reason='Empty/nonmesh PLY'))
   if n!=int(row['vertices_or_gaussians']) or f!=int(row['faces']):problems.append(dict(path=str(p),reason='Actual header counts differ'))
  source=row['source_checkpoint']
  if source and (not Path(source).is_absolute() or not Path(source).is_file()):problems.append(dict(path=source,reason='Source checkpoint missing'))
 check('All indexed paths sizes counts and native file types verified',bool(artifacts) and not problems,OUT/'artifact_index/ALL_ARTIFACTS.csv',problems)
 map_problems=[]
 for row in artifacts:
  kind=row['artifact_type'];coordinate=row['coordinate_system'];path=Path(row['absolute_path'])
  if kind in ['Depth Map','Normal Map','RGB Map','Alpha Map','Stereo Confidence'] and 'pixel grid' not in coordinate:
   map_problems.append(dict(path=str(path),reason='Map is not identified as an image grid'))
  if kind=='Depth Map' and 'camera-axis Z' not in coordinate:
   map_problems.append(dict(path=str(path),reason='Depth coordinate convention missing'))
  if kind=='Normal Map' and not any(frame in coordinate for frame in ['camera-coordinate','world-coordinate']):
   map_problems.append(dict(path=str(path),reason='Normal vector frame missing'))
  if kind=='Exposure Parameters' and 'image-color space' not in coordinate:
   map_problems.append(dict(path=str(path),reason='Exposure treated as spatial geometry'))
  if row['experiment_id']=='E14' and any(name in path.parts for name in ['estimated_dense_depth','estimated_dense_normal']):
   if 'BEFORE' not in coordinate or row['source_checkpoint']:
    map_problems.append(dict(path=str(path),reason='Upstream prior falsely attributed to native terminal checkpoint or already aligned world frame'))
 check('Image maps normal frames exposure and upstream prior coordinates explicit',bool(artifacts) and not map_problems,OUT/'artifact_index/ALL_ARTIFACTS.csv',map_problems)
 for eid,folder in [('E12','E12_GGGS_EO'),('E13','E13_QGS'),('E14','E14_2D_SuGaR')]:
  root=OUT/folder
  marker=root/('densification_events.complete.json' if eid=='E12' else 'training_complete.json')
  state=read_status(marker)
  checkpoint=root/('training/chkpnt30000.pth' if eid=='E14' else 'checkpoints/chkpnt30000.pth')
  ply=root/('training/point_cloud/iteration_30000/point_cloud.ply' if eid=='E14' else 'checkpoints/point_cloud/iteration_30000/point_cloud.ply')
  native_ok=ply.is_file() and checkpoint.is_file()
  if native_ok:
   n,f,props=header(ply);scales=sum(k.startswith('scale_') for k in props)
   native_ok=n>0 and f==0 and scales=={'E12':3,'E13':6,'E14':2}[eid]
  if eid!='E12':native_ok=native_ok and state.get('status')=='complete' and state.get('iterations')==30000
  check(eid+' actual native30K PLY and Checkpoint',native_ok,marker,state)
  terminal=[a for a in artifacts if a['experiment_id']==eid and a['absolute_path']==str(ply.resolve())]
  check(eid+' terminal native model in actual index',native_ok and bool(terminal),ply)
 refine=read_status(OUT/'E14_2D_SuGaR/refinement_complete.json')
 prior=read_status(OUT/'E14_2D_SuGaR/prior_numerical_validation.json')
 check('E14 actual208 Metric3D depth/normal files retain explicit upstream frame and usable numerical values',
       prior.get('status')=='complete' and prior.get('views')==208 and prior.get('original_predictions_unchanged'),
       OUT/'E14_2D_SuGaR/prior_numerical_validation.json',prior.get('problems'))
 pruning=read_status(OUT/'E14_2D_SuGaR/cluster_pruning/stage_provenance_validation.json')
 check('E14 exact native before/removed/after DBSCAN and paired checkpoints',
       pruning.get('status')=='complete' and pruning.get('all_PLY_fields_bitwise_partition_source') and pruning.get('before_and_after_checkpoint_fields_bitwise_match'),
       OUT/'E14_2D_SuGaR/cluster_pruning/stage_provenance_validation.json')
 terminal_audit=read_status(OUT/'E14_2D_SuGaR/native_terminal_provenance_validation.json')
 check('E14 actual30K native PLY bitwise matches readable native Checkpoint with numerical quality separately reported',
       terminal_audit.get('status')=='complete' and terminal_audit.get('all_saved_PLY_parameters_bitwise_equal') and terminal_audit.get('iteration')==30000,
       OUT/'E14_2D_SuGaR/native_terminal_provenance_validation.json')
 files=[refine.get(k,'') for k in ['native_model','gaussian_ply','textured_obj','refined_raw_ply','refined_post_ply']]
 check('E14 actual native15K refinement, model, Gaussian, raw/post and textured OBJ',
       refine.get('status')=='complete' and refine.get('iterations')==15000 and all(p and Path(p).is_file() for p in files),
       OUT/'E14_2D_SuGaR/refinement_complete.json',refine)
 texture_paths=[];texture_errors=[]
 obj=Path(refine['textured_obj']) if refine.get('textured_obj') else None
 if obj and obj.is_file():
  import shlex
  for line in obj.read_text(errors='replace').splitlines():
   if not line.startswith('mtllib '):continue
   for name in shlex.split(line)[1:]:
    mtl=(obj.parent/name).resolve();texture_paths.append(str(mtl))
    if not mtl.is_file():texture_errors.append('Missing actual MTL '+str(mtl));continue
    for material_line in mtl.read_text(errors='replace').splitlines():
     if material_line.startswith('map_Kd '):
      texture=(mtl.parent/shlex.split(material_line)[-1]).resolve();texture_paths.append(str(texture))
      if not texture.is_file():texture_errors.append('Missing actual texture '+str(texture))
  indexed={r['absolute_path'] for r in artifacts}
  texture_errors += ['OBJ dependency absent from index '+p for p in texture_paths if p not in indexed]
 check('E14 native textured OBJ actual MTL and texture dependencies all exist and are indexed',
       bool(texture_paths) and not texture_errors,OUT/'artifact_index/ALL_ARTIFACTS.csv',texture_errors)
 stages=rows(OUT/'comparison/ALL_MESH_EXPERIMENTS.csv')
 expected=['M1_Omega_NKSR','M2_Standard_NKSR','M3_E10_GGGS_NKSR','M4_E10_GGGS_TSDF','M5_E11_2DGS_TSDF',
  'P1','P2_TSDF','P2_Tetrahedra','P3_native','P3_shared',
  'P4_expected_mask','P4_expected_mask_alpha','P4_median_mask','P4_median_mask_alpha',
  'P4_median_mask_alpha_voxel075','P4_median_mask_alpha_voxel125',
  'E12_TSDF','E12_Tetrahedra','E13_QGS','E14_cluster_before','E14_cluster_after','E14_initial','E14_refined']
 for method in expected:
  selected=[r for r in stages if r['mesh_id']==method]
  present={r['mesh_stage'] for r in selected}
  ok={'raw','post'}<=present
  for row in selected:
   for key in ['silhouette_iou','boundary_f1','boundary_chamfer_px','foreground_coverage','background_geometry_ratio','hole_area','surface_roughness']:
    try:ok=ok and math.isfinite(float(row[key]))
    except (ValueError,KeyError,TypeError):ok=False
  check(method+' actual raw/post and unified208 geometry metrics',ok,OUT/'comparison/ALL_MESH_EXPERIMENTS.csv',sorted(present))
 gaussian=rows(OUT/'comparison/ALL_GAUSSIAN_EXPERIMENTS.csv')
 frozen=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text())
 identity={int(r['source_id']):(r['gt_sha256'],r['mask_sha256']) for r in frozen}
 groups=defaultdict(list)
 for row in rows(OUT/'comparison/rendering_metrics.csv'):groups[(row['experiment'],row.get('stage','raw'))].append(row)
 identity_errors=[]
 for key,records in groups.items():
  ids=[int(r['source_id']) for r in records]
  if len(records)!=208 or len(set(ids))!=208 or set(ids)!=set(identity):identity_errors.append(dict(group=key,reason='Source IDs do not equal actual frozen208 manifest'))
  for row in records:
   if (row.get('gt_sha256'),row.get('mask_sha256'))!=identity.get(int(row['source_id'])):identity_errors.append(dict(group=key,source_id=row['source_id'],reason='GT/RAP identity differs'))
 check('Every available photometric stage has exact frozen208 IDs and GT/RAP hashes',bool(groups) and not identity_errors,OUT/'comparison/rendering_metrics.csv',identity_errors)
 for eid in [f'E{i}' for i in range(1,15)]+['P2','E14_initial','E14_cluster_before','E14_cluster_after','E14_refined']:
  selected=next((r for r in gaussian if r['experiment_id']==eid),{})
  check(eid+' actual Gaussian and unified208 photometric metrics',selected.get('status')=='model_and208_metrics_verified',OUT/'comparison/ALL_GAUSSIAN_EXPERIMENTS.csv',selected.get('status','missing'))
 gauss_protocol=read_status(OUT/'comparison/visualizations/gaussian/PROTOCOL.json')
 for method in ['E10','P2_Cleaned','E11','E12_raw','E12_corrected','E13_QGS','E14_2D_30K','E14_Refined']:
  selected=[r for r in gauss_protocol.get('records',[]) if r['method']==method]
  check(method+' actual fixed8 RGB Alpha Depth Normal and common crop images',
        {r['source_id'] for r in selected}==EXPECTED_IDS and all(Path(r['output']).is_dir() for r in selected),
        OUT/'comparison/visualizations/gaussian/PROTOCOL.json')
 mesh_protocol=read_status(OUT/'comparison/visualizations/mesh_comparison/PROTOCOL.json')
 visual_names=['M1_Omega_NKSR','M2_Standard_NKSR','M3_E10_GGGS_NKSR',*expected[3:]]
 for method in visual_names:
  check(method+' actual fixed8 mesh gray normal color wire and common ROIs',
        method in mesh_protocol.get('methods',[]) and set(mesh_protocol.get('views',[]))==EXPECTED_IDS,
        OUT/'comparison/visualizations/mesh_comparison/PROTOCOL.json')
 highlight=read_status(OUT/'comparison/visualizations/highlight_correspondence/PROTOCOL.json')
 check('All requested native brightness-query image comparisons',
       {'E10','E11','E12_raw','E12_corrected','E13_QGS','E14_2D_30K','E14_refined_aux'}<=set(highlight.get('methods',[])) and highlight.get('all_paths_exist',False),
       OUT/'comparison/visualizations/highlight_correspondence/PROTOCOL.json')
 for name in ['E12_ANALYSIS.md','E13_ANALYSIS.md','E14_ANALYSIS.md','P1_P4_ANALYSIS.md','FULL_HISTORY_COMPARISON.md']:
  path=OUT/name;check('Actual analysis '+name,path.is_file(),path)
 layer_rows=rows(OUT/'comparison/same_ray_mesh_layers/ALL_MESH_LAYER_METRICS.csv')
 measured={r['evaluated_mesh_path'] for r in layer_rows if r['group']=='all_fixed_queries' and r['queries']=='24375' and r['source_views']=='69'}
 standard_paths={r['evaluated_mesh_path'] for r in stages if not r['mesh_id'].startswith('M1_') and r.get('evaluated_mesh_path')}
 check('Actual same-ray Mesh intersections for every available standard-gauge stage; primary208 view metrics separate',
       bool(standard_paths) and standard_paths<=measured,OUT/'comparison/same_ray_mesh_layers/ALL_MESH_LAYER_METRICS.csv',sorted(standard_paths-measured))
 joint=rows(OUT/'comparison/same_ray_mesh_layers/MATCHED_FIRST_PAIR_SPACING.csv')
 joint_names={r['candidate_mesh_id'] for r in joint if r['reference_mesh_id']=='M4_E10_GGGS_TSDF' and r['group']=='all_fixed_queries'}
 required_joint={'P1','P2_TSDF','P3_shared','E12_TSDF','E13_QGS','E14_initial','E14_refined'}
 check('Actual first-face spacing on jointly eligible fixed source rays for all terminal routes',required_joint<=joint_names,
       OUT/'comparison/same_ray_mesh_layers/MATCHED_FIRST_PAIR_SPACING.csv',sorted(required_joint-joint_names))
 shapes=rows(OUT/'comparison/native_primitive_shapes/ALL_NATIVE_SHAPE_SUMMARY.csv')
 present_shapes={r['method'] for r in shapes if r['group']=='original_RGB_brightness' and r['queries']=='134'}
 required_shapes={'E10','P2','E11','E12','E13','E14_30K','E14_refined_aux'}
 check('Native shape definitions, nearest-center association and originalRGB brightness cohort separately recorded for every terminal Gaussian',
       required_shapes<=present_shapes,OUT/'comparison/native_primitive_shapes/ALL_NATIVE_SHAPE_SUMMARY.csv',sorted(required_shapes-present_shapes))
 paired=rows(OUT/'comparison/paired_view_changes/PAIRED_VIEW_CHANGE_SUMMARY.csv')
 photo_paired={r['candidate_method'] for r in paired if r['family']=='Gaussian' and r['baseline_method']=='E10' and r['metric']=='fg_mask_psnr' and int(r['actual_paired_views'])==208}
 required_paired={'P2','E11','E12','E13','E14','E14_refined'}
 check('Actual208 paired-view changes cover all terminal Gaussian routes with descriptive within-sequence statistics',
       required_paired<=photo_paired,OUT/'comparison/paired_view_changes/PAIRED_VIEW_CHANGE_SUMMARY.csv',sorted(required_paired-photo_paired))
 pending=[r for r in checks if not r['passed']]
 result=dict(status='actual_model_delivery_ready_pending_scientific_review' if not pending else 'incomplete',
  passed=len(checks)-len(pending),requirements=len(checks),pending=pending,checks=checks,
  artifact_entries=len(artifacts),unique_physical_files=len({r['absolute_path'] for r in artifacts}),
  interpretation='Passing is a real artifact/metric/visual availability gate, never an automatic scientific ranking or final completion claim. Historical input differences and absent scan truth remain explicit.')
 write_status(OUT/'artifact_index/FULL_DELIVERY_VALIDATION.json',result)
 print(result['status'],result['passed'],'/',len(checks),'requirements; remaining',len(pending),flush=True)
 if args.require_complete and pending:raise SystemExit(1)

if __name__=='__main__':main()
