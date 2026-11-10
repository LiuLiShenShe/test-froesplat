"""Actual full comparison/visual/model-index work after all dependent training stages.
This marker means review-ready, never an automatic final scientific selection.
"""
import os,json,csv,time,subprocess,hashlib,fcntl
from pathlib import Path
from queue_io import read_status, write_status
OUT=Path(__file__).resolve().parents[1];BASE=OUT.parents[2];OLD=OUT.parent/'geometry_mesh_final_selection';DUAL=OUT.parent/'geometry_rgb_dual_frontend';PY='/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python';GGGS=str(BASE/'.venv/bin/python');SUGAR=str(OUT/'E14_2D_SuGaR/.venv/bin/python');QGS=str(OUT/'E13_QGS/.venv/bin/python')
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MPLBACKEND='Agg')
_lock=(OUT/'finalization_execution.lock').open('a')
fcntl.flock(_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(x):write_status(OUT/'FINALIZATION_QUEUE_STATUS.json',x)
def wait_status(p,s):
    while True:
        upstream=read_status(p)
        if upstream.get('status')==s:return
        if upstream.get('status') in ['failed','partial','failed_dependency']:
            status(dict(status='failed_dependency',dependency=str(p),upstream=upstream,pid=os.getpid()))
            raise RuntimeError(f'Upstream execution needs recovery: {p}')
        status(dict(status='waiting_for_dependency',dependency=str(p),required_status=s,pid=os.getpid()));time.sleep(5)
def run(name,args,required=None,py=PY):
    if required and required.is_file():return
    cmd=['uv','run','--no-project','--python',str(py),'python',*map(str,args)];status(dict(status='running',stage=name,pid=os.getpid(),command=cmd));start=time.time()
    with (OUT/'logs'/f'finalization_{name}.log').open('a') as f:
        child=subprocess.Popen(cmd,cwd=OUT,stdout=f,stderr=subprocess.STDOUT);(OUT/'FINALIZATION_CHILD_PROCESS.json').write_text(json.dumps(dict(stage=name,pid=child.pid),indent=2));rc=child.wait()
    with (OUT/'actual_commands.jsonl').open('a') as f:f.write(json.dumps(dict(stage='finalization_'+name,command=cmd,returncode=rc,wall_seconds=time.time()-start))+'\n')
    if rc or (required and not required.is_file()):status(dict(status='failed',stage=name,returncode=rc));raise SystemExit(rc or 1)
    print(name,'complete',flush=True)
wait_status(OUT/'HISTORY_QUEUE_STATUS.json','complete')
wait_status(OUT/'CPU_QUEUE_STATUS.json','new_mesh_queue_complete')
wait_status(OUT/'AVAILABLE_MESH_VISUAL_STATUS.json','complete_available')
wait_status(OUT/'ANALYSIS_QUEUE_STATUS.json','complete')
wait_status(OUT/'REFLECTION_MESH_QUEUE_STATUS.json','complete')
run('native_QGS_loss_branch_audit',[OUT/'E13_QGS/scripts/audit_native_loss_log.py'],OUT/'E13_QGS/native_loss_log_audit.json',py=QGS)
run('native_QGS_stability_plot',[OUT/'E13_QGS/scripts/build_stability_panels.py'],py=QGS)
run('native_QGS_terminal_normal_objective',[OUT/'E13_QGS/scripts/decompose_terminal_normal_objective.py'],OUT/'E13_QGS/native_normal_objective_decomposition/complete.json',py=QGS)
qgs=OUT/'E13_QGS/meshes/native_TSDF'
run('E13_mesh208',[OUT/'scripts/evaluate_mesh.py','--mesh-dir',qgs,'--input',OUT/'E13_QGS/surface_inputs','--method','E13_QGS','--stages','raw','official_top10_post','post'],qgs/'unified_evaluation/complete.json')
run('all_cached_mesh_proxies',[OUT/'scripts/recompute_history_mesh.py','--include-new'],OUT/'comparison/history_mesh_uniform/E14_refined/complete.json')
run('P4_stage_decomposition',[OUT/'scripts/p4_stage_decomposition.py'],OUT/'comparison/P4_DECOMPOSITION.json')
run('P4_post_depth_changes',[OUT/'scripts/audit_p4_post_depth.py'],OUT/'comparison/P4_post_depth/complete.json')
for name,folder,ply in [('P1',OUT/'P1_GGGS_Tetrahedra',DUAL/'E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply'),('P2',OUT/'P2_GGGS_Cleaning/Tetrahedra',OUT/'P2_GGGS_Cleaning/cleaned_gaussians.ply'),('E12',OUT/'E12_GGGS_EO/Tetrahedra',OUT/'E12_GGGS_EO/checkpoints/point_cloud/iteration_30000/point_cloud.ply')]:
    dest=folder/'recon_post_vertex_color.ply'
    run(name+'_tetra_color',[OUT/'scripts/color_tetra.py','--mesh',folder/'recon_post.ply','--ply',ply,'--out',dest],dest.with_suffix('.colour.json'),py=GGGS)
inspection=OUT/'comparison/shared_inspection_views.json'
for name,ply in [('E10',DUAL/'E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply'),('P2',OUT/'P2_GGGS_Cleaning/cleaned_gaussians.ply'),('E12',OUT/'E12_GGGS_EO/checkpoints/point_cloud/iteration_30000/point_cloud.ply')]:
    dest=OUT/'comparison/visualizations'/f'{name}_top_oblique'
    run(name+'_top',[OUT/'scripts/render_gggs_free.py','--ply',ply,'--out',dest,'--protocol',inspection],dest/'complete.json',py=GGGS)
for name,protocol in [('free',OUT/'comparison/shared_free_views.json'),('top',inspection)]:
    dest=OUT/'comparison/visualizations'/f'E11_{name}'
    run('E11_'+name,[OUT/'E14_2D_SuGaR/scripts/render_e11_shared_free.py','--out',dest,'--protocol',protocol],dest/'complete.json')
dest=OUT/'comparison/visualizations/E13_top_oblique'
run('E13_top',[OUT/'E13_QGS/scripts/render_qgs_free.py','--protocol',inspection,'--out',dest],dest/'complete.json',py=QGS)
for method in ['surfel','refined']:
    dest=OUT/'comparison/visualizations'/f'E14_{method}_top_oblique'
    run('E14_'+method+'_top',[OUT/'E14_2D_SuGaR/scripts/render_visual_views.py','--method',method,'--protocol',inspection,'--out',dest],dest/'complete.json',py=SUGAR)
run('actual_inventory',[OUT/'scripts/build_inventory.py'])
artifacts=list(csv.DictReader((OUT/'artifact_index/ALL_ARTIFACTS.csv').open()));seen=set();supplements=[]
for a in artifacts:
    if a['artifact_type'] not in ['Mesh PLY','Mesh OBJ','Textured OBJ']:continue
    if a['status'].startswith(('exists_empty_failed','exists_archived_UV_seam')):continue
    # These incidental old meshes predate the frozen corrected camera gauge.
    # Keep their actual files indexed, without rendering them through E10 K/W2C.
    if a['experiment_id'] in ['E1','E2','E3','E4','E5','E6']:continue
    mesh=Path(a['absolute_path']);key=str(mesh.resolve())
    if key in seen:continue
    seen.add(key);eid=a['experiment_id'];alias=eid+'_'+hashlib.sha256(key.encode()).hexdigest()[:12]
    inp=OLD/'surface_inputs/e8_gggs_surface' if eid=='M1' else OLD/'surface_inputs/e10_gggs_surface'
    if eid in ['P4','M5']:inp=OLD/'surface_inputs/e11_2dgs_surface'
    if eid=='P2':inp=OUT/'P2_GGGS_Cleaning/surface_inputs'
    if eid=='E12':inp=OUT/'E12_GGGS_EO/surface_inputs'
    if eid=='E13':inp=OUT/'E13_QGS/surface_inputs'
    if eid=='E14':inp=OUT/'E14_2D_SuGaR/stage_renders/final_30000'
    dest=OUT/'comparison/visualizations/mesh'/alias
    run(alias+'_views',[OUT/'E14_2D_SuGaR/scripts/render_mesh_geometry_visuals.py','--mesh',mesh,'--input',inp,'--out',dest,'--normal','camera'],dest/'complete.json')
    if '_vertex_color.ply' not in mesh.name:
        dest=OUT/'comparison/surface_supplement'/alias
        args=[OUT/'P3_GS2Mesh/scripts/mesh_surface_supplement.py','--mesh',mesh,'--out',dest,'--frontend','omega' if eid=='M1' else 'standard','--mode','all','--support-cap','200000','--queries','3000','--seed','0','--workers','4']
        # The freshly generated actual8 cache uses a different format; boundary
        # projection support is taken from the original canonical208 cache.
        cache=mesh.parent/'unified_evaluation/views'
        if not cache.is_dir():cache=mesh.parent/'evaluation/views'
        stage={'recon_init.ply':'raw','recon.ply':'binary','recon_post.ply':'post'}.get(mesh.name,mesh.stem)
        camera=inp/'camera_runtime.json'
        if cache.is_dir() and camera.is_file() and all((cache/f'{stage}_{sid:04d}.npz').is_file() for sid in [0,26,53,79,105,131,157,183]):args+=['--camera-index',camera,'--evaluation-dir',cache.parent,'--stage',stage]
        run(alias+'_geometry',args,dest/'supplement_complete.json')
        supplements.append(dict(experiment_id=eid,mesh=str(mesh.resolve()),stage=a['stage'],summary=str((dest/'supplement_complete.json').resolve()),roughness_csv=str((dest/'fixed_scale_roughness.csv').resolve()),boundary_json=str((dest/'boundary_complete.json').resolve()),visualization=str((OUT/'comparison/visualizations/mesh'/alias/'complete.json').resolve())))
(OUT/'comparison/SURFACE_SUPPLEMENT_INDEX.json').write_text(json.dumps(supplements,ensure_ascii=False,indent=2))
run('Gaussian_visual_panels',[OUT/'scripts/build_gaussian_panels.py'])
run('native_normal_sampling_v2',[OUT/'scripts/recompute_reflection_normals_v2.py'])
run('actual_original_RGB_candidate_audit',[OUT/'scripts/audit_original_RGB_correspondences.py'])
run('actual_original_RGB_Mesh_candidate_summary',[OUT/'scripts/summarize_original_RGB_mesh_correspondences.py'])
run('actual_native_shape_and_same_query_Mesh_audit',[OUT/'scripts/audit_native_primitive_shapes.py'])
run('fixed_GT_highlight_native_panels',[OUT/'scripts/build_highlight_panels.py'])
run('fixed_RGB_query_missingness',[OUT/'scripts/audit_reflection_missingness.py'])
run('P4_identical_window_stage_visuals',[OUT/'scripts/build_p4_stage_visuals.py'],OUT/'comparison/visualizations/P4_stage_visuals/PROTOCOL.json')
run('P4_local_dark_foreground_spatial',[OUT/'scripts/audit_p4_dark_foreground.py'],OUT/'comparison/P4_local_failure/complete.json')
run('final_inventory',[OUT/'scripts/build_inventory.py'])
run('actual_native_PLY_and_checkpoint_structure',[OUT/'scripts/audit_native_artifact_structure.py'])
run('pinned_sources_and_script_manifest',[OUT/'scripts/write_source_manifest.py'])
run('actual_comparison_tables',[OUT/'scripts/aggregate_comparison.py'])
run('common_adjacent_normal_roughness',[OUT/'scripts/complete_adjacent_normal_roughness.py'])
run('comparison_with_all_adjacent_normal_metrics',[OUT/'scripts/aggregate_comparison.py'])
run('actual_same_source_ray_Mesh_layers',[OUT/'scripts/measure_same_ray_mesh_layers.py'])
run('actual_same_ray_front_back_and_joint_spacing',[OUT/'scripts/summarize_same_ray_pair_spacing.py'])
run('comparison_with_actual_same_ray_layers',[OUT/'scripts/aggregate_comparison.py'])
run('actual_paired208_view_change_distributions',[OUT/'scripts/summarize_paired_view_changes.py'])
run('leaf_region_mesh208',[OUT/'scripts/leaf_region_mesh_metrics.py'])
run('leaf_visible_PCA_surface208',[OUT/'scripts/leaf_visible_surface_roughness.py'])
run('actual_Mesh_panels',[OUT/'scripts/build_mesh_panels.py'])
run('actual_surface_and_originalRGB_audit',[OUT/'scripts/write_methodology_audit.py'])
run('actual_visual_comparison_index',[OUT/'scripts/write_visual_comparison_index.py'])
run('measured_analysis_reports',[OUT/'scripts/write_current_reports.py'])
run('final_current_source_and_script_manifest',[OUT/'scripts/write_source_manifest.py'])
run('real_delivery_validation',[OUT/'scripts/validate_full_delivery.py','--require-complete'])
status(dict(status='actual_models_metrics_visuals_indices_ready_pending_scientific_review',pid=os.getpid(),mesh_files_visualized=len(seen),supplemented_mesh_files=len(supplements)))
