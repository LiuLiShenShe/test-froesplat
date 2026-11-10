"""Gather actual completed208 metrics, retain legacy metrics/conditions and native stages."""
import csv,json,math,argparse
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection';DUAL=OUT.parent/'geometry_rgb_dual_frontend';COMP=OUT/'comparison'
parser=argparse.ArgumentParser()
parser.add_argument('--available-only',action='store_true',help='During execution, retain incomplete experiments as pending and aggregate only complete208 groups; final delivery uses the strict default')
args=parser.parse_args();pending_sources=[]
def read(p):return list(csv.DictReader(p.open())) if p.is_file() else []
def write(p,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def num(x):
    try:return float(x)
    except (TypeError,ValueError):return float('nan')
def avg(rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));result={}
    for k in keys:
        vals=[num(r.get(k)) for r in rows];finite=[x for x in vals if math.isfinite(x)]
        if finite:result[k]=float(np.mean(finite))
    return result
artifacts=read(OUT/'artifact_index/ALL_ARTIFACTS.csv');g=read(COMP/'ALL_GAUSSIAN_EXPERIMENTS.csv');legacy=[dict(r,metric_version='preserved_historical') for r in g]
if not (COMP/'LEGACY_GAUSSIAN_METRICS.csv').is_file():
    write(COMP/'LEGACY_GAUSSIAN_METRICS.csv',legacy)
sources={}
for e in ['E1','E5','E7','E8','E9','E10']:sources[e]=DUAL/'render_comparison'/e
sources['E11']=OLD/'surface_inputs/e11_2dgs_surface'
# E11 numerical metrics were calculated on exactly these native surface outputs.
for e in ['E2','E3','E4','E6']:sources[e]=COMP/'history_uniform'/e
sources.update(P2=OUT/'P2_GGGS_Cleaning/surface_inputs',E12=OUT/'E12_GGGS_EO/surface_inputs',E13=OUT/'E13_QGS',E14=OUT/'E14_2D_SuGaR/stage_renders/final_30000',E14_refined=OUT/'E14_2D_SuGaR/stage_renders/refined')
for stage in ['initial','cluster_before','cluster_after']:
    sources['E14_'+stage]=OUT/'E14_2D_SuGaR/stage_renders'/stage
renderrows=[];summaries={}
for e,root in sources.items():
    p=root/'per_view_metrics.csv'
    if e=='E11':p=OLD/'E11_2DGS_Corrected/per_view_metrics.csv'
    rr=read(p)
    if e=='E11' and not rr:
        # History report explicitly contains the trueFG208 average and scope.
        old=next((r for r in read(OLD/'comparisons/rendering_comparison.csv') if r['experiment']=='E11'),{})
        summaries[e]={'raw':old};continue
    if not rr:continue
    stages=set(r.get('stage','raw') for r in rr)
    if args.available_only and any(sum(r.get('stage','raw')==s for r in rr)!=208 for s in stages):
        pending_sources.append(dict(experiment_id=e,metric_source=str(p.resolve()),stage_rows={s:sum(r.get('stage','raw')==s for r in rr) for s in sorted(stages)}))
        continue
    summaries[e]={}
    for s in sorted(stages):
        group=[r for r in rr if r.get('stage','raw')==s]
        assert len(group)==208,(e,s,len(group))
        summaries[e][s]=avg(group)
        for r in group:renderrows.append(dict(r,experiment=e,stage=s,metric_source=str(p.resolve()),views=208))
write(COMP/'rendering_metrics.csv',renderrows)
byid={r['experiment_id']:r for r in g}
for e,stages in summaries.items():
    if not stages:continue
    r=byid.get(e)
    if r is None:
        baseline=byid['E10' if e=='P2' else 'E14'];r=dict(baseline,experiment_id=e,model_stage='cleaned30K' if e=='P2' else 'SuGaR_refined15K_after2D30K' if e=='E14_refined' else e.removeprefix('E14_'));g.append(r);byid[e]=r
    raw=stages.get('raw') or stages.get('final_30000') or stages.get('refined') or next(iter(stages.values()));corr=stages.get('corrected',{})
    r.update(psnr_raw=raw.get('full_psnr',''),psnr_corrected=corr.get('full_psnr',''),fg_psnr=raw.get('fg_mask_psnr',''),fg_psnr_raw=raw.get('fg_mask_psnr',''),fg_psnr_corrected=corr.get('fg_mask_psnr',''),ssim=raw.get('ssim',''),ssim_corrected=corr.get('ssim',''),lpips=raw.get('lpips_alex',''),lpips_corrected=corr.get('lpips_alex',''),gradient_preservation=raw.get('fg_gradient_retention',''),background_alpha=raw.get('background_alpha_mean',''),foreground_alpha=raw.get('foreground_alpha_mean',''),background_alpha_gt05_fraction=raw.get('background_alpha_gt05_fraction',''),evaluation_scope='208 actual frozen RGB/true RAP FG; native1080x1920 float clipped for metrics; historical K/background conditions retained per row',status='model_and208_metrics_verified',metric_source=str(sources[e].resolve()))
    r.update(fg_ssim_map_raw=raw.get('fg_mask_ssim_map',''),fg_ssim_map_corrected=corr.get('fg_mask_ssim_map',''),
        gradient_preservation_corrected=corr.get('fg_gradient_retention',''),
        fg_ssim_scope='Same full-image SSIMwin7 map averaged in actual RAP foreground, excluding image3px rim; not an independent foreground-only LPIPS measurement',
        evaluation_RGB_source_manifest=str((DUAL/'shared_manifest.csv').resolve()),
        evaluation_background='Actual saved03-SAM masked training RGB; background black except boundary-resize interpolation. Unmasked02-FFT context is separate from this photometric GT.')
    if e=='E13' or e.startswith('E14'):
        camera_file=OUT/'E13_QGS/surface_inputs/camera_runtime.json' if e=='E13' else OUT/'E14_2D_SuGaR/runtime_camera_validation.json'
        cameras=json.loads(camera_file.read_text())['records']
        assert len(cameras)==208 and len({c['name'] for c in cameras})==208
        r.update(training_views=208,saved_camera_records=len(cameras),training_camera_evidence=str(camera_file.resolve()))
    if e=='P2':
        cleaning=json.loads((OUT/'P2_GGGS_Cleaning/cleaning_complete.json').read_text())
        r.update(gaussian_ply_path=str((OUT/'P2_GGGS_Cleaning/cleaned_gaussians.ply').resolve()),num_gaussians=cleaning['cleaned_gaussians'],iterations=30000,training_minutes=0,postprocess_minutes=cleaning['wall_seconds']/60)
    if e in ['E14_initial','E14_cluster_before','E14_cluster_after']:
        root=OUT/'E14_2D_SuGaR';iteration=1 if e=='E14_initial' else 7000
        path=root/('cluster_pruning/after.ply' if e=='E14_cluster_after' else f'training/point_cloud/iteration_{iteration}/point_cloud.ply')
        primitive=next(a for a in artifacts if a['experiment_id']=='E14' and a['absolute_path']==str(path.resolve()) and a['artifact_type']=='Gaussian PLY')
        checkpoint=root/'cluster_pruning'/('after_pruning_checkpoint.pth' if e=='E14_cluster_after' else 'before_pruning_checkpoint.pth')
        clock=read(root/'training_metrics.csv');prefix=next(row for row in clock if int(row['iteration'])==iteration)
        r.update(gaussian_ply_path=str(path.resolve()),checkpoint_path=str(checkpoint.resolve()) if iteration==7000 and checkpoint.is_file() else '',
            num_gaussians=primitive['vertices_or_gaussians'],iterations=iteration,additional_refinement_iterations=0,
            gaussian_method='2D-SuGaR native2DGS',model_stage=e.removeprefix('E14_'),
            native_representation='Native two-scale surfel; unchanged parameters across DBSCAN deletion-only before/after',
            extra_prior='Same E14 Metric3Dv2 depth/normal and depth-guided initialization; native losses',
            training_minutes=float(prefix['elapsed_seconds'])/60,training_time_scope='Reused native E14 prefix, not an independent retraining; repeated stage times must not be summed',
            peak_vram_gb='',peak_vram_scope='Prefix peak was not independently measured; terminal30K peak remains separate',
            metric_source=str((sources[e]/'per_view_metrics.csv').resolve()))
    if e in ['E12','E13','E14','E14_refined']:
        candidates=[a for a in artifacts if a['experiment_id']==e.split('_')[0] and a['format']=='ply' and a['artifact_type'] in ['Gaussian PLY','Gaussian Model']]
        if e=='E13':candidates=[a for a in candidates if Path(a['absolute_path']).is_relative_to((OUT/'E13_QGS/checkpoints').resolve())]
        if e=='E14_refined':
            info=json.loads((OUT/'E14_2D_SuGaR/refinement_complete.json').read_text());r.update(gaussian_ply_path=str(Path(info['gaussian_ply']).resolve()),checkpoint_path=str(Path(info['native_model']).resolve()),iterations=15000,base_iterations=30000,additional_refinement_iterations=15000,gaussian_method='SuGaR mesh-bound refined3D Gaussian',model_stage='refined',extra_prior='Metric3Dv2 giant2 depth/normal + native initialization and DBSCAN')
        else:
            final=next((a for a in candidates if 'iteration_30000' in a['absolute_path']),None)
            if final:r.update(gaussian_ply_path=final['absolute_path'],num_gaussians=final['vertices_or_gaussians'],iterations=30000)
        if e=='E12':
            t=json.loads((OUT/'E12_GGGS_EO/densification_events.complete.json').read_text());r.update(training_minutes=t['wall_seconds']/60,peak_vram_gb=t['peak_allocated_gib'],photometric_objective='.8L1(EO(render),GT)+.2[1-SSIM(EO(render),GT)] + unchangedE10 geometry regularizers',exposure_parameters_path=str((OUT/'E12_GGGS_EO/exposure_params/exposure_matrices.npz').resolve()))
        if e=='E13':r.update(native_representation='Quadratic six signed-scale primitive; curvature is not thickness',extra_prior='same frozen VGGT initialization; no added depth/normal prior')
        if e=='E14':r.update(extra_prior='Metric3Dv2 giant2 depth/normal; depth-guided initialization; official depth_ratio1/lambda_dist1000/priornormal.1; DBSCAN7K',additional_refinement_iterations=0,model_stage='native2D30K before refinement')
        if e in ['E13','E14']:
            root=OUT/('E13_QGS' if e=='E13' else 'E14_2D_SuGaR')
            t=json.loads((root/'training_complete.json').read_text())
            r.update(training_minutes=t['training_wall_seconds']/60,peak_vram_gb=t['peak_allocated_gib'])
            if e=='E13':
                r.update(training_time_scope=t.get('training_wall_definition','Actual native training'),
                    successful_model_training_minutes=float(t.get('successful_model_training_wall_seconds',t['training_wall_seconds']))/60,
                    resumed_segment_training_minutes=float(t.get('successful_segment_wall_seconds',0))/60,
                    discarded_attempt_training_minutes=float(t.get('discarded_initial_attempt_wall_seconds',0))/60,
                    resumed_from_iteration=t.get('resume_iteration',0),
                    resumed_seed_trajectory='Native seed0 reset; original RNG/viewpoint stack not saved' if t.get('resume_checkpoint') else 'Fresh native seed0')
            if e=='E14':
                prior=json.loads((root/'priors_complete.json').read_text())
                r.update(prior_generation_minutes=prior['elapsed_seconds']/60,prior_peak_vram_gib=prior['peak_allocated_gib'],prior_weight_sha256=prior['weight_sha256'])
        if e=='E14_refined':
            r.update(training_minutes=info['elapsed_seconds']/60,peak_vram_gb=info['peak_allocated_gib'],training_time_scope='Native SuGaR refinement plus Gaussian/Mesh/texture export')
            r['peak_vram_scope']=info.get('peak_vram_scope','Native SuGaR refinement and export')
            if info.get('reused_native15K_checkpoint'):
                r['export_peak_vram_gib']=info['peak_allocated_gib']
                r['peak_vram_gb']=''
            native=next((a for a in candidates if a['absolute_path']==r['gaussian_ply_path']),None)
            if native:r['num_gaussians']=native['vertices_or_gaussians']
qgs_row=byid.get('E13',{})
for shape in read(COMP/'native_primitive_shapes/ALL_NATIVE_SHAPE_SUMMARY.csv'):
    experiment={'E14_30K':'E14','E14_refined_aux':'E14_refined'}.get(shape['method'],shape['method'])
    if shape['group']!='original_RGB_brightness' or experiment not in byid:continue
    row=byid[experiment]
    for key in ['queries','associated_finite_shape_queries','unique_associated_native_primitives','native_scale_fields',
        'tangent_or_needle_aspect_median','tangent_or_needle_aspect_p90','threeD_flattening_second_over_smallest_median',
        'max_characteristic_span_over_D_median','aspect_above_own_P99_fraction_associated']:
        row['original_RGB_brightness_native_shape_'+key]=shape[key]
    row['original_RGB_brightness_native_shape_scope']=shape['shape_definition']+' Previously recorded nearest center, not renderer contribution; query counts/unique IDs retained, own-model P99 is descriptive.'
    row['original_RGB_brightness_native_shape_source']=str((COMP/'native_primitive_shapes/ALL_NATIVE_SHAPE_SUMMARY.csv').resolve())
qgs_root=OUT/'E13_QGS'
request=qgs_root/'NATIVE_PRESET_REQUEST.json'
if request.is_file():qgs_row['native_preset']=json.loads(request.read_text())['native_preset']
archive=qgs_root/'runs/base_expandable_oom_13301/training_execution_records.json'
if archive.is_file():
    prior_runs=json.loads(archive.read_text())
    qgs_row['archived_base_attempt_training_minutes']=sum(float(r.get('duration_seconds',0)) for r in prior_runs)/60
    qgs_row['archived_base_attempt_cost_source']=str(archive.resolve())
qgs_row['native_parameter_quality']='not_yet_audited_for_selected_model'
for audit_name in ['partial_checkpoint_provenance_validation.json','checkpoint_provenance_validation.json']:
    audit_path=qgs_root/audit_name
    if not audit_path.is_file():continue
    for record in json.loads(audit_path.read_text()).get('records',[]):
        if record['native_ply_path']==qgs_row.get('gaussian_ply_path'):
            qgs_row.update(native_nonfinite_primitives=record['native_nonfinite_primitives'],
                native_nonfinite_fraction=record['native_nonfinite_fraction'],
                native_parameter_quality='nonfinite_native_parameters_retained' if record['native_nonfinite_primitives'] else 'selected_model_all_native_parameters_finite',
                native_parameter_audit_path=str(audit_path.resolve()),native_all_fields_bitwise_match_checkpoint=record['all_native_fields_bitwise_match_checkpoint'])
stability=qgs_root/'native_stability/PROTOCOL.json'
if stability.is_file():
    observed=json.loads(stability.read_text()).get('checkpoint_audits',[])
    if observed:
        latest=max(observed,key=lambda a:a['iteration'])
        qgs_row.update(latest_observed_native_iteration=latest['iteration'],
            latest_observed_native_nonfinite_primitives=latest['nonfinite_primitives'],
            latest_observed_native_nonfinite_fraction=latest['nonfinite_fraction'],
            latest_observed_native_audit_path=latest['audit_path'],
            latest_observed_native_scope='Observed saved state only; distinct from selected model and terminal208 metrics. Rolling paths can have changed since observation.')
for region in read(COMP/'EO_region_decomposition/summary.csv'):
    target=byid.get(region['method'])
    if target is None:continue
    stage=region['stage']
    target['background_ssim_'+stage+'_derived']=region['derived_background_ssim']
    target['background_psnr_'+stage+'_derived']=region['derived_background_psnr']
    target['background_photometric_partition_source']=str((COMP/'EO_region_decomposition/PROTOCOL.json').resolve())
    target['background_photometric_partition_scope']='Derived from saved full/FGwin7 scores and actual mask counts, mean208 views; native geometry unchanged.'
write(COMP/'ALL_GAUSSIAN_EXPERIMENTS.csv',g)
routes=[]
for folder in sorted((OLD/'meshes').glob('M*')):
    if folder.is_dir():routes.append((folder.name,folder,COMP/'history_mesh_uniform'/folder.name,'VGGT-Omega' if folder.name.startswith('M1') else 'standardVGGT' if folder.name.startswith('M2') else 'E11' if folder.name.startswith('M5') else 'E10','NKSR' if 'NKSR' in folder.name else 'TSDF'))
routes += [('P1',OUT/'P1_GGGS_Tetrahedra',OUT/'P1_GGGS_Tetrahedra/unified_evaluation','E10','Marching Tetrahedra + binary10')]
for name,method in [('TSDF','TSDF'),('Tetrahedra','Marching Tetrahedra + binary10')]:
    folder=OUT/'P2_GGGS_Cleaning'/name;routes.append(('P2_'+name,folder,folder/'unified_evaluation','P2_cleaned_E10',method))
for name,folder in [('P3_native',OUT/'P3_GS2Mesh/meshes'),('P3_shared',OUT/'P3_GS2Mesh/meshes_shared_depth_range')]:routes.append((name,folder,COMP/'history_mesh_uniform'/name,'E10','GS2Mesh DLNR + TSDF'))
for folder in (OUT/'P4_2DGS_Depth_Mesh').iterdir():
    if folder.is_dir() and (folder/'raw.ply').is_file():routes.append(('P4_'+folder.name,folder,folder/'unified_evaluation','E11','2DGS TSDF'))
for name,method in [('TSDF','TSDF'),('Tetrahedra','Marching Tetrahedra + binary10')]:
    folder=OUT/'E12_GGGS_EO'/name;routes.append(('E12_'+name,folder,folder/'unified_evaluation','E12',method))
folder=OUT/'E13_QGS/meshes/native_TSDF';routes.append(('E13_QGS',folder,folder/'unified_evaluation','E13','Official QGS bounded TSDF'))
for phase,directory in [('cluster_before','cluster_pruning/cluster_before_mesh'),('cluster_after','cluster_pruning/cluster_after_mesh'),('initial','initial_mesh'),('refined','refinement/mesh_post')]:
    folder=OUT/'E14_2D_SuGaR'/directory
    source='E14' if phase=='initial' else 'E14_'+phase
    routes.append(('E14_'+phase,folder,COMP/'history_mesh_uniform'/('E14_'+phase),source,'SuGaR refined mesh' if phase=='refined' else '2D-SuGaR nativeTSDF'))
meshrows=[];perview=[];surface_rows=[]
supplement_index=COMP/'SURFACE_SUPPLEMENT_INDEX.json'
supplements=json.loads(supplement_index.read_text()) if supplement_index.is_file() else []
supplements={r['mesh']:r for r in supplements}
normal_index=COMP/'ADJACENT_NORMAL_ROUGHNESS_INDEX.json'
normal_results={r['mesh']:r for r in json.loads(normal_index.read_text())} if normal_index.is_file() else {}
raw_reflection={r['actual_mesh_path']:r for r in read(COMP/'reflection_original_RGB_mesh_metrics.csv') if r['RGB_candidate_group']=='raw_context_candidates'}
layer_results={r['evaluated_mesh_path']:r for r in read(COMP/'same_ray_mesh_layers/ALL_MESH_LAYER_METRICS.csv') if r['group']=='all_fixed_queries'}
pair_results={r['evaluated_mesh_path']:r for r in read(COMP/'same_ray_mesh_layers/FIRST_PAIR_SPACING_METRICS.csv') if r['group']=='all_fixed_queries'}
for name,folder,evaluation,source,algorithm in routes:
    for path in sorted(evaluation.glob('*_per_view.csv')):
        if path.name=='all_references_per_view.csv':continue
        rr=read(path)
        if not rr:continue
        assert len(rr)==208,(name,path,len(rr));stage=rr[0]['stage'];m=avg(rr)
        structs=read(folder/'unified_evaluation/structure.csv') or read(folder/'evaluation/structure.csv')
        if not structs and name.startswith('M'):
            structs=[r for r in read(OLD/'comparisons/mesh_structure_comparison.csv') if r['method']==name]
        s=next((s for s in structs if s.get('stage')==stage),{})
        raw=folder/'raw.ply';post=folder/'post.ply'
        if name.startswith('P1') or 'Tetrahedra' in name:raw=folder/'recon_init.ply';post=folder/'recon_post.ply'
        item=dict(mesh_id=name,source_experiment=source,source_gaussian=byid.get(source,{}).get('gaussian_ply_path',''),mesh_algorithm=algorithm,depth_source='native surface output; source-specific actual depth ratio/priors in run records',mesh_stage=stage,**{k:s.get(k,'') for k in ['vertices','faces','components','boundary_edges','nonmanifold_edges']},silhouette_iou=m['silhouette_iou'],boundary_f1=m['boundary_fscore_2px'],boundary_chamfer_px=m['boundary_chamfer_px'],foreground_coverage=m['foreground_coverage'],background_geometry_ratio=m['background_geometry_fraction'],hole_area=m.get('hole_area_px',''),hole_area_unit='mean208 enclosed missing-projection pixels, not3Dphysical area',hole_area_foreground_ratio=m.get('hole_area_fg_ratio',''),hole_count_projected_mean=m.get('hole_count',''),open_edge_missing_area_px=m.get('open_edge_missing_area_px',''),surface_roughness=s.get('surface_roughness',''),surface_roughness_definition='adjacent face1-cos includes curvature; fixedscale PCA supplement separately reported',raw_mesh_path=str(raw.resolve()) if raw.is_file() else '',post_mesh_path=str(post.resolve()) if post.is_file() else '',textured_mesh_path='',metric_source=str(path.resolve()),evaluation_scope='same208 native integer-centre cameras/masks; no scan; trueRAP holes excluded and open leaf rim separate')
        if 'Tetrahedra' in algorithm:
            item['depth_source']='GGGS native transmittance0.5 / stochastic-solid SDF; tetra marching and binary10, not depth-map fusion'
        elif name.startswith(('M1_','M2_')):
            item['depth_source']='Frontend RAP-filtered PointMap and estimated oriented normals; no trained Gaussian depth'
        elif name.startswith(('M3_','M4_','P2_TSDF','E12_TSDF')):
            item['depth_source']='Native GGGS median camera-Z surface depth and normal; same frozen208 RAP constraints'
        elif name.startswith('M5_') or name.startswith('P4_expected'):
            item['depth_source']='Same E11 native2DGS alpha-normalized Expected Depth, depth_ratio0; RAP'+(' and alpha>=0.5' if '_alpha' in name else '')
        elif name.startswith('P4_median'):
            item['depth_source']='Same E11 native2DGS Median Depth, depth_ratio1 at extraction only; RAP'+(' and alpha>=0.5' if '_alpha' in name else '')
        elif name.startswith('P3_'):
            item['depth_source']='Official DLNR Middlebury on E10 GGGS generated stereo,10 recurrent iterations; Z=fx*baseline/disparity; LR3px filter; '+('shared maxZ1.5692457557' if name=='P3_shared' else 'official4-20 baseline range')
        elif name.startswith('E13'):
            actual_mesh=json.loads((folder/'complete.json').read_text())
            item['depth_source']=f"Native QGS Mesh render: SH0, kernel-999, stop_z_gradient=True, depth_ratio{actual_mesh['depth_ratio']}; RAP"
        elif name=='E14_refined':
            item['depth_source']='Mesh-bound SuGaR vertex refinement from native30K median-TSDF Mesh; no TSDF from auxiliary3D center-depth'
        elif name.startswith('E14_'):
            item['depth_source']='Native2D-SuGaR Median Depth, depth_ratio1; Metric3Dv2 depth/normal prior, native initialization and DBSCAN7K; RAP'
        item['evaluation_scope']='same208 source cameras/masks; no scan; RAP-mask voids excluded from missing-FG proxy, open leaf rim separate; mask voids not independently certified physical holes'
        item['background_geometry_ratio_definition']='Mean208 mesh-covered pixels outside RAP / all pixels outside RAP, same historical evaluator denominator'
        item['background_geometry_prediction_ratio']=1-float(m['foreground_precision']) if m.get('foreground_precision') is not None else ''
        item['runtime_scope']='One extraction run including its postprocessing, excluding training/quality evaluation; repeated across raw/post rows, do not sum stages'
        info=folder/'complete.json'
        if info.is_file():
            t=json.loads(info.read_text())
            seconds=next((t[k] for k in ['elapsed_seconds','wall_seconds','total_wall_seconds','extraction_seconds'] if k in t),None)
            item['extraction_minutes']=seconds/60 if seconds is not None else ''
        if name.startswith('P3_'):
            render=OUT/'P3_GS2Mesh/render_complete.json';stereo=OUT/'P3_GS2Mesh/stereo_complete.json'
            if render.is_file() and stereo.is_file() and info.is_file():
                render_info=json.loads(render.read_text());stereo_info=json.loads(stereo.read_text())
                item['fusion_minutes']=item['extraction_minutes']
                item['stereo_render_minutes']=render_info['wall_seconds']/60
                item['stereo_inference_minutes']=stereo_info['wall_seconds']/60
                item['extraction_minutes']=item['fusion_minutes']+item['stereo_render_minutes']+item['stereo_inference_minutes']
                item['stereo_peak_allocated_gib']=stereo_info['peak_allocated_gib']
                item['runtime_scope']='E10 stereo render + bidirectional DLNR + this fusion/post; same stereo reused by two depth-range routes; quality evaluation excluded, do not sum raw/post or shared stereo twice'
        supplement=COMP/'surface_supplement'/name/stage/'roughness_summary.csv'
        if supplement.is_file():item['fixed_scale_geometry_summary']=str(supplement.resolve())
        source_id='P2' if source=='P2_cleaned_E10' else source
        item['source_gaussian']=byid.get(source_id,{}).get('gaussian_ply_path','')
        if name=='E14_initial':item['source_stage']='Native2D30K initial Mesh BEFORE SuGaR refinement; distinct from E14_initial Gaussian iteration1'
        elif name in ['E14_cluster_before','E14_cluster_after']:item['source_stage']='Native7K '+('before' if name.endswith('before') else 'after')+' DBSCAN before optimizer.step; immediate matching primitive stage'
        color=folder/'recon_post_vertex_color.ply'
        if color.is_file():item['vertex_colored_mesh_path']=str(color.resolve())
        if name=='E14_refined':
            ref=OUT/'E14_2D_SuGaR/refinement_complete.json'
            if ref.is_file():
                obj=Path(json.loads(ref.read_text())['textured_obj'])
                if obj.is_file():item['textured_mesh_path']=str(obj.resolve())
        actual=folder/(stage+'.ply')
        if name=='P1' or 'Tetrahedra' in name:
            actual=folder/{'raw':'recon_init.ply','binary':'recon.ply','post':'recon_post.ply'}[stage]
        if actual.is_file():
            item['evaluated_mesh_path']=str(actual.resolve())
            layers=layer_results.get(str(actual.resolve()))
            if layers:
                for key,value in layers.items():
                    if key not in ['mesh_id','mesh_stage','evaluated_mesh_path','group','scope']:
                        item['same_ray_'+key]=value
                item['same_ray_metric_source']=str((COMP/'same_ray_mesh_layers/ALL_MESH_LAYER_METRICS.csv').resolve())
                item['same_ray_scope']='Supplementary fixed image-only matches from69 source cameras and195 source/target photos; primary silhouette/hole metrics remain208-view. Near layers can be true front/back, curled or occluded leaves; not certified erroneous double surfaces.'
            pairs=pair_results.get(str(actual.resolve()))
            if pairs:
                for key,value in pairs.items():
                    if key not in ['mesh_id','mesh_stage','evaluated_mesh_path','group','scope']:
                        item['same_ray_first_pair_'+key]=value
                item['same_ray_first_pair_scope']=pairs['scope']
                item['same_ray_first_pair_source']=str((COMP/'same_ray_mesh_layers/FIRST_PAIR_SPACING_METRICS.csv').resolve())
            reflected=raw_reflection.get(str(actual.resolve()))
            if reflected:
                for key in ['queries','mesh_hit_queries','mesh_missing_queries','mesh_hit_fraction','mesh_vs_native_relative_depth_median','mesh_vs_native_relative_depth_p90','mesh_local_normal_jump3px_deg_median','mesh_local_normal_jump3px_deg_p90']:
                    item['original_RGB_brightness_'+key]=reflected[key]
                item['original_RGB_brightness_geometry_source']=reflected['source_query_geometry']
                item['original_RGB_brightness_geometry_scope']='Frozen originalRGB brightness candidates, exact source-ray Mesh triangle normals. Mesh/native depth disagreement is not scan truth; sampled miss count is not208 projected hole area.'
            adjacent=normal_results.get(str(actual.resolve()))
            if adjacent:
                measured=json.loads(Path(adjacent['summary']).read_text())
                assert measured['file_size_bytes']==actual.stat().st_size and measured['faces']==int(item['faces'])
                if item['surface_roughness']!='':assert np.isclose(float(item['surface_roughness']),measured['surface_roughness'],atol=1e-12)
                item['surface_roughness']=measured['surface_roughness']
                item['adjacent_normal_roughness_source']=adjacent['summary']
            supplement=supplements.get(str(actual.resolve()))
            if supplement:
                item['fixed_scale_geometry_summary']=supplement['summary']
                for sr in read(Path(supplement['roughness_csv'])):
                    surface_rows.append(dict(sr,mesh_id=name,mesh_stage=stage,mesh_path=str(actual.resolve()),metric_source=supplement['roughness_csv']))
                for sr in read(Path(supplement['roughness_csv'])):
                    if sr['radius_divisor']=='64' and sr['neighborhood']=='same_oriented_normal':
                        item['fixed_scale_PCA_query_distance_normalized_median']=sr['query_plane_distance_normalized_median']
                        item['fixed_scale_PCA_valid_fraction']=sr['valid_fraction']
        meshrows.append(item);perview += [dict(r,mesh_id=name,source_experiment=source,metric_source=str(path.resolve())) for r in rr]
write(COMP/'ALL_MESH_EXPERIMENTS.csv',meshrows);write(COMP/'mesh_metrics.csv',perview)
if surface_rows:write(COMP/'surface_geometry_metrics.csv',surface_rows)
(COMP/'AGGREGATION_STATUS.json').write_text(json.dumps(dict(status='actual_available_metrics',available_only=args.available_only,pending_partial_sources=pending_sources,gaussian_rows=len(g),mesh_stage_rows=len(meshrows),rendering_perview_rows=len(renderrows),mesh_perview_rows=len(perview),all_paths_from_existing_files=True,legacy_preserved=True),indent=2))
print('Aggregated',len(g),'Gaussian rows',len(meshrows),'Mesh stages',flush=True)
