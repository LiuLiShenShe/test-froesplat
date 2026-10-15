"""Assemble measured results without ranking heterogeneous proxy families."""
from common import *
import statistics

FULL=['PSNR','SSIM','LPIPS']
FG=['foreground_MSE','foreground_PSNR','fixed_mask_composited_SSIM','fixed_mask_composited_LPIPS','foreground_absolute_color_error','foreground_coverage_proxy','missing_alpha_fraction']
BG=['outside_ROI_RGB_energy','outside_ROI_alpha_mean','outside_ROI_alpha_sum','proxy_silhouette_IoU','outside_alpha_components_ge16px','outside_alpha_largest_component_pixels','outside_alpha_pixels','depth_valid_ROI_fraction','normal_consistency_cosine_proxy','normal_consistency_abs_cosine_proxy','depth_p01','depth_p05','depth_p50','depth_p95','depth_p99']
MODEL=['Gaussian_count','opacity_p01','opacity_p50','opacity_p95','opacity_p99','scale_p01','scale_p50','scale_p95','scale_p99']
MESH=['mesh_ROI_coverage_proxy','mesh_proxy_silhouette_IoU','mesh_Gaussian_depth_MAE','mesh_Gaussian_depth_median_absolute_difference','mesh_Gaussian_depth_p95_absolute_difference','mesh_Gaussian_relative_depth_MAE']

def mean_metric(values,key):
    entries=[float(r[key]) for r in values if r.get(key) not in [None,'','None']]
    assert entries,key
    return statistics.mean(entries)

def main():
    summaries=[];perviews=[];foreground=[];background=[];meshes=[];meshcomparison=[];mesh_runs=[];training=[];events=[];capacity=[]
    legacy=rows(E2/'MULTISEED_TRAINING_MANIFEST.csv')
    executions=read_json(STAGE/'reports/TRAINING_EXECUTION.json')
    for sample in SCENES:
        reference_group=reference(sample)['actual_group']
        old=next(r for r in legacy if r['sample']==sample and r['group']==reference_group and r['seed']=='0')
        for method in ['M0','M1','M2']:
            label=f'{sample}_{method}'
            full=rows(STAGE/f'metrics/{label}_FULL.csv');fg=rows(STAGE/f'metrics/{label}_FOREGROUND.csv');bg=rows(STAGE/f'metrics/{label}_BACKGROUND_GEOMETRY.csv')
            assert len(full)==len(fg)==len(bg)
            assert {r['loader_image_name'] for r in full}=={r['loader_image_name'] for r in fg}=={r['loader_image_name'] for r in bg}
            perviews+=full;foreground+=fg;background+=bg
            summary={'sample':sample,'method':method,'seed':0,'training_repeats':1,'heldout_views':len(full),
                     'aggregation':'Arithmetic mean of frozen held-out views; views are not independent training replicates',
                     'independent_3D_GT':'NONE','ROI_is_independent_GT':False}
            for values,keys in [(full,FULL),(fg,FG),(bg,BG)]:
                for key in keys:summary[key]=mean_metric(values,key)
            for key in MODEL:
                distinct={r[key] for r in bg};assert len(distinct)==1,(label,key)
                summary[key]=float(next(iter(distinct)))
            if method=='M0':
                run={**old,'method':method,'new_training':False,'status':'INHERITED_COMPLETED',
                     'baseline_protocol_evidence':str(STAGE/'reports/BASELINE_EQUIVALENCE.json'),
                     'initial_RNG_limit':'Historical initial state hash absent; source path equivalent, no archived RNG identity claim'}
                summary.update(training_wall_seconds=float(old['training_wall_seconds']),peak_VRAM_allocated_bytes=float(old['peak_VRAM_allocated_bytes']))
            else:
                observed=read_json(STAGE/f'reports/{label}_TRAIN_RUN.json');assert observed['status']=='PASS' and observed['completed_iterations']==30000
                executed=next(r for r in executions if r['sample']==sample and r['method']==method)
                output=model_path(sample,method)
                run={**observed,'new_training':True,'iterations':observed['completed_iterations'],'process_execution':json.dumps(executed,ensure_ascii=False),
                     'checkpoint_path':str(output/'chkpnt30000.pth'),'PLY_path':str(output/'point_cloud/iteration_30000/point_cloud.ply'),
                     'environment_path':str(STAGE/'reports/ENVIRONMENT.json'),'environment_sha256':sha(STAGE/'reports/ENVIRONMENT.json'),
                     'initialization_path':str(STAGE/f'reports/{label}_INITIALIZATION.json'),'initialization_sha256':sha(STAGE/f'reports/{label}_INITIALIZATION.json'),
                     'initial_rng_json':json.dumps(read_json(STAGE/f'reports/{label}_INITIALIZATION.json')['initial_rng'],sort_keys=True),
                     'native_optimizer_updates':observed['optimizer_steps']}
                cap=read_json(output/'capacity_control/capacity_summary.json')
                capacity.append({'sample':sample,'method':method,**cap,'source_path':str(output/'capacity_control/capacity_summary.json'),'sha256':sha(output/'capacity_control/capacity_summary.json')})
                run.update(capacity_summary_path=str(output/'capacity_control/capacity_summary.json'),capacity_summary_sha256=sha(output/'capacity_control/capacity_summary.json'))
                own_events=rows(STAGE/f'metrics/{label}_PRUNING.csv')
                for event in own_events:
                    event['reference_max_seen_count']=cap['max_seen_count']
                    event['reference_count_evidence']='Native post-standard-pruning per-iteration maximum; densification ended before all extra rounds'
                events+=own_events
                summary.update(training_wall_seconds=observed['training_wall_seconds'],peak_VRAM_allocated_bytes=observed['peak_cuda_allocated_bytes'],
                               extra_pruning_removed_total=sum(int(r['removed_count']) for r in own_events),
                               floor_blocked_total=sum(int(r['floor_blocked_count']) for r in own_events),
                               floor_evaluated_events=sum(r['floor_evaluated']=='True' for r in own_events))
                ev=read_json(STAGE/f'reports/{label}_EVALUATION.json')
                summary.update(inference_render_mean_seconds=ev['inference_render_mean_seconds'],inference_render_median_seconds=ev['inference_median_seconds'])
            training.append(run);summaries.append(summary)
            mesh_report=read_json(STAGE/f'reports/{label}_MESH_RUN.json');assert mesh_report['status']=='PASS'
            mesh_runs.append({**mesh_report,'timings':json.dumps(mesh_report['timings'],sort_keys=True),
                              'protocol_path':str(STAGE/f'configs/{sample}_MESH.json'),'normalization':'depth=2*reference_train_camera_radius; voxel=depth/1024; sdf=5*voxel',
                              'interpreter_teardown_included':False})
            structure=rows(STAGE/f'metrics/{label}_MESH_STRUCTURE.csv');meshes+=structure
            reprojection=rows(STAGE/f'metrics/{label}_MESH_REPROJECTION.csv')
            for row in structure:
                views=[r for r in reprojection if r['mesh_stage']==row['mesh_stage']]
                assert len(views)==len(full)
                meshcomparison.append({**row,'heldout_views':len(views),**{k:mean_metric(views,k) for k in MESH},
                                       'aggregation':'Per-view arithmetic mean, consistency only; not geometry accuracy',
                                       'extraction_wall_seconds':mesh_report['wall_seconds'],'peak_process_RSS_KiB':mesh_report['peak_process_RSS_KiB']})
    write_csv(STAGE/'M2M3_TRAINING_RUNS.csv',training)
    write_csv(STAGE/'M2M3_PRUNING_EVENT_LOG.csv',events)
    write_csv(STAGE/'M2M3_NATIVE_CAPACITY_SUMMARIES.csv',capacity)
    write_csv(STAGE/'M2M3_HELDOUT_METRICS.csv',perviews)
    write_csv(STAGE/'M2M3_FOREGROUND_PROXY_METRICS.csv',foreground)
    write_csv(STAGE/'M2M3_BACKGROUND_GEOMETRY_PROXY_METRICS.csv',background)
    write_csv(STAGE/'M2M3_SCENE_SUMMARY.csv',summaries)
    write_csv(STAGE/'MESH_EXTRACTION_RUNS.csv',mesh_runs)
    write_csv(STAGE/'MESH_STRUCTURE_METRICS.csv',meshes)
    write_csv(STAGE/'M2M3_MESH_COMPARISON.csv',meshcomparison)
    effects=[];mesh_effects=[]
    for sample in SCENES:
        for left,right in [('M0','M1'),('M1','M2')]:
            a=next(r for r in summaries if r['sample']==sample and r['method']==left);b=next(r for r in summaries if r['sample']==sample and r['method']==right)
            for key in FULL+FG+BG+MODEL+['training_wall_seconds','peak_VRAM_allocated_bytes']:
                effects.append({'sample':sample,'comparison':f'{left}_to_{right}','metric':key,'baseline':a[key],'comparison_value':b[key],
                                'difference_comparison_minus_baseline':b[key]-a[key],
                                'training_repeats':1,'scope':'Three-scene single-seed mechanism pilot; no significance claim'})
            for mesh_stage in ['raw','post']:
                a=next(r for r in meshcomparison if r['sample']==sample and r['method']==left and r['mesh_stage']==mesh_stage)
                b=next(r for r in meshcomparison if r['sample']==sample and r['method']==right and r['mesh_stage']==mesh_stage)
                for key in ['vertices','triangles','connected_components','largest_component_triangle_ratio','boundary_edge_count','non_manifold_edge_count','degenerate_face_ratio']+MESH:
                    mesh_effects.append({'sample':sample,'comparison':f'{left}_to_{right}','mesh_stage':mesh_stage,'metric':key,
                                         'baseline':a[key],'comparison_value':b[key],'difference_comparison_minus_baseline':float(b[key])-float(a[key]),
                                         'independent_geometry_GT':False})
    write_csv(STAGE/'M2M3_PAIRED_EFFECTS.csv',effects);write_csv(STAGE/'M2M3_MESH_PAIRED_EFFECTS.csv',mesh_effects)
    print('SUMMARIZED',len(training),'model runs;',len(perviews),'held-out views;',len(events),'pruning events;',len(meshes),'main meshes',flush=True)

if __name__=='__main__':main()
