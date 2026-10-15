from __future__ import annotations
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from common import *

PRIMARY=('PSNR','SSIM','LPIPS')
FOREGROUND=('foreground_MSE','foreground_PSNR','fixed_mask_composited_SSIM','fixed_mask_composited_LPIPS','foreground_absolute_color_error','foreground_coverage_proxy','missing_alpha_fraction')
BACKGROUND=('outside_ROI_RGB_energy','outside_ROI_alpha_mean','outside_ROI_alpha_sum','proxy_silhouette_IoU',
            'outside_alpha_components_ge16px','outside_alpha_largest_component_pixels','outside_alpha_pixels',
            'depth_valid_ROI_fraction','depth_p01','depth_p05','depth_p50','depth_p95','depth_p99',
            'normal_consistency_cosine_proxy','normal_consistency_abs_cosine_proxy')
DISTRIBUTION=('Gaussian_count','opacity_p01','opacity_p50','opacity_p95','opacity_p99','scale_p01','scale_p50','scale_p95','scale_p99',
              'distance_from_sparse_centroid_p01','distance_from_sparse_centroid_p50','distance_from_sparse_centroid_p95','distance_from_sparse_centroid_p99')
METRICS=PRIMARY+FOREGROUND+BACKGROUND+DISTRIBUTION
LOWER={'LPIPS','foreground_MSE','fixed_mask_composited_LPIPS','foreground_absolute_color_error','missing_alpha_fraction','outside_ROI_RGB_energy','outside_ROI_alpha_mean','outside_ROI_alpha_sum'}


def descriptive(values):
    return {'mean':statistics.mean(values),'sample_SD':statistics.stdev(values) if len(values)>1 else None,'min':min(values),'max':max(values)}


def row_mean(rows,key):
    vals=[float(r[key]) for r in rows if r.get(key,'') not in ('',None)]
    return statistics.mean(vals) if vals else ''


def flags(values):
    flagged=[]
    for index,value in enumerate(values):
        other=values[:index]+values[index+1:]
        sd=statistics.stdev(other)
        if sd>0 and abs(value-statistics.mean(other))>3*sd:flagged.append(index)
    return flagged


def classification(raw,roi):
    # Uses only the prospective rule recorded in EXPERIMENT_PLAN.json.
    raw_direction=[1 if x>0 else -1 if x<0 else 0 for x in raw['effects']]
    roi_direction=[1 if x>0 else -1 if x<0 else 0 for x in roi['effects']]
    if len(set(raw_direction))==1 and len(set(roi_direction))==1 and raw_direction[0]==roi_direction[0] and raw_direction[0]!=0:
        exceeds=all(abs(r['paired_mean'])>max(r['B_sample_SD'],r['C_sample_SD']) for r in (raw,roi))
        if exceeds:return 'ROBUST_POSITIVE' if raw_direction[0]>0 else 'ROBUST_NEGATIVE'
        return 'INCONCLUSIVE'
    if any(a*b<0 for a,b in zip(raw_direction,roi_direction)):return 'MIXED'
    return 'INCONCLUSIVE'


def main():
    reports=[read_json(p) for p in sorted((STAGE/'reports').glob('*_EVALUATION.json'))]
    assert len(reports)==16 and all(r['status']=='PASS' for r in reports)
    full=[];foreground=[];background=[];manifest=[]
    legacy_runs={(r['sample'],r['group']):r for r in read_csv(E1/'TRAINING_RUN_MANIFEST.csv')}
    for report in reports:
        sample,group,seed=report['sample'],report['group'],report['seed'];label=run_id(sample,group,seed)
        tables=[read_csv(STAGE/f'metrics/{label}_{suffix}.csv') for suffix in ('FULL','FOREGROUND','BACKGROUND_GEOMETRY')]
        for target,rows in zip((full,foreground,background),tables):target.extend(rows)
        if seed==0:
            legacy=legacy_runs[(sample,group)]
            runtime=read_json(E1/f'reports/{sample}_{group}_RUNTIME.json')
            process=read_json(E1/f'reports/{sample}_{group}_PROCESS.json')
            validation=read_json(E1/f'reports/{sample}_{group}_VALIDATION.json')
            checkpoint=validation['checkpoint_path'];ply=validation['final_PLY_path']
            cfgpath=ROOT/legacy['config_path'];attempt='inherited';source_commit=E1_TRAIN_COMMIT
            initial_hash='HISTORICAL_NOT_CAPTURED';initial_parameters='E1 observer asserted E0 preflight hashes'
            peak=process['sampled_gpu_peak_mib'];command=process['observer_command'];status='INHERITED_COMPLETED'
            process_wall=process['process_wall_seconds'];envpath=E1/'reports/ENVIRONMENT.json'
        else:
            accepted=[read_json(p) for p in (STAGE/'reports').glob(f'{label}_attempt*_VALIDATION.json') if read_json(p)['status']=='PASS']
            assert len(accepted)==1;validation=accepted[0];attempt=validation['attempt'];source_commit=BASELINE_COMMIT
            runtime=read_json(STAGE/f'reports/{run_id(sample,group,seed,attempt)}_RUNTIME.json')
            process=read_json(STAGE/f'reports/{run_id(sample,group,seed,attempt)}_PROCESS.json')
            checkpoint=validation['checkpoint_path'];ply=validation['PLY_path'];cfgpath=e2_config_path(sample,group,seed)
            initial_hash=runtime['initial_rng_sha256']['combined']
            initial_parameters=str(STAGE/f'reports/{run_id(sample,group,seed,attempt)}_INITIALIZATION.json')
            peak=process['sampled_device_peak_MiB'];command=process['command_argv'];status='NEW_COMPLETED'
            process_wall=process['wall_seconds'];envpath=STAGE/'reports/PRETRAINING_ENVIRONMENT.json'
        manifest.append({'sample':sample,'group':group,'seed':seed,'attempt':attempt,'independent_training_run':True,'status':status,
                         'source_commit':source_commit,'iterations':runtime['observed_iteration'],'native_optimizer_updates':29999,
                         'config_path':str(cfgpath),'config_sha256':sha256(cfgpath),
                         'executed_config_path':str(Path(report['model_path'])/'FULL_FROZEN_TRAIN_CONFIG.json'),
                         'executed_config_sha256':sha256(Path(report['model_path'])/'FULL_FROZEN_TRAIN_CONFIG.json'),
                         'output_path':report['model_path'],
                         'checkpoint_path':checkpoint,'checkpoint_sha256':validation['checkpoint_sha256'],'checkpoint_bytes':Path(checkpoint).stat().st_size,
                         'PLY_path':ply,'PLY_sha256':sha256(ply),'PLY_bytes':Path(ply).stat().st_size,
                         'Gaussian_count':report['Gaussian_distribution']['Gaussian_count'],'GPU_UUID':GPU_UUID,
                         'environment_path':str(envpath),'environment_sha256':sha256(envpath),'initial_RNG_combined_sha256':initial_hash,
                         'initialization_evidence':initial_parameters,'start_time':runtime['start_time'],'end_time':runtime['end_time'],
                         'training_wall_seconds':runtime['training_wall_seconds'],'process_wall_seconds':process_wall,
                         'peak_VRAM_allocated_bytes':runtime['cuda_peak_allocated_bytes'],'peak_VRAM_reserved_bytes':runtime['cuda_peak_reserved_bytes'],
                         'sampled_device_peak_MiB':peak,'command_argv_json':json.dumps(command,ensure_ascii=False),
                         'stdout_stderr_path':process['stdout_stderr_path'],'stdout_stderr_sha256':process['stdout_stderr_sha256'],
                         'seed0_comparability':'SOURCE_PATH_EQUIVALENT_HISTORICAL_RNG_NOT_CAPTURED','shared_model_alias':''})
    # HongZhang is one original C model, explicitly aliased as B, not a new run.
    for table in (full,foreground,background):
        alias=[{**r,'group':'B_R0','shared_model_alias':'HongZhang/C_R1/seed0'} for r in table if r['sample']=='HongZhang' and r['group']=='C_R1' and int(r['seed'])==0]
        table.extend(alias)
    alias_manifest=next(r for r in manifest if r['sample']=='HongZhang' and r['group']=='C_R1' and r['seed']==0)
    manifest.append({**alias_manifest,'group':'B_R0','independent_training_run':False,'status':'INHERITED_SHARED_ALIAS','shared_model_alias':'HongZhang/C_R1/seed0'})
    for name,rows in [('MULTISEED_PER_VIEW_METRICS.csv',full),('FOREGROUND_PROXY_METRICS.csv',foreground),('BACKGROUND_GEOMETRY_PROXY_METRICS.csv',background),('MULTISEED_TRAINING_MANIFEST.csv',manifest)]:
        rows.sort(key=lambda r:(r['sample'],r['group'],int(r['seed']),r.get('loader_image_name','')))
        write_csv(STAGE/name,rows,list(dict.fromkeys(k for r in rows for k in r)))
    summaries=[];valid_counts={};validity=[];omissions=[]
    for m in manifest:
        identity=(m['sample'],m['group'],int(m['seed']))
        selected=[]
        for table in (full,foreground,background):
            selected.append([r for r in table if (r['sample'],r['group'],int(r['seed']))==identity])
        assert len({len(rows) for rows in selected})==1
        for table in selected:
            assert len({r['loader_image_name'] for r in table})==len(table)
        merged=[{**a,**b,**c} for a,b,c in zip(*[sorted(t,key=lambda r:r['loader_image_name']) for t in selected])]
        for metric in METRICS:
            defined=sum(row.get(metric,'') not in ('',None) for row in merged)
            valid_counts[identity,metric]=defined
            validity.append({'sample':identity[0],'group':identity[1],'seed':identity[2],
                             'metric':metric,'defined_views':defined,'total_heldout_views':len(merged),
                             'all_fixed_views_defined':defined==len(merged)})
            for row in merged:
                if row.get(metric,'') not in ('',None):continue
                if metric.startswith('depth_'):
                    reason='No finite positive depth on alpha>0.5 fixed-ROI support'
                elif metric.startswith('normal_'):
                    reason='No finite nonzero normal pair on valid depth/alpha/ROI support'
                elif metric.startswith('outside_ROI_'):
                    reason='No outside-ROI pixels; outside mean is undefined'
                else:
                    reason='Metric undefined on this frozen view; not imputed'
                omissions.append({'sample':identity[0],'group':identity[1],'seed':identity[2],
                                  'loader_image_name':row['loader_image_name'],'metric':metric,'reason':reason})
        s={'sample':identity[0],'group':identity[1],'seed':identity[2],'views':len(merged),
           **{key:row_mean(merged,key) for key in METRICS},'shared_model_alias':m['shared_model_alias'],
           'independent_training_run':m['independent_training_run'],'training_wall_seconds':m['training_wall_seconds'],
           'peak_VRAM_allocated_bytes':m['peak_VRAM_allocated_bytes'],'checkpoint_sha256':m['checkpoint_sha256'],
           'ROI_is_independent_GT':False,'geometry_GT':'NONE'}
        summaries.append(s)
    summaries.sort(key=lambda r:(r['sample'],r['group'],int(r['seed'])))
    write_csv(STAGE/'MULTISEED_SCENE_SUMMARY.csv',summaries)
    write_csv(STAGE/'reports/METRIC_VALID_VIEW_COUNTS.csv',validity)
    write_csv(STAGE/'reports/METRIC_OMISSIONS.csv',omissions,
              ['sample','group','seed','loader_image_name','metric','reason'])
    indexed={(r['sample'],r['group'],r['seed']):r for r in summaries}
    effects=[];stats=[];decisions={};paired_omissions=[]
    for sample in SCENES:
        seed_effects=[]
        for seed in SEEDS:
            b=indexed[sample,'B_R0',seed];c=indexed[sample,'C_R1',seed]
            row={'sample':sample,'seed':seed,'experimental_unit':'ONE_TRAINING_PAIR','heldout_views':b['views']}
            for metric in METRICS:
                complete=all(valid_counts[(sample,group,seed),metric]==b['views'] for group in GROUPS)
                if b[metric]!='' and c[metric]!='' and complete:
                    row.update({f'B_{metric}':b[metric],f'C_{metric}':c[metric],f'delta_{metric}':c[metric]-b[metric]})
                else:
                    paired_omissions.append({'sample':sample,'seed':seed,'metric':metric,
                                             'reason':'At least one fixed heldout view is undefined; no unequal-subset paired comparison',
                                             'B_defined_views':valid_counts[(sample,'B_R0',seed),metric],
                                             'C_defined_views':valid_counts[(sample,'C_R1',seed),metric],
                                             'total_heldout_views':b['views']})
            effects.append(row);seed_effects.append(row)
        per_metric={}
        for metric in METRICS:
            if not all(f'delta_{metric}' in r for r in seed_effects):continue
            delta=[r[f'delta_{metric}'] for r in seed_effects];b=[r[f'B_{metric}'] for r in seed_effects];c=[r[f'C_{metric}'] for r in seed_effects]
            summary={'sample':sample,'metric':metric,'training_pairs':3,'paired_mean':statistics.mean(delta),
                     'paired_sample_SD':statistics.stdev(delta),'paired_min':min(delta),'paired_max':max(delta),
                     'seed0_delta':delta[0],'seed1_delta':delta[1],'seed2_delta':delta[2],
                     'effect_signs':','.join('+' if d>0 else '-' if d<0 else '0' for d in delta),
                     'preferred_direction':'LOWER' if metric in LOWER else 'HIGHER' if metric in PRIMARY+FOREGROUND or metric=='proxy_silhouette_IoU' else 'DESCRIPTIVE',
                     'B_mean':statistics.mean(b),'B_sample_SD':statistics.stdev(b),'B_min':min(b),'B_max':max(b),
                     'C_mean':statistics.mean(c),'C_sample_SD':statistics.stdev(c),'C_min':min(c),'C_max':max(c),
                     'B_descriptive_flagged_seeds':','.join(map(str,flags(b))),'C_descriptive_flagged_seeds':','.join(map(str,flags(c))),
                     'paired_descriptive_flagged_seeds':','.join(map(str,flags(delta))),
                     'no_exclusions':True,'SD_ddof':1,'independent_geometry_GT':False}
            maximum=max(summary['B_sample_SD'],summary['C_sample_SD'])
            summary['absolute_mean_effect_over_max_group_SD']=abs(summary['paired_mean'])/maximum if maximum else ''
            summary['mean_effect_exceeds_B_and_C_SD']=abs(summary['paired_mean'])>maximum
            stats.append(summary);per_metric[metric]={**summary,'effects':delta}
        decisions[sample]={'classification':classification(per_metric['PSNR'],per_metric['foreground_PSNR']),
                           'full_PSNR':per_metric['PSNR'],'foreground_PSNR':per_metric['foreground_PSNR']}
    added_only=[]
    for sample in SCENES:
        selected=[r for r in effects if r['sample']==sample and r['seed'] in NEW_SEEDS]
        for metric in METRICS:
            if not all(f'delta_{metric}' in r for r in selected):continue
            b=[r[f'B_{metric}'] for r in selected];c=[r[f'C_{metric}'] for r in selected];d=[r[f'delta_{metric}'] for r in selected]
            added_only.append({'sample':sample,'metric':metric,'training_pairs':2,'seeds':'1,2',
                               'paired_mean':statistics.mean(d),'paired_sample_SD':statistics.stdev(d),
                               'paired_min':min(d),'paired_max':max(d),'seed1_delta':d[0],'seed2_delta':d[1],
                               'effect_signs':','.join('+' if v>0 else '-' if v<0 else '0' for v in d),
                               'B_sample_SD':statistics.stdev(b),'C_sample_SD':statistics.stdev(c),
                               'scope':'Uniform current wrapper, descriptive n=2, no significance claim'})
    write_csv(STAGE/'reports/NEW_SEEDS_ONLY_STATISTICS.csv',added_only)
    write_csv(STAGE/'PAIRED_SEED_EFFECTS.csv',effects)
    write_csv(STAGE/'reports/PAIRED_METRIC_OMISSIONS.csv',paired_omissions,
              ['sample','seed','metric','reason','B_defined_views','C_defined_views','total_heldout_views'])
    write_csv(STAGE/'reports/PAIRED_MULTISEED_STATISTICS.csv',stats)
    classifications=[d['classification'] for d in decisions.values()]
    if all(v=='ROBUST_POSITIVE' for v in classifications):overall='ROBUST_POSITIVE'
    elif all(v=='ROBUST_NEGATIVE' for v in classifications):overall='ROBUST_NEGATIVE'
    elif 'MIXED' in classifications or ('ROBUST_POSITIVE' in classifications and 'ROBUST_NEGATIVE' in classifications):overall='MIXED'
    else:overall='INCONCLUSIVE'
    write_json(STAGE/'reports/ROBUSTNESS_DECISION.json',{'timestamp':now(),'scene_decisions':decisions,'R1_3D_BENEFIT':overall,
               'R1_FINAL_ROLE':'ROBUST_SELECTION_ONLY','M2M3_NEXT_ACTION':'REWRITE_TO_IMPLEMENTATION',
               'classification_scope':'Observed appearance of reconstructed models; not independent geometric accuracy',
               'no_independent_3D_GT':True,'no_training_repeat_exclusions':True,'actual_new_trainings':8,
               'heldout_rows':len(full),'unique_model_camera_renders':len(full)-27,'biological_scene_count':2,'seeds_per_changed_scene':3})
    print(json.dumps({'status':'SUMMARY_COMPLETE','rows':len(full),'scene_decisions':{k:v['classification'] for k,v in decisions.items()},'overall':overall},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
