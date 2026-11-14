#!/usr/bin/env python3
"""Audit real group decisions and KNN additionality, including running outputs."""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.foreground_support import project_metrics
from ForeSplat_AutoGaussianCleaner.collect_comparison import experiment_files


def recorded_trial_path(run, value):
    """Keep restarted-run diagnostics bound to their preserved archived files."""
    path=Path(value)
    if re.search(r'\.interrupted_\d+$',run.name):
        original=run.with_name(re.sub(r'\.interrupted_\d+$','',run.name)).resolve()
        if path.resolve().is_relative_to(original):
            return run/path.resolve().relative_to(original)
    return path


def group_benefit_diagnostic(candidate,current,config):
    """Keep the two independent AND branches of the engine's benefit guard."""
    before=current['mean']['bg_alpha_mass_per_fg']
    training=(before-candidate['mean']['bg_alpha_mass_per_fg'])/max(before,1e-12)
    training_local=max((b['bg_alpha_mass_per_fg']-a['bg_alpha_mass_per_fg'])/max(b['bg_alpha_mass_per_fg'],1e-12)
        for a,b in zip(candidate['views'],current['views']))
    free_before=current.get('free_background',{}).get('alpha_mass',0)
    free_after=candidate.get('free_background',{}).get('alpha_mass',free_before)
    free=(free_before-free_after)/max(free_before,1e-12)
    free_local=max([(b['alpha_mass']-a['alpha_mass'])/max(b['alpha_mass'],1e-12)
        for a,b in zip(candidate.get('free_background',{}).get('views',[]),current.get('free_background',{}).get('views',[]))] or [0])
    global_floor=config['group_minimum_global_gain'];local_floor=config['group_minimum_view_gain']
    training_severity=max(0,(global_floor-training)/global_floor,(local_floor-training_local)/local_floor)
    free_severity=max(0,(global_floor-free)/global_floor,(local_floor-free_local)/local_floor,(-1e-7-training)/1e-7)
    return dict(benefit_training_global_gain=training,benefit_training_local_gain=training_local,
        benefit_free_global_gain=free,benefit_free_local_gain=free_local,
        benefit_global_gain_floor=global_floor,benefit_local_gain_floor=local_floor,
        benefit_training_branch_pass=training>=global_floor and training_local>=local_floor,
        benefit_free_branch_pass=free>=global_floor and free_local>=local_floor and training>=-1e-7,
        benefit_training_branch_violation=training_severity,benefit_free_branch_violation=free_severity,
        benefit_rejection_severity=min(training_severity,free_severity))


def severity(rule,view,candidate,current,initial,config,item):
    # Structural guards use the fixed original photo/depth support in C10+.
    # Photometric values stay raw; report the threshold actually enforced.
    candidate=project_metrics(candidate,config)
    current=project_metrics(current,config)
    initial=project_metrics(initial,config)
    q=config['quality'];local=view!='global'
    a=candidate['mean'] if not local else next(v for v in candidate['views'] if v['name']==view)
    refs=[r['mean'] if not local else next(v for v in r['views'] if v['name']==view) for r in [current,initial]]
    aliases={'dark_leaf_mse':'dark_foreground_mse','dark_leaf_alpha':'dark_foreground_alpha',
        'foreground MSE regression':'fg_mse','foreground SSIM regression':'fg_ssim',
        'foreground LPIPS regression':'fg_masked_lpips','foreground-local boundary F1 regression':'fg_local_boundary_f1',
        'new depth holes':'new_depth_invalid_fraction','depth_holes':'new_depth_invalid_fraction'}
    key=aliases.get(rule,rule);prefix='view_' if local else ''
    if rule=='unsupported_local_photo':
        actual=a['unsupported_photo_max_rmse_increase']
        threshold=config.get('unsupported_local_rmse_cap',1/255)
        return actual,threshold,(actual-threshold)/max(threshold,1e-12)
    drops={'fg_alpha':'foreground_alpha_drop','fg_coverage':'foreground_coverage_drop',
        'skeleton_coverage':'skeleton_coverage_drop','thin_coverage':'thin_coverage_drop',
        'boundary_coverage':'boundary_coverage_drop','fg_local_boundary_f1':'boundary_f1_drop','boundary_f1':'boundary_f1_drop',
        'dark_foreground_alpha':'foreground_alpha_drop'}
    if key in drops:
        tk=prefix+drops[key]
        if local and key in ['skeleton_coverage','boundary_coverage','thin_coverage']:tk='view_thin_coverage_drop'
        if local and key=='fg_coverage':tk='view_foreground_alpha_drop'
        actual=a[key];threshold=max(r[key] for r in refs)-q[tk]
        return actual,threshold,(threshold-actual)/max(abs(threshold),1e-12)
    if key in ['fg_mse','dark_foreground_mse']:
        actual=a[key];threshold=min(r[key] for r in refs)*(1+q[prefix+'foreground_mse_relative_increase'])+q['foreground_mse_absolute_increase']
        return actual,threshold,(actual-threshold)/max(abs(threshold),1e-12)
    if key in ['depth_relative_p95','normal_angle_p95_deg','depth_changed_fraction']:
        actual=a[key];threshold=q[prefix+key];return actual,threshold,(actual-threshold)/max(abs(threshold),1e-12)
    if key=='new_depth_invalid_fraction':
        actual=a[key];threshold=q[prefix+'foreground_alpha_drop'];return actual,threshold,(actual-threshold)/max(threshold,1e-12)
    if key in ['fg_ssim','fg_masked_lpips']:
        actual=a[key];sign=-1 if key=='fg_ssim' else 1
        threshold=(max(r[key] for r in refs)-.001) if sign<0 else (min(r[key] for r in refs)+.001)
        return actual,threshold,sign*(actual-threshold)/max(abs(threshold),1e-12)
    if 'benefit' in rule:
        actual=item['relative_gain'];threshold=config['group_minimum_global_gain']
        return actual,threshold,group_benefit_diagnostic(candidate,current,config)['benefit_rejection_severity']
    return '','',''


def main():
    base=Path(__file__).resolve().parent/'experiments/GGGS_Cleaner_Comparison';out=base/'comparisons'
    rows=[];reject=[];knn=[];summaries=[];schedules=[]
    for ref in sorted(experiment_files(base,'original_reference.json')):
        run=ref.parent
        if any((p/'INVALIDATED.json').exists() for p in [run,*run.parents] if p==base or base in p.parents):continue
        interrupted=any((p/'INTERRUPTED.json').exists() for p in [run,*run.parents] if p==base or base in p.parents)
        if not (run/'comparison/initial/metrics.json').exists():continue
        reference=json.loads(ref.read_text())
        if not str(reference.get('variant','')).startswith('C5-'):continue
        config=reference['config'];initial=json.loads((run/'comparison/initial/metrics.json').read_text());current=initial
        items={}
        manifest=run/'edit_manifest.json'
        if manifest.exists():
            for item in json.loads(manifest.read_text())['edits']:
                key=(item['round'],item['action'],item.get('group',0),item.get('step',0));items[key]=item
        for decision in sorted((run/'rounds').rglob('decision.json')):
            item=json.loads(decision.read_text());key=(item['round'],item['action'],item.get('group',0),item.get('step',0));items[key]=item
        items=sorted(items.values(),key=lambda x:(x['round'],x['action']=='local_covariance_refine',x.get('group',0),x.get('step',0)))
        counts=Counter()
        for index,item in enumerate(items):
            if 'metrics' not in item:continue
            path=recorded_trial_path(run,item['metrics'])
            if not path.exists():continue
            candidate=json.loads(path.read_text());changed=''
            if item.get('parameters'):
                with np.load(recorded_trial_path(run,item['parameters'])) as z:
                    changed=int(np.logical_or.reduce([z['old'][k]!=z['new'][k] for k in z['old'].dtype.names]).sum())
            complete=candidate.get('metrics_complete',True)
            if item['accepted'] and not complete:
                raise ValueError(f'An incomplete trial was recorded as accepted: {path}')
            measured=set(candidate.get('rerendered_views',[v['index'] for v in candidate['views']]))
            old_views={v['index']:v for v in current['views']}
            observed=[v for v in candidate['views'] if v['index'] in measured]
            observed_alpha_loss=float(np.mean([old_views[v['index']]['fg_alpha']-v['fg_alpha'] for v in observed])) if observed else 0.
            observed_mse_change=float(np.mean([v['fg_mse']-old_views[v['index']]['fg_mse'] for v in observed])) if observed else 0.
            common=dict(method=reference['variant'],version=config['method_version'],plant=reference['scene']['plant'],
                experiment=str(run.resolve()),candidate_group=f"{reference['variant']}_{item['round']:02d}_{item['action']}_{item.get('group',index+1):03d}_{item.get('step',0):03d}",
                experiment_status='interrupted_diagnostic' if interrupted else 'complete' if (run/'COMPLETE.json').exists() else 'running_diagnostic',
                candidate_count=item['candidates'],actual_parameter_changed_rows=changed,action=item['action'],round=item['round'],
                background_gain=item['relative_gain'],local_background_gain=item['local_view_gain'],
                training_background_gain=item.get('training_relative_gain',''),free_background_gain=item.get('free_relative_gain',''),
                root_coverage_trial=item.get('root_coverage_trial',item.get('split_depth',-1)==0),
                beyond_initial_trial_budget=item.get('beyond_initial_trial_budget',False),
                metrics_complete=complete,
                dark_mse_tradeoff_views=len(candidate.get('dark_mse_tradeoff_analysis',[])),
                dark_mse_tradeoff_analysis=json.dumps(candidate.get('dark_mse_tradeoff_analysis',[]),ensure_ascii=False),
                depth_recovery_views=len(candidate.get('depth_recovery_analysis',[])),
                depth_recovery_analysis=json.dumps(candidate.get('depth_recovery_analysis',[]),ensure_ascii=False),
                measured_view_count=len(measured),unmeasured_affected_view_count=len(candidate.get('unmeasured_affected_views',[])),
                background_gain_scope=item.get('background_gain_scope','Complete actual evaluation'),
                foreground_change_scope='Complete original camera inventory' if complete else 'Measured preflight cameras only; global foreground change unmeasured',
                observed_mean_foreground_loss=observed_alpha_loss,observed_mean_foreground_mse_change=observed_mse_change,
                foreground_loss=current['mean']['fg_alpha']-candidate['mean']['fg_alpha'] if complete else '',
                supported_foreground_loss=(current['mean'].get('supported_fg_alpha',current['mean']['fg_alpha'])-candidate['mean'].get('supported_fg_alpha',candidate['mean']['fg_alpha'])) if complete else '',
                foreground_mse_change=candidate['mean']['fg_mse']-current['mean']['fg_mse'] if complete else '',
                boundary_loss=current['mean']['fg_local_boundary_f1']-candidate['mean']['fg_local_boundary_f1'] if complete else '',
                depth_change=candidate['mean']['depth_changed_fraction'] if complete else '',accepted=item['accepted'],
                initial_fg_contribution=item.get('predicted_fg',''),initial_bg_contribution=item.get('predicted_bg',''),
                gain_definition=item.get('gain_definition','Actual max(training, auxiliary free) relative gain'),metrics=str(path))
            if complete:
                common.update(group_benefit_diagnostic(candidate,current,config))
            rows.append(common)
            for reason in item['reasons'] or ['']:
                name,rule=('global',reason)
                if ':' in reason and reason.split(':',1)[0] in {v['name'] for v in candidate['views']}:
                    name,rule=reason.split(':',1)
                elif ':' in reason and reason.split(':',1)[0] in ['fg_alpha','boundary_f1','thin_coverage','boundary_coverage','fg_coverage','skeleton_coverage']:
                    rule=reason.split(':',1)[0]
                for r in rule.split(','):
                    actual,threshold,s=severity(r,name,candidate,current,initial,config,item) if r else ('','','')
                    scope=('fixed_original_photo_or_depth_structural_support' if config.get('foreground_support_mode')=='photo_relative_or_depth' and r in ['fg_alpha','thin_coverage','boundary_coverage','fg_coverage','skeleton_coverage','dark_leaf_alpha','depth_relative_p95','normal_angle_p95_deg','depth_changed_fraction','new depth holes','depth_holes'] else
                        'every_excluded_original_support_pixel' if r=='unsupported_local_photo' else 'raw_photometry_or_target_local_contour_or_background_gain')
                    reject.append(dict(**common,rejection_rule=r,original_rejection_reason=reason,rejection_view=name,actual=actual,threshold=threshold,rejection_severity=s,metric_scope=scope))
                    if r:counts[r]+=1
            if item['accepted']:current=candidate
        for schedule_path in sorted((run/'rounds').glob('*/group_scheduling.json')):
            schedule=json.loads(schedule_path.read_text())
            schedules.append(dict(experiment=str(run.resolve()),method=reference['variant'],version=config['method_version'],
                plant=reference['scene']['plant'],round=schedule_path.parent.name,
                trials=schedule['trials'],initial_trial_budget=schedule.get('initial_trial_budget',config['group_max_trials']),
                original_root_groups=schedule.get('original_root_groups',''),root_coverage_trials=schedule.get('root_coverage_trials',''),
                root_coverage_extra_trials=schedule.get('root_coverage_extra_trials',''),untried_root_groups=schedule.get('untried_root_groups',''),
                untried_groups=schedule['untried_groups'],untried_predicted_bg=schedule['untried_predicted_bg'],
                order=schedule['order'],untried_groups_are_not_claimed_safe=schedule['untried_groups_are_not_claimed_safe']))
        summaries.append(dict(experiment=str(run.resolve()),method=reference['variant'],version=config['method_version'],
            experiment_status='interrupted_diagnostic' if interrupted else 'complete' if (run/'COMPLETE.json').exists() else 'running_diagnostic',
            completed=(run/'COMPLETE.json').exists(),trials=len(rows and [r for r in rows if r['experiment']==str(run.resolve())] or []),
            accepted_trials=sum(i['accepted'] for i in items),rejection_rules=dict(counts)))
        priors=run/'discovery_priors.npz'
        if not priors.exists():continue
        with np.load(priors) as z:bits=z['reason_bits'].copy()
        removed=np.zeros(len(bits),bool)
        if (run/'original_gaussian_ids.npy').exists():
            removed[:]=True;removed[np.load(run/'original_gaussian_ids.npy')]=False
        for file in sorted((run/'rounds').glob('*/evidence.npz')):
            with np.load(file) as e:
                oid=e['original_ids'];bg=e['bg']+e['free_bg'];fraction=bg/np.maximum(e['fg']+bg,1e-15)
                sensitive=(e['boundary']+e['thin'])/np.maximum(e['fg'],1e-15)>.2
                valid=(bg>1e-7)&((e['visible_views']>=2)|(e['free_visible_views']>=2))
                prior_without=(bits[oid]&127)!=0
                safe=valid&((fraction>config['candidate_background_fraction'])|(prior_without&(fraction>config['mixed_candidate_background_fraction'])))&~sensitive
                if config.get('protect_supported_centers',False):
                    center_file=run/'original_center_depth_support.npy'
                    if not center_file.exists():raise ValueError(f'Missing original center protection: {center_file}')
                    centers=np.load(center_file,mmap_mode='r')
                    safe &= centers[oid]<config.get('center_minimum_views',2)
                if not config.get('depth_proxy_as_hint',False):safe &= (e['geometry_views']<2)&(e['occlusion_views']==0)
                p=np.flatnonzero(safe);p=p[np.argsort(-bg[p]/(1+e['fg'][p]/1e-8),kind='stable')]
                p=p[np.cumsum(e['fg'][p])<=config['candidate_foreground_budget']*e['fg'].sum()]
                additional=np.setdiff1d(oid[e['prune']],oid[p]);local=np.isin(oid,additional)
                knn.append(dict(experiment=str(run.resolve()),version=config['method_version'],method=reference['variant'],round=file.parent.name,
                    original_knn_only_priors=int((bits==128).sum()),prune_candidates=len(e['prune']),
                    extra_candidates_due_to_knn=len(additional),extra_candidates_actually_removed_so_far=int(removed[additional].sum()),
                    extra_fg_contribution=float(e['fg'][local].sum()),extra_bg_contribution=float(bg[local].sum()),
                    final_model_complete=(run/'COMPLETE.json').exists(),evidence=str(file.resolve())))
    def save(name,data):
        if not data:return
        fields=list(dict.fromkeys(k for row in data for k in row))
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    save('C5_GROUP_DECISIONS.csv',rows);save('KNN_DISCOVERY_ADDITIONALITY.csv',knn);save('C5_ROOT_GROUP_COVERAGE.csv',schedules)
    grouped={}
    for row in reject:
        if row['accepted'] or not row['metrics_complete']:continue
        key=(row['experiment'],row['candidate_group'])
        grouped.setdefault(key,[]).append(row)
    closest=[]
    for experiment in sorted({k[0] for k in grouped}):
        candidates=[]
        for key,violations in grouped.items():
            if key[0]!=experiment:continue
            common=violations[0]
            values=[float(v['rejection_severity']) for v in violations if isinstance(v['rejection_severity'],(float,int))]
            if not values:continue
            candidates.append(dict(experiment=experiment,candidate_group=common['candidate_group'],
                candidate_count=common['candidate_count'],background_gain=common['background_gain'],
                maximum_normalized_original_rule_violation=max(values),
                total_normalized_original_rule_violation=sum(max(0,v) for v in values),
                rejection_rules=';'.join(v['rejection_rule'] for v in violations),
                rejection_views=';'.join(sorted({v['rejection_view'] for v in violations})),
                metrics=common['metrics'],rank_scope='Complete actual trial metrics; normalized original-rule violations, with runtime dark/depth exception analyses separately recorded'))
        candidates.sort(key=lambda v:(v['maximum_normalized_original_rule_violation'],v['total_normalized_original_rule_violation'],-v['background_gain']))
        closest.extend(dict(rank=i+1,**candidate) for i,candidate in enumerate(candidates[:5]))
    save('CLOSEST_REJECTED_CANDIDATES.csv',closest)
    with (out/'C4_REJECTION_ANALYSIS.csv').open() as f:old=list(csv.DictReader(f))
    save('CANDIDATE_REJECTION_ANALYSIS.csv',old+reject)
    write_json(out/'C5_REJECTION_SUMMARY.json',summaries)
    print('Actual group trials:',len(rows),'KNN round audits:',len(knn))


if __name__=='__main__':main()
