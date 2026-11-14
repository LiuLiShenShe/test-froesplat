#!/usr/bin/env python3
"""Extract actual V3 candidate rejection values, thresholds and severities."""
import sys
import csv
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json


def main():
    base = Path(__file__).resolve().parent
    src = base/'experiments/all_19_plants_v3/development/KongQueZhuYu/C'
    out = base/'experiments/GGGS_Cleaner_Comparison/comparisons'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src/'edit_manifest.json').read_text())
    initial = json.loads((src/'comparison/initial/metrics.json').read_text())
    config = json.loads((src/'original_reference.json').read_text())['config']
    q = config['quality']
    with np.load(src/'rounds/01/evidence.npz') as z:
        e = {k:z[k] for k in ['fg','bg','boundary','thin','geometry_views','original_ids']}
    old_views = {v['name']:v for v in initial['views']}
    data, counts = [], {}
    mapping = dict(fg_alpha=('foreground_alpha_drop',-1), boundary_f1=('boundary_f1_drop',-1),
                   thin_coverage=('thin_coverage_drop',-1), boundary_coverage=('boundary_coverage_drop',-1),
                   fg_mse=('foreground_mse_relative_increase',1), dark_leaf_mse=('view_foreground_mse_relative_increase',1),
                   dark_leaf_alpha=('view_foreground_alpha_drop',-1), depth_changed_fraction=('depth_changed_fraction',1))
    for i,item in enumerate(manifest['edits']):
        cand = json.loads(Path(item['metrics']).read_text())
        views = {v['name']:v for v in cand['views']}
        ids = np.array(item.get('removed_ids',item.get('changed_ids',[])),np.int64)
        common = dict(candidate_group=f"C4_{item['action']}_{i+1}",candidate_count=len(ids),
            background_gain=item['relative_gain'],foreground_loss=initial['mean']['fg_alpha']-cand['mean']['fg_alpha'],
            boundary_loss=initial['mean']['boundary_f1']-cand['mean']['boundary_f1'],
            depth_change=cand['mean']['depth_changed_fraction'],accepted=item['accepted'],
            initial_fg_contribution=float(e['fg'][ids].sum()),initial_bg_contribution=float(e['bg'][ids].sum()),
            sensitive_contributors=int(((e['boundary'][ids]+e['thin'][ids])>0).sum()),
            depth_supported_contributors=int((e['geometry_views'][ids]>=2).sum()))
        for reason in item['reasons']:
            local = reason.startswith('crop_')
            name, rules = reason.split(':',1) if local else ('global',reason.split(':',1)[0])
            for rule in rules.split(','):
                av,bv = (views[name],old_views[name]) if local else (cand['mean'],initial['mean'])
                key = {'dark_leaf_mse':'dark_foreground_mse','dark_leaf_alpha':'dark_foreground_alpha'}.get(rule,rule)
                actual=threshold=severity=''
                if rule == 'background improvement plateau':
                    actual=item['relative_gain'];threshold=config['minimum_relative_gain'];severity=(threshold-actual)/threshold
                elif rule == 'foreground MSE regression':
                    actual=av['fg_mse'];threshold=bv['fg_mse']*(1+q['foreground_mse_relative_increase'])+q['foreground_mse_absolute_increase'];severity=(actual-threshold)/threshold
                elif rule in mapping:
                    tk,sign = mapping[rule]
                    if local and not tk.startswith('view_'):
                        tk='view_'+tk
                    if tk in ['view_boundary_coverage_drop','view_thin_coverage_drop']:
                        tk='view_thin_coverage_drop'
                    tol=q[tk]
                    actual=av[key]
                    if key in ['fg_mse','dark_foreground_mse']:
                        threshold=bv[key]*(1+tol)+q['foreground_mse_absolute_increase']
                    elif sign < 0:
                        threshold=bv[key]-tol
                    else:
                        threshold=tol
                    severity=sign*(actual-threshold)/max(abs(threshold),1e-12)
                data.append(dict(**common,rejection_rule=rule,rejection_view=name,actual=actual,threshold=threshold,
                                 rejection_severity=severity,metrics=item['metrics']))
                counts[rule]=counts.get(rule,0)+1
    with (out/'CANDIDATE_REJECTION_ANALYSIS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write_json(out/'C4_REJECTION_SUMMARY.json',dict(counts=counts,total_candidates=len(manifest['edits']),
        accepted=sum(x['accepted'] for x in manifest['edits']), source=str(src),
        diagnosis=dict(discovery='working: 31955 prune and 4096 refinement candidates',
            grouping='failed: unrelated contributors mixed; tie-ranked zero-FG points precede high-BG benefit',
            optimization='bounded mixed optimization produces 0.53–1.34% proposed gain but local dark/geometry regressions',
            acceptance='full prune unsafe in two views; half/quarter otherwise feasible but below global 0.2% gate',
            metric_scope='Global boundary F1 includes remote background contours; must measure target-local contour independently'),
        closest_quality_feasible='prune_0.5: 15978 IDs, quality checks passed, gain .19146076% < .2% gate',
        warning='Candidate gains are NOT accepted cleanup gains'))


if __name__=='__main__':
    main()
