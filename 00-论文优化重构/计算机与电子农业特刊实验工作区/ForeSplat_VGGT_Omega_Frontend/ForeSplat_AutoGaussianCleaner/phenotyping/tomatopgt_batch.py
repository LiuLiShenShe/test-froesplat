"""Run and report the longitudinal TomatoPGT reconstruction queue."""
import html
import shutil
import subprocess
import time

import batch_common as bc
from tomatopgt_pilot import MARKER_SCAN, MARKERS, OUT, SCAN, SCANS, STANDARDIZED, TomatoContext, run_scene


def summary(states, stage, active=None):
    from batch_measure import csvwrite
    from tomatopgt_evaluate import metrics

    folder = OUT / 'summary'
    rows, pairs, leaf_pairs = [], [], []
    for scene in states:
        base = OUT / scene
        get = lambda rel: bc.read(base / rel) if (base / rel).is_file() else {}
        m = bc.read(STANDARDIZED / scene / 'metadata.json')
        sfm = get(MARKERS['colmap'])
        scale, mesh, sem, comp, traits = (get(path) for path in [MARKERS['calibration'], MARKERS['tsdf'],
            'plant_only/complete.json', MARKERS['compound'], MARKERS['comparison']])
        visible = traits.get('visible_height', {})
        if visible.get('official_cm') is not None and visible.get('prediction_cm') is not None:
            pairs.append(dict(scene=scene, cultivar=m['cultivar'], **visible))
        for pair in traits.get('compound_pairs', []):
            leaf_pairs.append(dict(scene=scene, cultivar=m['cultivar'], **pair))
        rows.append(dict(scene=scene, cultivar=m['cultivar'], queue_status=states[scene]['status'],
                         failed_stage=states[scene].get('stage', '') if states[scene]['status'] == 'failed' else '',
                         rgb=m['num_input_images'], independent_colmap=sfm.get('registered_images'),
                         sparse_points=sfm.get('sparse_points'),
                         mean_sparse_error_working_px=sfm.get('mean_sparse_error_px'),
                         official_traits=bool(m.get('has_phenotype_reference')),
                         calibration=scale.get('status', ''), cm_per_unit=scale.get('scale_cm_per_unit'),
                         gggs_30k=(base / MARKERS['gggs']).is_file(), metric_mesh=mesh.get('metric', ''),
                         plant_faces=sem.get('plant_faces'),
                         leaf_candidates=get(MARKERS['instances']).get('stable_leaf_count'),
                         compound_candidates=comp.get('stable_leaf_count'),
                         official_compounds=traits.get('official_compound_count'),
                         paired_compounds=len(traits.get('compound_pairs', [])) if traits else '',
                         visible_height_cm=visible.get('prediction_cm'), official_height_cm=visible.get('official_cm'),
                         match_status=traits.get('compound_matching_status', '')))
    csvwrite(folder / 'batch_results.csv', rows)
    csvwrite(folder / 'batch_height_pairs.csv', pairs, fields=list(pairs[0]) if pairs else ['scene', 'cultivar', 'prediction_cm', 'official_cm'])
    csvwrite(folder / 'batch_compound_pairs.csv', leaf_pairs, fields=list(leaf_pairs[0]) if leaf_pairs else ['scene', 'cultivar', 'candidate_id', 'official_id'])
    bc.write(folder / 'batch_metrics.json', dict(height=metrics(pairs, 'prediction_cm', 'official_cm'),
             compound_length=metrics(leaf_pairs, 'length_cm', 'reference_length_cm'),
             compound_thickness_proxy=metrics(leaf_pairs, 'thickness_cm', 'reference_thickness_cm'),
             interpretation='Agreement with TomatoPGT derived point-cloud traits, not independent manual ground truth. Scans recur over time within cultivar.'))
    bc.write(folder / 'batch_progress.json', dict(batch_status=stage, active=active, total=len(states),
             complete=sum(v['status']=='complete' for v in states.values()), entries=states, updated_unix=time.time()))
    page = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
            '<title>ForeSplat TomatoPGT</title><style>body{font:15px system-ui;margin:20px;color:#19332a}',
            'table{border-collapse:collapse;white-space:nowrap}td,th{padding:7px;border-bottom:1px solid #d8e2dc}</style>',
            '<h1>TomatoPGT 批量结果</h1><p><a href="batch_results.csv">扫描结果</a> · ',
            '<a href="batch_height_pairs.csv">株高配对</a> · <a href="batch_compound_pairs.csv">复叶配对</a> · ',
            '<a href="batch_metrics.json">参考差异统计</a> · <a href="progress.json">当前步骤</a></p><div style="overflow:auto"><table>',
            '<tr><th>扫描</th><th>品种</th><th>进度</th><th>独立 COLMAP</th><th>厘米网格</th><th>小叶候选</th><th>复叶候选</th><th>官方复叶</th><th>已配对</th><th>可见株高 cm</th></tr>']
    for row in rows:
        scene = row['scene']
        name = f'<a href="../{scene}/summary/index.html">{scene}</a>' if (OUT / scene / 'summary/index.html').is_file() else scene
        if (OUT / scene / MARKERS['compound']).is_file():
            name += f' · <a href="../{scene}/compound_leaf_instances/index.html">复叶</a>'
        row['registered'] = f"{row['independent_colmap']}/{row['rgb']}" if row['independent_colmap'] is not None else ''
        page.append('<tr><td>' + name + '</td>' + ''.join('<td>' + html.escape(str(row[k] if row[k] is not None else '')) + '</td>'
            for k in ('cultivar','queue_status','registered','metric_mesh','leaf_candidates','compound_candidates','official_compounds','paired_compounds','visible_height_cm')) + '</tr>')
    page.append('</table></div>')
    tmp = folder / 'batch_index.html.tmp'
    tmp.write_text(''.join(page), encoding='utf-8')
    tmp.replace(folder / 'batch_index.html')


def run_all(stages):
    if stages[0] == 'colmap':
        anchor = TomatoContext(MARKER_SCAN)
        if not (anchor.out / MARKERS['calibration']).exists():
            run_scene(anchor, ['colmap', 'calibration'], reuse=True)
    order = list(dict.fromkeys([SCAN, MARKER_SCAN, 'BBH_05132025', 'CH_05132025',
                               *sorted(SCANS, key=lambda scene: (scene[-4:], scene.split('_')[1][:4], scene))]))
    states = {s: dict(status='queued') for s in order}
    for s in order:
        if stages[-1]=='comparison' and (OUT / s / MARKERS['comparison']).exists():
            states[s] = dict(status='complete')
    summary(states, 'running')
    for scene in order:
        if states[scene]['status']=='complete':
            continue
        if shutil.disk_usage(OUT).free < 22 * 2**30:
            summary(states, 'paused_low_disk')
            raise RuntimeError('TomatoPGT queue requires 22 GiB of free reserve')
        states[scene] = dict(status='running')
        summary(states, 'running', scene)
        try:
            run_scene(TomatoContext(scene), stages, reuse=True)
        except subprocess.CalledProcessError as exc:
            last = bc.read(OUT / 'summary/progress.json')
            states[scene] = dict(status='failed', stage=last['stage'], log=last['log'], returncode=exc.returncode)
            continue
        states[scene] = dict(status='complete' if stages[-1]=='comparison' else 'partial')
    summary(states, 'finished_with_failures' if any(s['status']=='failed' for s in states.values()) else 'finished')
