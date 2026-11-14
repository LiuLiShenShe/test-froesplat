#!/usr/bin/env python3
"""Measure untried automatic roots of a completed C5-C final round.

Each trial independently starts from the same completed native model. This is
an audit of its stopping budget, not a new C0-started comparison or a selected
cleaned model. It never exports a Gaussian model or touches heldout plants.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-output', type=Path, required=True)
    parser.add_argument('--implementation-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    module, completed, output = args.implementation_root.resolve(), args.completed_output.resolve(), args.output.resolve()
    sys.path.insert(0, str(module.parent))
    import numpy as np
    import torch
    import lpips
    from ForeSplat_AutoGaussianCleaner.cleaner.io import sha, write_json
    from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
    from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
    from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import native_subset
    from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.grouping import groups
    from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.legacy import features
    from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.evaluation import Evaluator
    from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import choose
    from ForeSplat_AutoGaussianCleaner.execution_provenance import native_renderer_fingerprint

    begin = time.perf_counter()
    marker = json.loads((completed/'COMPLETE.json').read_text())
    report = json.loads((completed/'cleanup_metrics.json').read_text())
    config, spec = report['config'], marker['scene']
    if report['method'] != 'C5-C' or spec['plant'] != 'KongQueZhuYu':
        raise ValueError('This development-only stopping audit requires the completed primary C5-C experiment')
    if (completed/'INVALIDATED.json').exists() or marker['native_sha256'] != sha(completed/'cleaned_gaussians.ply'):
        raise ValueError('Completed audit model differs from its native completion hash')
    if marker['input_sha256'] != sha(spec['model']):
        raise ValueError('Untouched original native checkpoint changed')
    for name, expected in config['implementation_sources'].items():
        if sha(module/name) != expected:
            raise ValueError('Audit must load the exact original cleaning implementation: '+name)
    if native_renderer_fingerprint([spec['gggs_repo']]) != config['native_renderer_fingerprint']:
        raise ValueError('Audit native GGGS renderer differs from the original experiment')
    edits = json.loads((completed/'edit_manifest.json').read_text())['edits']
    iteration = report['rounds']
    last = [v for v in edits if v['round'] == iteration]
    if any(v['accepted'] for v in last):
        raise ValueError('Audit assumes a final unchanged no-improving round')
    with np.load(completed/'rounds'/f'{iteration:02d}'/'evidence.npz') as cache:
        evidence = {k:cache[k].copy() for k in ['original_ids','prune','fg','bg','free_bg']}
    ids = np.load(completed/'original_gaussian_ids.npy')
    if not np.array_equal(ids, evidence['original_ids']):
        raise ValueError('Final native IDs differ from the unchanged final-round evidence')
    def fingerprint(values):
        return hashlib.sha256(np.sort(np.asarray(values, dtype=np.int64)).tobytes()).hexdigest()
    tried = {fingerprint(v['removed_ids']) for v in last if v['action'] == 'group_prune' and v['split_depth'] == 0}
    output.mkdir(parents=True, exist_ok=False)
    scene = Scene(spec, config)
    model, rows = scene.load_model(completed/'cleaned_gaussians.ply')
    grouped = groups(evidence['prune'], features(rows, knn=False),
                     {**evidence, 'bg':evidence['bg']+evidence['free_bg']}, config)
    untried = [g for g in grouped if fingerprint(ids[g]) not in tried]
    write_json(output/'AUDIT_PROTOCOL.json', dict(completed_output=str(completed),
        native_sha256=marker['native_sha256'], input_sha256=marker['input_sha256'],
        original_root_groups=len(grouped), tried_root_groups=len(tried), untried_root_groups=len(untried),
        configuration=config, helper_sha256=sha(Path(__file__)),
        scope='Independent diagnostic deletion trials from the completed main C5-C model. No Gaussian model export, no heldout test, no threshold adjustment; gains cannot be summed as a cumulative accepted result.'))
    original_model, _ = scene.load_model(spec['model'])
    surface = external_surface(scene, scene.evidence_indices())
    perceptual = lpips.LPIPS(net='alex').cuda().eval()
    # The main run may have reused an external original-geometry cache. A
    # self-contained audit creates its own fresh native original reference.
    evaluator = Evaluator(scene, output, None, perceptual, surface, original_model=original_model)
    initial = evaluator.evaluate(original_model, output/'comparison/initial')
    del original_model; torch.cuda.empty_cache()
    current = evaluator.evaluate(model, output/'comparison/current')
    for label, actual in [('initial', initial), ('final', current)]:
        for key in ['fg_mse','fg_alpha','bg_alpha_mass_per_fg','thin_coverage','supported_fg_alpha']:
            if abs(actual['mean'][key]-report[label]['mean'][key]) > 1e-6:
                raise ValueError('Fresh audit reference differs from the completed native measurement: '+label+':'+key)
    results = []
    for index, local in enumerate(untried):
        keep = np.ones(len(ids), dtype=bool); keep[local] = False
        candidate = native_subset(scene, model, keep)
        folder = output/f'root_{index:03d}'
        metrics, affected = evaluator.evaluate_group(candidate, folder, ids[local], current)
        accepted, gain, local_gain, reasons = choose(metrics, current, initial, config)
        record = dict(automatic_untried_root=index, candidates=len(local),
            original_id_set_sha256=fingerprint(ids[local]), independently_accepted=accepted,
            measured_gain=gain, local_view_gain=local_gain, reasons=reasons,
            metrics_complete=metrics.get('metrics_complete', True), metrics=str((folder/'metrics.json').resolve()),
            predicted_bg=float((evidence['bg'][local]+evidence['free_bg'][local]).sum()),
            predicted_fg=float(evidence['fg'][local].sum()), affected_views=len(affected))
        results.append(record); write_json(folder/'AUDIT_DECISION.json', record)
        np.save(folder/'diagnostic_candidate_original_ids.npy', ids[local])
        write_json(output/'AUDIT_PROGRESS.json', dict(completed=len(results), expected=len(untried), decisions=results))
        print(record, flush=True)
        del candidate; torch.cuda.empty_cache()
    write_json(output/'STOPPING_BUDGET_AUDIT.json', dict(decisions=results,
        untried_root_groups=len(untried), individually_quality_accepted=sum(r['independently_accepted'] for r in results),
        individually_accepted_candidate_gaussians=sum(r['candidates'] for r in results if r['independently_accepted']),
        seconds=time.perf_counter()-begin, native_sha256=marker['native_sha256'],
        scope='Actual independent local trials, not a cumulative cleaned Gaussian model or C0-started matched experiment. Any production scheduling change requires new frozen development runs.'))


if __name__ == '__main__':
    main()
