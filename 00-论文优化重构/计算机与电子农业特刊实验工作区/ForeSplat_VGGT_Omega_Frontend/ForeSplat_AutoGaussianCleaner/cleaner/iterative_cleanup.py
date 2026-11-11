import copy
import time
from pathlib import Path
import numpy as np
import torch
from .io import sha, save_ply, bake_viewer, write_json
from .runtime import Scene
from .contribution_analysis import analyze
from .geometry_consistency import external_surface, add_spatial_features
from .artifact_scoring import classify
from .automatic_selection import select
from .gaussian_refinement import refine
from ..evaluation.metrics import evaluate
from ..evaluation.free_view_evaluation import evaluate_free


def enhanced_analyze(scene, model, indices, config):
    e = analyze(scene, model, indices, config)
    add_spatial_features(e, model, None)
    scores = classify(e, config)
    uncertain_mass = e['bg'][scores['labels'] == 4].sum()
    remaining = sorted(set(range(len(scene.views)))-set(indices))
    if remaining and uncertain_mass > .02*max(e['bg'].sum(), 1e-12):
        print('A4 evidence enhancement: remaining', len(remaining), 'cameras', flush=True)
        extra = analyze(scene, model, remaining, config)
        a, b = len(indices), len(remaining)
        for key in extra:
            if key == 'records':
                e[key] += extra[key]
            elif key.endswith('_views') or key == 'depth_error_sum':
                e[key] += extra[key]
            elif key == 'max_radius_fraction':
                e[key] = np.maximum(e[key], extra[key])
            else:
                e[key] = (a*e[key]+b*extra[key])/(a+b)
    return e


def run(spec, config, output):
    begin = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = Scene(spec, config)
    model, original_rows = scene.load_model(spec['model'])
    rows = original_rows.copy()
    ids = np.arange(len(rows))
    original_sha = sha(spec['model'])
    write_json(output/'original_reference.json', dict(scene=spec, config=config,
        input_model_sha256=original_sha, original_gaussians=len(rows), physical_gpu=1))
    try:
        import lpips
        lp = lpips.LPIPS(net='alex').cuda().eval()
    except (ImportError, OSError, RuntimeError) as error:
        print('LPIPS unavailable:', str(error), flush=True)
        lp = None
    indices = scene.evidence_indices()
    surface = external_surface(scene, indices)
    initial = evaluate(scene, model, output/'comparison/initial', lpips_model=lp, save_images=True)
    initial_free = evaluate_free(scene, model, surface, output/'comparison/initial/free')
    current = initial
    edits = []
    stop = 'round_limit'
    for iteration in range(config['max_rounds']):
        dest = output/'rounds'/f'{iteration+1:02d}'
        dest.mkdir(parents=True, exist_ok=True)
        evidence = enhanced_analyze(scene, model, indices, config)
        add_spatial_features(evidence, model, surface)
        scores = classify(evidence, config)
        records = evidence.pop('records')
        np.savez(dest/'evidence.npz', original_ids=ids, **evidence, labels=scores['labels'], harm=scores['harm'])
        write_json(dest/'evidence_summary.json', dict(records=records, groups=scores['counts'],
            candidates=len(scores['prune']), mixed=len(scores['mixed'])))
        accepted_round = False
        prune_was_rejected = False
        if len(scores['prune']):
            # Global budget ranked pruning; back off uniformly after rejected full-view candidates.
            for fraction in [1., .5, .25]:
                remove = scores['prune'][:max(1, round(len(scores['prune'])*fraction))]
                keep = np.ones(len(rows), bool)
                keep[remove] = False
                path = dest/f'prune_{fraction}.ply'
                save_ply(path, rows[keep])
                candidate, _ = scene.load_model(path)
                metrics = evaluate(scene, candidate, dest/f'prune_{fraction}', output/'comparison/initial', lp)
                accepted, gain, reasons = select(metrics, current, initial, config)
                prune_was_rejected |= not accepted
                item = dict(round=iteration+1, action='prune', candidates=len(remove), fraction=fraction,
                    removed_ids=ids[remove].tolist(), accepted=accepted, relative_gain=gain, reasons=reasons, metrics=str((dest/f'prune_{fraction}'/'metrics.json').resolve()))
                edits.append(item)
                write_json(output/'progress.json', item)
                if accepted:
                    rows, ids, model, current = rows[keep].copy(), ids[keep], candidate, metrics
                    accepted_round = True
                    break
                del candidate
        # Reattribute after pruning; never reuse stale mixed scores after an accepted edit.
        if accepted_round and config['refinement']:
            evidence = enhanced_analyze(scene, model, indices, config)
            add_spatial_features(evidence, model, surface)
            scores = classify(evidence, config)
        if config['refinement'] and prune_was_rejected:
            # Rendered foreground benefit remains real even if geometric support is uncertain.
            # Whole-point deletion failed quality verification: offer these mixed contributors
            # bounded local optimization, using freshly attributed scores after any accepted prune.
            promote = np.flatnonzero((scores['labels'] == 2) & (evidence['fg'] > 1e-7) & (evidence['fg_views'] >= 2))
            scores['mixed'] = np.unique(np.concatenate([scores['mixed'], promote]))
            write_json(dest/'quality_driven_mixed_promotion.json', dict(original_ids=ids[promote],
                reason='Whole-point candidate rejected by multi-view quality; actual foreground contribution requires constrained local refinement',
                promoted=len(promote)))
        if config['refinement'] and len(scores['mixed']):
            mixed = scores['mixed']
            rank = evidence['bg'][mixed] * scores['harm'][mixed]
            mixed = mixed[np.argsort(-rank, kind='stable')[:config['refine_max_gaussians']]]
            best = {'rows': rows, 'metrics': current, 'accepted': False, 'step': 0}
            base_rows = rows.copy()
            def checkpoint(step, indices_local, parameters):
                changed = base_rows.copy()
                for axis in range(3):
                    changed[f'scale_{axis}'][indices_local] = parameters['scale'][:, axis]
                for axis in range(4):
                    changed[f'rot_{axis}'][indices_local] = parameters['rotation'][:, axis]
                changed['opacity'][indices_local] = parameters['opacity'][:, 0]
                path = dest/f'refine_{step:03d}.ply'
                save_ply(path, changed)
                candidate, _ = scene.load_model(path)
                metrics = evaluate(scene, candidate, dest/f'refine_{step:03d}', output/'comparison/initial', lp)
                accepted, gain, reasons = select(metrics, best['metrics'], initial, config)
                del candidate
                if accepted:
                    best.update(rows=changed, metrics=metrics, accepted=True, step=step)
                item = dict(round=iteration+1, action='refine', step=step, mixed=len(indices_local),
                    accepted=accepted, relative_gain=gain, reasons=reasons, changed_ids=ids[indices_local].tolist(),
                    metrics=str((dest/f'refine_{step:03d}'/'metrics.json').resolve()))
                edits.append(item)
                return dict(accepted=accepted, relative_gain=gain, reasons=reasons)
            history = refine(scene, model, mixed, indices, config, checkpoint)
            write_json(dest/'refinement_history.json', history)
            if best['accepted']:
                np.savez(dest/'accepted_parameter_changes.npz', original_ids=ids[mixed],
                    old=base_rows[mixed], new=best['rows'][mixed])
                rows, current = best['rows'], best['metrics']
                save_ply(dest/'best_refined.ply', rows)
                model, _ = scene.load_model(dest/'best_refined.ply')
                accepted_round = True
        write_json(output/'edit_manifest.json', dict(edits=edits, original_model=spec['model'], input_sha256=original_sha,
            original_gaussians=len(original_rows), current_gaussians=len(rows)))
        save_ply(output/'best_so_far.ply', rows)
        np.save(output/'original_gaussian_ids.npy', ids)
        if not accepted_round:
            stop = 'no_feasible_improving_candidate'
            break
    save_ply(output/'cleaned_gaussians.ply', rows)
    viewer, baking = bake_viewer(rows)
    save_ply(output/'cleaned_gaussians_supersplat.ply', viewer)
    removed_mask = np.ones(len(original_rows), bool)
    removed_mask[ids] = False
    save_ply(output/'removed_gaussians.ply', original_rows[removed_mask])
    np.save(output/'removed_original_gaussian_ids.npy', np.flatnonzero(removed_mask))
    final = evaluate(scene, model, output/'comparison/final', output/'comparison/initial', lp, save_images=True)
    final_free = evaluate_free(scene, model, surface, output/'comparison/final/free')
    # Final native-viewer rendering parity validates scale, SH, rotations AND effective alpha.
    viewer_model, _ = scene.load_model(output/'cleaned_gaussians_supersplat.ply')
    parity = []
    for index in indices[::max(1, len(indices)//4)][:4]:
        c, *_ = scene.get_view(index)
        with torch.no_grad():
            a, b = scene.render(c, model), scene.render(c, viewer_model)
        parity.append(dict(index=index, rgb_max_abs=float((a['render']-b['render']).abs().max()),
            alpha_max_abs=float((a['mask']-b['mask']).abs().max())))
    parity_passed = all(r['rgb_max_abs'] < 1/255 and r['alpha_max_abs'] < 1/255 for r in parity)
    baking.update(render_parity=parity, parity_passed=parity_passed)
    write_json(output/'viewer_baking.json', baking)
    if not parity_passed:
        raise ValueError('Native/SuperSplat render parity failed')
    bg_before, bg_after = initial['mean']['bg_alpha_mass_per_fg'], final['mean']['bg_alpha_mass_per_fg']
    gain = (bg_before-bg_after)/max(bg_before, 1e-12)
    outcome = 'improved_with_residuals' if gain > config['minimum_relative_gain'] else 'no_feasible_gain'
    alarms = dict(large_background_cloud='proxy_alarm' if final['mean']['largest_bg_component_fraction'] > .002 else 'not_detected_by_proxy',
        leaf_coverage='proxy_alarm' if final['mean']['fg_coverage'] < initial['mean']['fg_coverage']-.001 else 'not_detected_by_proxy',
        leaf_tip_boundary='proxy_alarm' if final['mean']['boundary_coverage'] < initial['mean']['boundary_coverage']-.002 else 'not_detected_by_proxy',
        thin_petiole='proxy_alarm' if final['mean']['thin_coverage'] < initial['mean']['thin_coverage']-.001 else 'not_detected_by_proxy',
        depth_anomaly='proxy_alarm' if final['mean']['depth_changed_fraction'] > config['quality']['depth_changed_fraction'] else 'not_detected_by_proxy',
        pot_neighborhood='not_assessable_without_semantic_pot_labels',
        dark_leaf_misclassification='color_not_used_as_pruning_rule; inspect_measured_dark_foreground_per_view',
        black_cloud='proxy_alarm' if final['mean']['dark_bg_high_alpha_fraction'] > .002 else 'not_detected_by_proxy')
    result = dict(plant=spec['plant'], original_gaussians=len(original_rows), cleaned_gaussians=len(rows),
        removed_gaussians=int(removed_mask.sum()), artifact_reduction=gain,
        foreground_preservation=final['mean']['fg_alpha']/max(initial['mean']['fg_alpha'], 1e-12),
        initial=initial, final=final, initial_free=initial_free, final_free=final_free,
        outcome=outcome, stop_reason=stop, rounds=iteration+1,
        seconds=time.perf_counter()-begin, peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2, physical_gpu=1,
        native_ply=str(output/'cleaned_gaussians.ply'), supersplat_ply=str(output/'cleaned_gaussians_supersplat.ply'),
        source_ply=str(Path(spec['model']).resolve()), native_sha256=sha(output/'cleaned_gaussians.ply'),
        automatic_failure_proxies=alarms,
        viewer_sha256=sha(output/'cleaned_gaussians_supersplat.ply'), semantic_failures_status='Only binary mask/depth proxies measured; leaf/pot/petiole-specific ground truth unavailable',
        historical_baselines=spec.get('baseline_models', {}))
    write_json(output/'cleanup_metrics.json', result)
    (output/'AUTO_CLEAN_REPORT.md').write_text(f'# {spec["plant"]} automatic GGGS cleaning\n\n'
        f'Outcome: **{outcome}**; stop: {stop}.\n\n'
        f'Gaussians: {len(original_rows)} → {len(rows)}. Measured training-view background-alpha reduction {gain:.2%}; '
        f'foreground alpha ratio {result["foreground_preservation"]:.6f}.\n\n'
        f'FG PSNR {initial["mean"]["fg_psnr"]:.4f} → {final["mean"]["fg_psnr"]:.4f}; '
        f'boundary F1 {initial["mean"]["boundary_f1"]:.6f} → {final["mean"]["boundary_f1"]:.6f}; '
        f'thin proxy coverage {initial["mean"]["thin_coverage"]:.6f} → {final["mean"]["thin_coverage"]:.6f}.\n\n'
        f'Native: `{result["native_ply"]}`\n\nViewer: `{result["supersplat_ply"]}`\n\n'
        'Free views use external depth geometry proxies; they do not constitute ground truth. '
        'Dark color is diagnostic only. Binary masks cannot independently certify leaf/pot/petiole classes. '
        'All rejected candidates and current-model contributions are retained in rounds/. '
        'See cleanup_metrics.json for all-camera measurements and edit_manifest.json for IDs and decisions.\n')
    write_json(output/'COMPLETE.json', dict(status='complete', input_sha256=original_sha, config=config,
        native_sha256=result['native_sha256'], scene=spec))
    return result
