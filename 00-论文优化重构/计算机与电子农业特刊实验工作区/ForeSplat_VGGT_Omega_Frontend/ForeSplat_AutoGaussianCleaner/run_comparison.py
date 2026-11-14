#!/usr/bin/env python3
"""C0–C4 independent GGGS baselines. Run from the frontend uv environment."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import sys
import argparse
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import yaml
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply, write_json, sha
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.legacy import features, rules
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.artifacts import export, parity, visuals
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', default=['C0','C1_raw','C1_adaptive','C2_raw','C2_adaptive','C3','C4'])
    a = p.parse_args()
    base = Path(__file__).resolve().parent
    config = yaml.safe_load((base/'configs/default.yaml').read_text())
    spec = json.loads(a.scene.read_text())
    sel = Path(spec['baseline_models']['R5']).parents[3]
    spec.update(historical_free_protocol=str(sel/'comparison/shared_free_views.json'),
                historical_free_silhouettes=str(sel/'gaussian_mesh_artifact_refinement/task0_gaussian_cleaning/diagnosis/free_contribution_views'))
    original = read_ply(spec['model'])
    legacy = base/'experiments/all_19_plants_v3/development/KongQueZhuYu/C'
    ref = json.loads((legacy/'original_reference.json').read_text())
    if ref['input_model_sha256'] != sha(spec['model']) or ref['config']['evaluation_max_side'] != config['evaluation_max_side']:
        raise ValueError('Cached original evaluation does not match input/protocol')
    scene = Scene(spec, config)
    torch.set_num_threads(8)
    surface = external_surface(scene, scene.evidence_indices())
    import lpips
    lp = lpips.LPIPS(net='alex').cuda().eval()
    f = None
    for method in a.methods:
        dest = a.output/method
        if (dest/'COMPLETE.json').exists():
            print('reuse completed', method, flush=True)
            continue
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        ids = np.arange(len(original))
        rows = original
        metadata = dict(method=method, input_model=spec['model'], input_sha256=sha(spec['model']),
                        evaluation_baseline=str(legacy/'comparison/initial'), scene=spec, config=config)
        if method.startswith(('C1', 'C2')):
            if f is None:
                f = features(original)
            removed, bits, detail = rules(f, 2 if method.startswith('C1') else 3, method.endswith('adaptive'))
            ids = ids[~removed]
            rows = original[~removed]
            metadata.update(detail)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez(dest/'deletion_reasons.npz', original_ids=np.arange(len(original)), reason_bits=bits, removed=removed)
        elif method in ['C3','C3_pre_marker']:
            hist = Path(spec['baseline_models']['R5'])
            if method=='C3_pre_marker':
                hist=hist.parent/'historical_scene_gaussians.ply'
                ids_path=hist.parent.parent/'additional_rounds/round_05/original_gaussian_ids.npy'
            else:
                ids_path=hist.parent/'original_gaussian_ids.npy'
            rows = read_ply(hist)
            ids = np.load(ids_path)
            if len(rows)!=len(ids) or not all(np.array_equal(rows[k],original[k][ids]) for k in rows.dtype.names):
                raise ValueError('Historical retained model must preserve exact original native row identity')
            metadata.update(historical_reuse=str(hist), historical_manifest=json.loads((hist.parent/'best_manifest.json').read_text()),
                            warning=('C3 includes historical manual marker exclusion, separately disclosed; no new semantic exclusion' if method=='C3' else
                                'Historical round5 before 3592-point marker exclusion; secondary contrast separates background cleaning from semantic removal'))
        elif method == 'C4':
            rows = read_ply(legacy/'cleaned_gaussians.ply')
            ids = np.load(legacy/'original_gaussian_ids.npy')
            metadata.update(reused_run=str(legacy), historical_seconds=json.loads((legacy/'cleanup_metrics.json').read_text())['seconds'],
                            modifications=0, status='no_feasible_gain')
        export(dest, original, rows, ids)
        model, _ = scene.load_model(dest/'cleaned_gaussians.ply')
        metrics = evaluate(scene, model, dest/'evaluation', legacy/'comparison/initial', lp, save_images=True)
        visuals(scene, model, dest/'visualizations', surface)
        parity(scene, model, dest)
        if f is not None and method.startswith(('C1', 'C2')):
            with np.load(legacy/'rounds/01/evidence.npz') as z:
                fg, bg = z['fg'], z['bg']
            metadata['removed_native_contribution'] = dict(foreground_fraction=float(fg[removed].sum()/fg.sum()),
                background_fraction=float(bg[removed].sum()/bg.sum()),
                warning='Initial T*alpha mass; rerender metrics measure actual gain after exposed layers')
            metadata['per_rule_native_contribution'] = {name:dict(count=int(((bits & (1 << i)) != 0).sum()),
                fg=float(fg[(bits & (1 << i)) != 0].sum()), bg=float(bg[(bits & (1 << i)) != 0].sum()))
                for i, name in enumerate(metadata['reasons'])}
        metadata.update(original_gaussians=len(original), cleaned_gaussians=len(rows), removed_gaussians=len(original)-len(rows),
            seconds=time.perf_counter()-begin, peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            metrics=metrics, native_ply=str((dest/'cleaned_gaussians.ply').resolve()),
            supersplat_ply=str((dest/'cleaned_gaussians_supersplat.ply').resolve()),
            removed_ply=str((dest/'removed_gaussians.ply').resolve()))
        write_json(dest/'cleanup_metrics.json', metadata)
        write_json(dest/'COMPLETE.json', dict(input_sha256=metadata['input_sha256'],native_sha256=sha(dest/'cleaned_gaussians.ply')))
        print(method, 'removed', len(original)-len(rows), 'seconds', metadata['seconds'], flush=True)
        del model


if __name__ == '__main__':
    main()
