#!/usr/bin/env python3
"""Re-evaluate historical models under the new common protocol, without editing them."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import yaml
import torch
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json
from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import evaluate
from ForeSplat_AutoGaussianCleaner.evaluation.free_view_evaluation import evaluate_free


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/default.yaml')
    a = p.parse_args()
    torch.set_num_threads(8)
    spec = json.loads(a.scene.read_text())
    scene = Scene(spec, yaml.safe_load(a.config.read_text()))
    import lpips
    lp = lpips.LPIPS(net='alex').cuda().eval()
    surface = external_surface(scene, scene.evidence_indices())
    models = {'Original': spec['model'], **spec.get('baseline_models', {})}
    results = {}
    for name, path in models.items():
        model, rows = scene.load_model(path)
        metrics = evaluate(scene, model, a.output/name, None if name == 'Original' else a.output/'Original', lp, save_images=True)
        free = evaluate_free(scene, model, surface, a.output/name/'free')
        results[name] = dict(model=path, gaussians=len(rows), metrics=metrics, free=free)
    write_json(a.output/'baseline_comparison.json', results)


if __name__ == '__main__':
    main()
