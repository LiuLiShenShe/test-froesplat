#!/usr/bin/env python3
"""Secondary historical-protocol audit only; never used by cleaning/selection.

The general cleaner uses automatically generated free cameras. This independent
audit retains the old eight views solely to quantify R5's 4011-point residual
under a comparable protocol, without feeding IDs/ROIs back into the algorithm.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    history = root/'experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/gaussian_mesh_artifact_refinement'
    sys.path.insert(0, str(history/'scripts'))
    from common import setup, cameras, mask, free_camera, SEL
    from artifact_evidence import contribution
    from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json
    Model, native = setup()
    from utils.sh_utils import eval_sh
    torch.set_num_threads(8)
    baseline = history/'task0_gaussian_cleaning/best_cleaned'
    source_ids = np.load(baseline/'original_gaussian_ids.npy')
    current_indices = np.load(a.result/'original_gaussian_ids.npy')
    models = [('R5', baseline/'cleaned_gaussians.ply', source_ids),
        ('R5_auto_C', a.result/'cleaned_gaussians.ply', source_ids[current_indices])]
    training, refs = cameras()
    protocol = json.loads((SEL/'comparison/shared_free_views.json').read_text())
    original_fg = np.load(history/'task0_gaussian_cleaning/diagnosis/actual_contribution_evidence.npz')['fg_contribution']
    pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    summaries = {}
    a.output.mkdir(parents=True, exist_ok=True)
    for label, path, ids in models:
        model = Model(3, 0); model.load_ply(str(path))
        fg = np.zeros(len(ids)); bg = np.zeros(len(ids)); black = np.zeros(len(ids)); dark = np.zeros(len(ids))
        views = []
        for c, ref in zip(training, refs):
            if ref['source_id'] not in [0, 26, 53, 79, 105, 131, 157, 183]:
                continue
            m = mask(c); outside = ndimage.distance_transform_edt(~m) > 3
            g, _, _ = contribution(c, model, np.stack([m, outside, outside]))
            fg += g[:, 0]/8; bg += g[:, 1]/8
        for row in protocol['views']:
            c = free_camera(row)
            with np.load(history/'task0_gaussian_cleaning/diagnosis/free_contribution_views'/f'{row["name"]}.npz') as z:
                reference = z['reference_mesh_silhouette'].copy()
            outside = ndimage.distance_transform_edt(~reference) > 5
            with torch.no_grad():
                out = native.render(c, model, pipe, torch.full((3,), protocol['background'], device='cuda'), 0.)
                rgb = out['render'].permute(1, 2, 0).cpu().numpy()
                alpha = out['mask'].squeeze().cpu().numpy()
                dirs = torch.nn.functional.normalize(model.get_xyz-c.camera_center, dim=1)
                color = (eval_sh(3, model.get_features.transpose(1, 2), dirs)+.5).clamp_min(0).cpu().numpy()
            cloud = outside & (rgb @ [.2126, .7152, .0722] < .12)
            g, _, _ = contribution(c, model, np.stack([reference, outside, cloud]))
            bg += g[:, 1]/8; black += g[:, 2]/8
            dark += g[:, 2] * (color @ [.2126, .7152, .0722] < .12)/8
            views.append(dict(name=row['name'], black_high_alpha_pixels=int((cloud & (alpha > .5)).sum()), black_weight=float(g[:, 2].sum())))
        fg = np.maximum(fg, original_fg[ids]/208)
        harm = bg/np.maximum(fg+bg, 1e-12)
        np.savez(a.output/f'{label}_legacy_evidence.npz', original_e10_ids=ids, fg=fg, bg=bg, black=black, dark_sh=dark, harm=harm)
        summaries[label] = dict(model=str(path), gaussians=len(ids), black_weight_gt005_gaussians=int((black > .05).sum()),
            black_high_alpha_pixels_sum=sum(v['black_high_alpha_pixels'] for v in views),
            black_weight_per_view_sum=float(black.sum()), mixed_black_weight_fraction=float(black[harm <= .95].sum()/max(black.sum(), 1e-12)), views=views)
    before, after = summaries['R5'], summaries['R5_auto_C']
    write_json(a.output/'LEGACY_R5_RESIDUAL_COMPARISON.json', dict(protocol='Historical fixed8/P3 silhouette, comparison only, not general-method evidence or selection',
        baseline_claim='Prior 4011 black contributors, 68362 black high-alpha pixels; this audit remeasures both models', models=summaries,
        black_pixel_reduction=(before['black_high_alpha_pixels_sum']-after['black_high_alpha_pixels_sum'])/max(before['black_high_alpha_pixels_sum'], 1),
        contributor_count_reduction=before['black_weight_gt005_gaussians']-after['black_weight_gt005_gaussians']))


if __name__ == '__main__':
    main()
