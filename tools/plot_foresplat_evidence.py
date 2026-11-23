#!/usr/bin/env python3
"""Draw from archived numeric tables, depth arrays or RGB pairs; no inference."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image


TRAITS = (('base_to_tip_chord', 'Base-to-tip chord'),
          ('midsection_transverse_width', 'Midsection width'))
COLORS = {'Beefsteak': '#0072B2', 'Big Boy Hybrid': '#D55E00', 'Celebrity Hybrid': '#009E73'}
RGB_RESIDUAL_MAX = 64


def depth_preview(src, output):
    with np.load(src, allow_pickle=False) as data:
        depth = data['depth']
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError(f'No finite positive depth: {src}')
    low, high = np.quantile(depth[valid], [0.02, 0.98])
    values = np.zeros(depth.shape)
    values[valid] = np.clip((depth[valid] - low) / max(high-low, 1e-9), 0, 1)
    colors = matplotlib.colormaps['viridis'](values, bytes=True)[..., :3]
    colors[~valid] = 255
    Image.fromarray(colors).save(output/'Rendered_Depth.png')
    Image.fromarray(valid.astype(np.uint8)*255).save(output/'Valid_Depth_Mask.png')
    (output/'Depth_Display_Protocol.json').write_text(json.dumps(dict(source_key='depth', valid='finite and > 0',
        display_quantile_2=float(low), display_quantile_98=float(high), colors='viridis; invalid white'), indent=2)+'\n')


def rgb_residual_preview(reference, prediction, output):
    with Image.open(reference) as target, Image.open(prediction) as rendered:
        if target.size != rendered.size:
            raise ValueError(f'Mismatched RGB pair dimensions: {reference}, {prediction}')
        difference = np.asarray(rendered.convert('RGB'), dtype=np.int16) - np.asarray(target.convert('RGB'), dtype=np.int16)
    residual = np.abs(difference).mean(axis=2)
    colors = matplotlib.colormaps['magma'](np.clip(residual/RGB_RESIDUAL_MAX, 0, 1), bytes=True)[..., :3]
    Image.fromarray(colors).save(output)


def plot_pairs(csv_path, output):
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        rows = [row for row in csv.DictReader(f) if row['status'] == 'included_reference_guided_landmarks']
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3), layout='constrained')
    for i, (key, name) in enumerate(TRAITS):
        xy = [(r['cultivar'], float(r['reference']), float(r['model'])) for r in rows if r['trait'] == key]
        lim = 1.10 * max(max(x, y) for _, x, y in xy)
        scatter, residual = axes[0, i], axes[1, i]
        scatter.plot([0, lim], [0, lim], '--', c='0.42', lw=0.9, zorder=0)
        residual.axhline(0, ls='--', c='0.42', lw=0.9, zorder=0)
        for cultivar, color in COLORS.items():
            values = [(x,y) for c,x,y in xy if c==cultivar]
            if not values:
                continue
            x, y = np.array(values).T
            scatter.scatter(x, y, s=22, c=color, label=cultivar)
            residual.scatter(x, y-x, s=22, c=color)
        scatter.set(title=f'{name} (n={len(xy)})', xlim=(0, lim), ylim=(0, lim), ylabel='ForeSplat (cm)')
        scatter.set_aspect('equal')
        residual.set(xlim=(0, lim), xlabel='Reference (cm)', ylabel='Mesh - reference (cm)')
        scatter.spines[['right','top']].set_visible(False)
        residual.spines[['right','top']].set_visible(False)
    axes[0, 1].legend(loc='upper left', frameon=False, fontsize=7)
    fig.suptitle('Matched compound leaves: reference-guided landmarks', fontsize=12)
    for suffix in ('png','pdf','svg'):
        fig.savefig(output/f'Tomato_reference_guided_leaf_agreement.{suffix}', dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pairs', type=Path)
    mode.add_argument('--depth', type=Path)
    mode.add_argument('--rgb-residual', type=Path, nargs=2, metavar=('REFERENCE', 'PREDICTION'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.depth:
        depth_preview(args.depth, args.output)
    elif args.rgb_residual:
        rgb_residual_preview(*args.rgb_residual, args.output/'RGB_Absolute_Error.png')
    else:
        plot_pairs(args.pairs, args.output)


if __name__ == '__main__':
    main()
