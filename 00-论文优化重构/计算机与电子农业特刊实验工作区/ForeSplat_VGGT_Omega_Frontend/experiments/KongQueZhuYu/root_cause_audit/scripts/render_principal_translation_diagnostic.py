#!/usr/bin/env python3
"""CPU image-translation diagnostic on recorded checkpoint renders, with GT fixed.

This approximates a principal-point shift in image space. It does not change a
camera, invoke a rasterizer, retrain, or establish a benefit of retraining.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy import ndimage

AUDIT = Path(__file__).resolve().parents[1]
SOURCES = [0, 19, 38, 57, 76, 95, 114, 133, 152, 170, 189, 209]


def psnr(image, gt, selected):
    mse = float(np.mean((image[selected]-gt[selected])**2, dtype=np.float64))
    return float(-10*np.log10(mse)) if mse > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-root', type=Path, default=AUDIT/'render_error_analysis/checkpoint_rerender_v1')
    parser.add_argument('--output', type=Path, default=AUDIT/'render_error_analysis/principal_translation_v1')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Preserve previous diagnostic: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    shifts = {'E1': {'train_saved_principal': [1.40625, 1.3953488372093]},
              'E5': {'train_saved_principal': [.25, .25],
                     'true_PIL_pixel_transform': [1.8532818532818252]*2},
              'E6': {'train_saved_principal': [.25, .25],
                     'true_PIL_pixel_transform': [1.8532818532818252]*2}}
    rows = []
    for experiment, scenarios in shifts.items():
        for source in SOURCES:
            with np.load(args.render_root/experiment/'views'/f'crop_{source:04d}'/'raw_render_outputs.npz') as raw:
                image = raw['render_rgb']; gt = raw['gt_rgb']; fg = raw['foreground_mask']
            boundary = ndimage.binary_dilation(fg, iterations=5)^ndimage.binary_erosion(fg, iterations=5)
            regions = {'full_psnr': np.ones_like(fg), 'fg_mask_psnr': fg,
                       'boundary_band5_psnr': boundary, 'inner_fg_edge5_psnr': fg&boundary}
            baseline = {key: psnr(image, gt, region) for key, region in regions.items()}
            for scenario, (dx, dy) in scenarios.items():
                moved = ndimage.shift(image, (dy, dx, 0), order=1, mode='constant', cval=0, prefilter=False)
                row = dict(experiment=experiment, source_id=source, scenario=scenario,
                           delta_cx=dx, delta_cy=dy, changed_camera=False, new_renderer_call=False)
                for key, region in regions.items():
                    after = psnr(moved, gt, region)
                    row[key+'_baseline'] = baseline[key]; row[key+'_translated'] = after
                    row[key+'_delta'] = after-baseline[key]
                rows.append(row)
    with (args.output/'per_view_translation_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = []
    for experiment, scenarios in shifts.items():
        for scenario in scenarios:
            selected = [r for r in rows if r['experiment']==experiment and r['scenario']==scenario]
            row = dict(experiment=experiment, scenario=scenario, views=len(selected))
            for key in selected[0]:
                if key.endswith(('_baseline', '_translated', '_delta')):
                    row[key] = float(np.mean([r[key] for r in selected]))
            summaries.append(row)
    with (args.output/'summary_translation_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    settings = dict(scope='CPU image-domain translation approximation; existing real RGB render only',
        delta_definition='desired K principal point minus runtime centered principal point, positive right/down',
        ground_truth_and_mask_unchanged=True, interpolation='bilinear, zero/black outside image; no wraparound',
        source_ids=SOURCES, shifts=shifts,
        delta_source='camera_projection_comparison/intrinsic_chain_per_frame.csv, parent verified constants across all208',
        interpretation='Measures sensitivity to this small principal-point translation only. Cannot establish shader-level or retraining gains; does not exclude other K/FoV errors.',
        summaries=summaries)
    (args.output/'translation_diagnostic.json').write_text(json.dumps(settings, indent=2, allow_nan=False))
    print('COMPLETE CPU principal-point image translation diagnostic')


if __name__ == '__main__':
    main()
