#!/usr/bin/env python3
"""Check a browser build with actual standard-PLY known-opacity splats."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply, save_ply, write_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--editor-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    labels = []
    source = read_ply(a.template)[:1]
    for alpha in [.004, .02, .5]:
        rows = np.zeros(3, dtype=source.dtype)
        rows['z'] = 1
        rows['rot_0'] = 1
        rows['opacity'] = np.log(alpha / (1 - alpha))
        for k in range(3):
            rows[f'scale_{k}'] = np.log(.06)
            rows[f'f_dc_{k}'] = -.5 / .28209479177387814
        # Two invisible anchors avoid a zero-extent scene clipping plane.
        rows['x'][1:] = [-.4,.4]
        rows['z'][1:] = [.8,1.2]
        rows['opacity'][1:] = -25
        for k in range(3):
            rows[f'scale_{k}'][1:] = np.log(.001)
        label = f'alpha_{alpha:g}'
        labels.append(label)
        save_ply(root/'comparison/models'/f'{label}_supersplat.ply', rows)
    write_json(root/'stage2_residual/visualizations/automatic_free_views.json', dict(
        protocol='known_standard_ply_opacity_conformance_v1', definitions=[dict(
            name='center', E=np.eye(4), K=[[600,0,480],[0,600,360],[0,0,1]], width=960, height=720)]))
    subprocess.run([sys.executable, str(Path(__file__).with_name('review_historical_supersplat.py')),
        '--experiment', str(root), '--editor-source', str(a.editor_source), '--models', *labels], check=True)
    observations = []
    for alpha, label in zip([.004,.02,.5], labels):
        folder = root/'comparison/official_supersplat'/label
        white = np.asarray(Image.open(folder/'center_white.png').convert('RGB'), dtype=float)
        black = np.asarray(Image.open(folder/'center_black.png').convert('RGB'), dtype=float)
        # Ignore editor chrome. Dark splat on white measures composited alpha.
        region = (slice(430,620),slice(600,840))
        attenuation = 1 - white[region].mean(axis=-1)/255
        observations.append(dict(input_alpha=alpha, white_attenuation_max=float(attenuation.max()),
            black_max=float(black[region].max()), standard_ply_sha256=sha(root/'comparison/models'/f'{label}_supersplat.ply')))
    passed = all(abs(m['white_attenuation_max']-m['input_alpha']) <= 2/255
        and m['black_max'] <= 1 for m in observations)
    result = dict(passed=passed, editor_build_sha256=sha(a.editor_source/'dist/index.js'),
        measurements=observations, absolute_alpha_tolerance=2/255,
        method='Actual browser screenshots; one known-alpha standard-PLY splat plus two invisible scene-bound anchors per fixture; center white attenuation',
        gaussian_model_workaround=False)
    write_json(root/'OPACITY_CONFORMANCE.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
