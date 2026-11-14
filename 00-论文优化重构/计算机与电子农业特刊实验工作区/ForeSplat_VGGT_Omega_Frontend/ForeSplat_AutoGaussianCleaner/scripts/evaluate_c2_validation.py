"""Evaluate only the three completed C2_raw plants, serially on physical GPU1."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLANTS = ['ChangShouHua1', 'DouBanLv1', 'WangWenCao1']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plants', nargs='+', choices=PLANTS, default=PLANTS)
    plants = parser.parse_args().plants
    result_root = ROOT/'experiments/C2_fast_validation'
    report_root = ROOT/'reports/C2_fast_validation'
    anchor = json.loads((result_root/'KongQueZhuYu_equivalence/cleanup_report.json').read_text())
    assert anchor['remaining_gaussians'] == 428967
    assert anchor['historical_verification']['identical_ids']
    for plant in plants:
        report = json.loads((result_root/plant/'cleanup_report.json').read_text())
        assert report['algorithm'] == 'legacy.rules(version=3, normalized=False)'
        assert (result_root/plant/'cleaned_gaussians.ply').is_file()
        if (report_root/plant/'paired_quality.json').exists():
            raise FileExistsError('Preserve existing evaluation: '+plant)
    timing_file = report_root/'evaluation_wall_times.json'
    timings = json.loads(timing_file.read_text()) if timing_file.exists() else {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
    for plant in plants:
        manifest = report_root/(plant+'_models.json')
        manifest.write_text(json.dumps({'C2_raw': str(result_root/plant/'cleaned_gaussians.ply')}, indent=2)+'\n')
        print('Evaluating', plant, 'on physical GPU1', flush=True)
        started = time.perf_counter()
        with (report_root/(plant+'_evaluation.log')).open('x') as log:
            subprocess.run([sys.executable, '-m', 'fast_ply.evaluate',
                '--scene', str(ROOT/'configs/scenes'/(plant+'.json')),
                '--external-models', str(manifest), '--skip-baselines', '--skip-ablations',
                '--views', '16', '--max-side', '960', '--output', str(report_root/plant)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        timings[plant] = time.perf_counter()-started
        (report_root/'evaluation_wall_times.json').write_text(json.dumps(timings, indent=2)+'\n')
        print('Completed', plant, 'evaluation:', round(timings[plant], 2), 's', flush=True)


if __name__ == '__main__':
    main()
