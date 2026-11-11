#!/usr/bin/env python3
"""Wait for the serialized batch before running the independent legacy baseline audit."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--driver-pid', type=int, required=True)
    p.add_argument('--batch-root', type=Path, required=True)
    a = p.parse_args()
    while Path(f'/proc/{a.driver_pid}/stat').exists():
        # /proc names may contain spaces inside parentheses.
        state = Path(f'/proc/{a.driver_pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        if state == 'Z':
            break
        time.sleep(30)
    root = Path(__file__).resolve().parents[2]
    output = a.batch_root/'development/KongQueZhuYu/R5_auto_C'
    reports = a.batch_root/'reports'
    audit = reports/'legacy_r5'
    if not (output/'COMPLETE.json').exists():
        reports.mkdir(exist_ok=True)
        (reports/'LEGACY_R5_AUDIT_STATUS.json').write_text(json.dumps(dict(status='unavailable', reason='Automatic R5 result not complete; see main batch_status.json'), indent=2))
        return
    command = ['uv', 'run', '--no-project', '--python', str(root/'.venv/bin/python'), 'python',
        str(Path(__file__).with_name('legacy_r5_audit.py')), '--result', str(output), '--output', str(audit)]
    code = subprocess.run(command, cwd=root, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'}).returncode
    (reports/'LEGACY_R5_AUDIT_STATUS.json').write_text(json.dumps(dict(status='complete' if code == 0 else 'failed', returncode=code, command=command), indent=2))
    if code == 0:
        result = json.loads((audit/'LEGACY_R5_RESIDUAL_COMPARISON.json').read_text())
        before, after = result['models']['R5'], result['models']['R5_auto_C']
        with (reports/'FINAL_AUTOMATIC_GAUSSIAN_CLEANING_REPORT.md').open('a') as f:
            f.write('\n## R5 历史协议独立残余核对\n\n'
                f'重新实测固定八相机黑像素 {before["black_high_alpha_pixels_sum"]} → {after["black_high_alpha_pixels_sum"]}，'
                f'黑贡献>.05 Gaussian {before["black_weight_gt005_gaussians"]} → {after["black_weight_gt005_gaussians"]}。'
                f'黑像素净减少 {result["black_pixel_reduction"]:.2%}。这组历史相机/P3轮廓只用于旧基线可比性，'
                '没有用于通用清理或选优。完整文件：legacy_r5/LEGACY_R5_RESIDUAL_COMPARISON.json。\n')


if __name__ == '__main__':
    main()
