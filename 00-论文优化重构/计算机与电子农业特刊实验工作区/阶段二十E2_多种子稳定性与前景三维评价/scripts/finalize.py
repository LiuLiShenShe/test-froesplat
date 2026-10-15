"""Run the frozen evaluation sequence only after all real trainings validate."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from common import GPU_INDEX, GPU_UUID, PYTHON, ROOT, STAGE, now, read_json, sha256, write_json


def training_complete():
    active_path = STAGE / 'reports/ACTIVE_TRAINING.json'
    progress_path = STAGE / 'reports/MATRIX_PROGRESS.json'
    try:
        active = read_json(active_path)
        progress = read_json(progress_path)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (active.get('status') == 'ALL_EIGHT_COMPLETED'
            and progress.get('completed') == 8
            and len(progress.get('runs', [])) == 8
            and all(row['completed'] for row in progress['runs']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    report_path = STAGE / 'reports/FINALIZATION_PROGRESS.json'
    assert not report_path.exists(), 'Retain earlier finalization evidence; recover failed steps explicitly'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_INDEX, OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
    report = {'status': 'WAITING_FOR_REAL_TRAININGS', 'start_time': now(),
              'GPU_UUID': GPU_UUID, 'steps': [], 'no_training_commands': True,
              'script_sha256': sha256(__file__)}
    write_json(report_path, report)
    while not training_complete():
        assert args.wait_for_training, 'Eight completed and reloadable models are required'
        time.sleep(30)
    print('ALL_EIGHT_VALIDATED: starting common-reference evaluation', flush=True)
    sequence = [('evaluate.py', ['--all']), ('summarize.py', []),
                ('recompute_metrics.py', []), ('make_figures.py', []),
                ('acceptance.py', []), ('write_final_reports.py', [])]
    for script, arguments in sequence:
        command = [str(PYTHON), str(STAGE / 'scripts' / script), *arguments]
        log_path = STAGE / 'logs' / f'FINALIZE_{script.removesuffix(".py")}.txt'
        row = {'script': script, 'command_argv': command, 'start_time': now(),
               'script_sha256': sha256(STAGE / 'scripts' / script),
               'log_path': str(log_path), 'status': 'RUNNING'}
        report['steps'].append(row)
        report['status'] = 'RUNNING'
        write_json(report_path, report)
        print('START_FINALIZATION_STEP', script, now(), flush=True)
        started = time.perf_counter()
        with log_path.open('xb') as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT)
        row.update({'end_time': now(), 'wall_seconds': time.perf_counter() - started,
                    'exit_code': result.returncode, 'log_sha256': sha256(log_path),
                    'status': 'PASS' if result.returncode == 0 else 'FAILED'})
        if result.returncode:
            report['status'] = 'FAILED'
            write_json(report_path, report)
            raise RuntimeError(f'Preserved failed step {script}: {log_path}')
        write_json(report_path, report)
        print('COMPLETED_FINALIZATION_STEP', script, now(), flush=True)
    report.update({'status': 'PASS', 'end_time': now()})
    write_json(report_path, report)
    print('EVALUATION_AND_ACCEPTANCE_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
