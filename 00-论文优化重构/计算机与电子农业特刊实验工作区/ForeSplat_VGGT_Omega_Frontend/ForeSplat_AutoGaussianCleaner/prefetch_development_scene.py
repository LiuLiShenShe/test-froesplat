#!/usr/bin/env python3
"""Run one frozen development scene ahead of the serial batch, then expose its verified output.

The batch's unchanged --resume path rechecks the result when it reaches that
plant. Independent plants and incomplete outputs are never exposed this way.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha, write_json
from ForeSplat_AutoGaussianCleaner.execution_provenance import (
    completed_identity, resolve_variant, scene_input_fingerprint, native_renderer_fingerprint)


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plant', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--batch-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    batch, source = a.batch_root.resolve(), a.source_root.resolve()
    frozen = read(batch/'METHOD_FREEZE.json')
    manifest = read(a.manifest)
    job = next(j for j in manifest['plants'] if j['plant']==a.plant)
    if job['group'] != 'development':
        raise ValueError('Prefetch is restricted to the frozen development set')
    if sha(a.manifest) != frozen['manifest_sha256'] or sha(a.config) != frozen['config_sha256']:
        raise ValueError('Frozen development manifest/configuration changed')
    spec = read(Path(job['scene']))
    def context_intact():
        renderer = native_renderer_fingerprint([spec['gggs_repo']])
        return (sha(a.manifest)==frozen['manifest_sha256']
            and sha(a.config)==frozen['config_sha256']
            and sha(job['scene'])==frozen['scenes_sha256'][a.plant]
            and scene_input_fingerprint(spec)==frozen['scene_contents'][a.plant]
            and bool(renderer)
            and all(frozen['native_renderer_sources'].get(k)==v for k,v in renderer.items()))
    def intact():
        return all(sha(source/relative)==value for relative,value in frozen['sources'].items())
    if not intact() or not context_intact():
        raise ValueError('Prefetch source/scene/native renderer differs from the running frozen batch')
    expected = batch/'development'/a.plant
    if expected.exists() or expected.is_symlink():
        raise FileExistsError('Primary batch has already claimed this scene')
    output = batch/'prefetched_development'/a.plant
    output.parent.mkdir(exist_ok=True)
    lock = (output.parent/(a.plant+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if output.exists():
        raise FileExistsError('Choose a fresh prefetch output; do not replace an existing result')
    command = ['uv', 'run', '--no-project', '--python', sys.executable, 'python', '-u',
        str(source/'auto_clean_gaussians.py'), '--scene', job['scene'], '--method', frozen['method'],
        '--config', str(a.config.resolve()), '--output', str(output)]
    schedule = output.parent/(a.plant+'_SCHEDULE.json')
    record = dict(plant=a.plant, command=command, output=str(output), expected_batch_output=str(expected),
        physical_gpu=1, scope='One external development scene overlaps the serial development worker; at most two cleaner processes. Sources, data and thresholds are unchanged. Primary batch resume performs final admission.',
        started=time.time(), state='starting')
    write_json(schedule, record)
    with (output.parent/(a.plant+'.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=source.parent,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES':'1'}, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        record.update(state='running', producer_pid=process.pid); write_json(schedule, record)
        while process.poll() is None:
            if expected.exists() or expected.is_symlink():
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                if output.exists():
                    write_json(output/'INTERRUPTED.json', dict(
                        reason='Prefetch cancelled because primary batch claimed the scene',
                        scope='Scheduling cancellation, not an algorithm-quality rejection'))
                record.update(state='cancelled_primary_batch_claimed_scene', exit_code=process.returncode,
                    finished=time.time(), exposed_to_batch=False)
                write_json(schedule, record)
                return
            time.sleep(10)
    record.update(exit_code=process.returncode, finished=time.time(), exposed_to_batch=False)
    if process.returncode != 0:
        record['state']='failed'; write_json(schedule, record)
        raise SystemExit(process.returncode)
    report = read(output/'cleanup_metrics.json')
    variant = resolve_variant(frozen['method'], frozen['configuration'])
    expected_sources = {k:v for k,v in frozen['sources'].items()
        if k.startswith(('cleaner/', 'evaluation/')) or k in ['auto_clean_gaussians.py', 'execution_provenance.py']}
    if (not intact() or not context_intact() or report['config'].get('implementation_sources')!=expected_sources
            or any(report['config'].get(k)!=v for k,v in frozen['configuration'].items())
            or not completed_identity(output, report['config'], spec, variant)):
        raise ValueError('Completed prefetch does not match the frozen development execution')
    if not expected.exists() and not expected.is_symlink():
        expected.parent.mkdir(exist_ok=True)
        expected.symlink_to(output, target_is_directory=True)
        record['exposed_to_batch']=True
    record.update(state='verified_prefetch_complete', native_sha256=read(output/'COMPLETE.json')['native_sha256'])
    write_json(schedule, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
