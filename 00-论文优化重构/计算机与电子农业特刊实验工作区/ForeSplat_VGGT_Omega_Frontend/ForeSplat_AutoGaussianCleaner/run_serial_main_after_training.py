#!/usr/bin/env python3
"""Retry frozen main ablations serially after the large native training job.

This coordinator controls resources only. It cannot edit a cleaning/training
configuration, change a source or admit independent test plants.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_gpu1_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    for line in result.stdout.splitlines():
        index, free = [part.strip() for part in line.split(',')]
        if index == '1':
            return int(free)
    raise ValueError('Physical GPU1 memory could not be read')


def post_densification_record(job, manifest_job):
    if job.get('status') != 'training' or not job.get('log'):
        return None
    parameters=Path(manifest_job['training_dir'])/'actual_training_parameters.json'
    data=read(parameters)
    if data['physical_gpu'] != 1 or data['opt']['iterations'] != 30000:
        raise ValueError('Resource admission requires the unchanged native GPU1 30K training')
    with Path(job['log']).open('rb') as stream:
        stream.seek(max(0,stream.seek(0,2)-65536))
        tail=stream.read().decode(errors='replace')
    iterations=re.findall(r'(\d+)/30000',tail)
    if not iterations or int(iterations[-1]) <= data['opt']['densify_until_iter']:
        return None
    return dict(actual_iteration=int(iterations[-1]),densify_until_iter=data['opt']['densify_until_iter'],
                parameters=str(parameters.resolve()),parameters_sha256=digest(parameters),
                scope='Observed native training is beyond its unchanged densification endpoint; single-main resource admission only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--comparison-root', type=Path, required=True)
    parser.add_argument('--preparation-status', type=Path, required=True)
    parser.add_argument('--wait-plant', required=True)
    parser.add_argument('--minimum-free-mib', type=int, default=12288)
    parser.add_argument('--allow-post-densification', action='store_true',
                        help='Admit one measured small main-model task after native densification ends and headroom is stable')
    parser.add_argument('--development-output', type=Path, required=True)
    args = parser.parse_args()
    module = Path(__file__).resolve().parent
    project, comparison, implementation = module.parent, args.comparison_root.resolve(), args.implementation_root.resolve()
    queue_root = comparison/'C5_Fused_GGGS/development_v12'
    lock = (queue_root/'serial_main_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources = list((implementation/'cleaner').rglob('*.py'))+list((implementation/'evaluation').glob('*.py'))
    sources += [implementation/'auto_clean_gaussians.py', implementation/'execution_provenance.py',
                implementation/'auto_clean_batch.py', implementation/'configs/default.yaml', args.config.resolve()]
    frozen = {str(path):digest(path) for path in sources}
    status = dict(scope='Resource-only serial coordinator; same frozen source/config and original E10 input; no test-plant cleanup',
                  sources=frozen, methods={}, state='waiting_for_large_native_training', started=time.time())
    status_path = queue_root/'SERIAL_MAIN_QUEUE_STATUS.json'
    write(status_path, status)
    completed_training = ['trained_gggs_30K', 'completed_existing_training', 'reused_final_gggs_checkpoint']
    manifest = read(args.manifest)
    waiting_job=next(job for job in manifest['plants'] if job['plant']==args.wait_plant)
    stable_samples=0
    while True:
        preparation = read(args.preparation_status) if args.preparation_status.exists() else {}
        job = preparation.get(args.wait_plant, {})
        if job.get('status') == 'failed':
            raise ValueError('Large native training actually failed: '+str(job))
        free = free_gpu1_memory()
        status.update(wait_plant=args.wait_plant, training_status=job.get('status','pending'), free_gpu1_mib=free)
        admission=post_densification_record(job,waiting_job) if args.allow_post_densification else None
        stable_samples=stable_samples+1 if admission and free >= args.minimum_free_mib else 0
        status.update(post_densification_admission=admission,stable_headroom_samples=stable_samples,
                      minimum_free_mib=args.minimum_free_mib)
        write(status_path, status)
        if job.get('status') in completed_training and free >= args.minimum_free_mib:
            break
        if admission and stable_samples>=3:
            break
        time.sleep(30)
    scene = next(job['scene'] for job in manifest['plants'] if job['plant'] == 'KongQueZhuYu' and job['group'] == 'development')
    main_pid = None
    for method in ['C5-A', 'C5-B', 'C5-C', 'C5-D']:
        while free_gpu1_memory() < args.minimum_free_mib:
            status.update(state='waiting_for_GPU1_headroom', next_method=method)
            write(status_path, status);time.sleep(30)
        if {str(path):digest(path) for path in sources} != frozen:
            raise ValueError('Frozen main source/configuration changed before retry')
        output = queue_root/method
        command = ['uv','run','--no-sync','python','-u',str(implementation/'auto_clean_gaussians.py'),
                   '--scene',scene,'--method',method,'--config',str(args.config.resolve()),'--output',str(output),
                   '--resume','--baseline-cache',str(comparison/'C5_Fused_GGGS/development_v8_1/C5-B/comparison/initial')]
        if method != 'C5-A':
            command += ['--evidence-cache',str(comparison/'C5_Fused_GGGS/development_v8_1/C5-B/rounds/01/evidence.npz')]
        log = comparison/f'{method.lower().replace("-","")}_v12_serial_run.log'
        record = dict(state='running', command=command, output=str(output), log=str(log), started=time.time())
        status.update(state='running_serial_main', active_method=method)
        with log.open('a') as stream:
            child = subprocess.Popen(command, cwd=project, env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            status['methods'][method] = record;write(status_path, status)
            code = child.wait()
        record.update(exit_code=code, child_wall_seconds=time.time()-record['started'], finished=time.time())
        if code != 0 or not (output/'COMPLETE.json').exists():
            record['state'] = 'execution_failed';write(status_path, status)
            raise ValueError('Serial main execution failed; not a successful cleaning result: '+method)
        if {str(path):digest(path) for path in sources} != frozen:
            raise ValueError('Frozen main source/configuration changed during retry')
        record['state'] = 'CLI_completed_and_source_intact';write(status_path, status)
        evaluation = ['uv','run','--no-sync','python','-u',str(module/'finalize_main_comparison.py'),'--outputs',str(output)]
        evaluation_log = comparison/f'{method.lower().replace("-","")}_v12_serial_evaluation.log'
        with evaluation_log.open('a') as stream:
            code = subprocess.call(evaluation, cwd=project, env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        record.update(common_evaluation_exit_code=code, common_evaluation_log=str(evaluation_log))
        write(status_path, status)
        if code != 0:
            raise ValueError('Actual common evaluation failed: '+method)
        if method == 'C5-C':main_pid = child.pid
    # Large development models have no measured small-model memory bound.
    # Finish all original native model preparation before their batch begins.
    while True:
        preparation=read(args.preparation_status)
        incomplete={job['plant']:preparation.get(job['plant'],{}).get('status','pending')
                    for job in manifest['plants'] if preparation.get(job['plant'],{}).get('status') not in completed_training}
        if any(value=='failed' for value in incomplete.values()):
            raise ValueError('Native model preparation actually failed before large-model development: '+str(incomplete))
        if not incomplete:
            break
        status.update(state='main_complete_waiting_for_all_native_inputs_before_large_development',
                      pending_native_inputs=incomplete)
        write(status_path,status);time.sleep(30)
    # Every main ablation has actually exited and been evaluated before the
    # fixed candidate version enters its six-scene development split.
    stage = ['uv','run','--no-sync','python','-u',str(module/'stage_development_after_main.py'),
             '--main-output',str(queue_root/'C5-C'),'--main-pid','0',
             '--entrypoint',str(implementation/'auto_clean_batch.py'),'--manifest',str(args.manifest.resolve()),
             '--config',str(args.config.resolve()),'--output-root',str(args.development_output.resolve()),'--jobs','1']
    status.update(state='all_main_ablations_complete_starting_development', development_command=stage,
                  completed_main_producer_pid=main_pid,
                  main_producer_scope='Actual CLI already waited and exited successfully; PID0 prevents waiting on a later unrelated PID reuse')
    write(status_path, status)
    with (comparison/'development_v12_serial_stage.log').open('a') as stream:
        child = subprocess.Popen(stage, cwd=project, env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,start_new_session=True)
    status.update(state='development_started', development_driver_pid=child.pid, main_complete=time.time())
    write(status_path, status)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--comparison-root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--comparison-root')+1]).resolve()
            write(root/'C5_Fused_GGGS/development_v12/SERIAL_MAIN_QUEUE_FAILURE.json',
                  dict(reason=str(error),failed=time.time(),scope='Coordinator failure; not a successful cleaning or a quality rejection'))
        raise
