"""Durable, serial GPU1 training; no historical metrics enter this experiment."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from runtime import OUT, PYTHONS, SOURCES, write


def run(method):
    output = OUT/'gaussian'/method
    if (output/'complete.json').exists(): return
    output.mkdir(parents=True,exist_ok=True)
    command = ['uv','run','--no-project','--python',str(PYTHONS[method]),'python',str(OUT/'scripts/native_train.py'),method]
    if method == 'MCMC_3DGS':
        baseline = json.loads((OUT/'gaussian/3DGS/complete.json').read_text())
        capacity = baseline['gaussian_count']
        lock = OUT/'gaussian/MCMC_3DGS/capacity_lock.json'
        if lock.exists(): assert json.loads(lock.read_text())['cap_max'] == capacity
        else: write(lock, dict(cap_max=capacity, source=str(OUT/'gaussian/3DGS/complete.json'), fixed_before_training=True))
        command += ['--cap-max', str(capacity)]
    if method == 'gsplat':
        command = ['uv','run','--no-project','--python',str(PYTHONS[method]),'python',str(OUT/'scripts/gsplat_train.py')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
               TORCH_CUDA_ARCH_LIST='8.6', CUDA_HOME='/usr/local/cuda-12.6', MAX_JOBS='4')
    record = dict(method=method, command=command, started_unix=time.time(), physical_gpu=1)
    with (OUT/'summary/actual_commands.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    print('START', method, flush=True)
    with (OUT/'logs'/f'{method}_training.log').open('a') as log:
        child = subprocess.Popen(command, cwd=SOURCES[method], env=env, stdout=log, stderr=subprocess.STDOUT)
        observer=subprocess.Popen([str(PYTHONS['GGGS']),str(OUT/'scripts/resource_monitor.py'),str(child.pid),str(output/'observed_resources.json')],stdout=log,stderr=subprocess.STDOUT)
        write(OUT/'summary/progress.json',dict(status='running',driver_pid=os.getpid(),child_pid=child.pid,**record))
        rc = child.wait()
        observer.wait(timeout=15)
    record.update(returncode=rc, ended_unix=time.time(), wall_s=time.time()-record['started_unix'])
    write(output/'process_resources.json',record)
    if rc or not (output/'complete.json').exists(): raise RuntimeError(f'{method} failed; inspect {OUT}/logs/{method}_training.log')
    print('DONE',method,flush=True)


if __name__ == '__main__':
    lock=(OUT/'summary/gpu1.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    methods = sys.argv[1:] or ['3DGS','MCMC_3DGS','2DGS','GGGS','gsplat']
    try:
        for method in methods: run(method)
    except BaseException as error:
        write(OUT/'summary/progress.json',dict(status='failed',driver_pid=os.getpid(),error=str(error)))
        raise
    write(OUT/'summary/progress.json',dict(status='training_complete',methods=methods))
