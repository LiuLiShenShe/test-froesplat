"""Wait for the active baseline, then run every remaining GPU stage serially."""
import fcntl
import json
import os
import subprocess
import sys
import time
from runtime import OUT, ROOT, PYTHONS, write
from run_training import run


def stage(name,python,script,args,marker):
    if marker.exists():return
    command=['uv','run','--no-project','--python',str(python),'python',str(OUT/'scripts'/script)]+args
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',
             CUDA_HOME='/usr/local/cuda-12.6',TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='4')
    record=dict(stage=name,command=command,started_unix=time.time())
    with (OUT/'summary/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    with (OUT/'logs'/f'{name}.log').open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        monitor=subprocess.Popen([str(PYTHONS['GGGS']),str(OUT/'scripts/resource_monitor.py'),str(child.pid),str(OUT/'summary'/f'{name}_resources.json')],stdout=log,stderr=subprocess.STDOUT)
        write(OUT/'summary/progress.json',dict(status='running',driver_pid=os.getpid(),child_pid=child.pid,**record))
        rc=child.wait();monitor.wait(timeout=15)
    if rc or not marker.exists():raise RuntimeError(f'{name} failed rc={rc}; inspect logs/{name}.log')


if __name__=='__main__':
    lock=(OUT/'summary/gpu1.lock').open('a')
    print('Waiting for current baseline lock',flush=True)
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        assert (OUT/'gaussian/3DGS/complete.json').exists(),'Active baseline did not finish successfully'
        stage('RAP_FSAM3',OUT/'.venv-rap/bin/python','rap_masks.py',[],OUT/'common_input/rap_masks_complete.json')
        for method in ['3DGS','MCMC_3DGS','2DGS','GGGS','gsplat']:
            run(method)
            stage('evaluate_'+method,PYTHONS[method],'evaluate.py',[method],OUT/'gaussian'/method/'evaluation/metrics.json')
        write(OUT/'summary/progress.json',dict(status='gaussian_evaluation_complete',requires='Inspect same-camera figures, choose 1–2 native models, then run conditional cleaning and meshes'))
    except BaseException as error:
        write(OUT/'summary/progress.json',dict(status='failed',driver_pid=os.getpid(),error=str(error)))
        raise
