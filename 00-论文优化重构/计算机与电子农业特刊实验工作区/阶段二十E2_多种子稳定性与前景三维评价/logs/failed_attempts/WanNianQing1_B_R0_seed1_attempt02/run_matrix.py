from __future__ import annotations
import json
import os
import subprocess
import time
from pathlib import Path
from common import *


def launch(sample,group,seed,attempt):
    label=run_id(sample,group,seed,attempt)
    log=STAGE/f'logs/{label}_STDOUT_STDERR.txt'
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':GPU_INDEX,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'})
    command=[str(PYTHON),str(STAGE/'scripts/train_observed.py'),'--sample',sample,'--group',group,'--seed',str(seed),'--attempt',str(attempt)]
    assert not log.exists()
    started=time.perf_counter(); launch_time=now(); samples=[]
    with log.open('xb') as handle:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        write_json(STAGE/'reports/ACTIVE_TRAINING.json',{'run_id':label,'pid':process.pid,'start_time':launch_time,'command':command,'log':str(log)})
        print(f'START {label} PID={process.pid} at {launch_time}',flush=True)
        while process.poll() is None:
            result=subprocess.run(['nvidia-smi','-i',GPU_INDEX,'--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if result.returncode==0:
                uuid,memory,util,temp=[v.strip() for v in result.stdout.strip().split(',')]
                assert uuid==GPU_UUID
                samples.append({'timestamp':now(),'elapsed_seconds':time.perf_counter()-started,'device_used_MiB':float(memory),'utilization_pct':float(util),'temperature_C':float(temp),'pid':process.pid})
            time.sleep(5)
    write_csv(STAGE/f'logs/{label}_GPU_SAMPLES.csv',samples)
    process_record={'sample':sample,'group':group,'seed':seed,'attempt':attempt,'pid':process.pid,
                    'start_time':launch_time,'end_time':now(),'wall_seconds':time.perf_counter()-started,
                    'exit_code':process.returncode,'command_argv':command,'cwd':str(ROOT),'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},
                    'GPU_UUID':GPU_UUID,'stdout_stderr_path':str(log),'stdout_stderr_sha256':sha256(log),
                    'sampled_device_peak_MiB':max((r['device_used_MiB'] for r in samples),default=None),'sampling_seconds':5}
    write_json(STAGE/f'reports/{label}_PROCESS.json',process_record)
    if process.returncode!=0:
        output=output_path(sample,group,seed,attempt)
        artifacts=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in output.rglob('*') if p.is_file()] if output.exists() else []
        write_json(STAGE/f'reports/{label}_FAILED_ATTEMPT.json',{'timestamp':now(),'process':process_record,'preserved_artifacts':artifacts,'no_checkpoint_selection':True})
        text=log.read_text(errors='replace')[-20000:].lower()
        retryable=any(word in text for word in ('out of memory','nan','interrupted','keyboardinterrupt','sigterm','killed')) or process.returncode<0
        return False,retryable
    validation_log=STAGE/f'logs/{label}_RELOAD_VALIDATION.txt'
    with validation_log.open('xb') as handle:
        check=subprocess.run([str(PYTHON),str(STAGE/'scripts/validate_model.py'),'--sample',sample,'--group',group,'--seed',str(seed),'--attempt',str(attempt)],env=env,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    assert check.returncode==0,f'Reload validation failed; preserve model and repair evaluator: {validation_log}'
    print(f'COMPLETED {label} and RELOAD_PASS at {now()}',flush=True)
    return True,False


def main():
    assert read_json(STAGE/'reports/PRETRAINING_INPUT_RECHECK.json')['status']=='PASS'
    assert read_json(STAGE/'reports/SEED_CONTROL_VERIFICATION.json')['status']=='PASS'
    progress=[]
    for index,(sample,group,seed) in enumerate(TRAIN_ORDER):
        completed=False
        for attempt in range(1,MAX_FULL_ATTEMPTS+1):
            validation=STAGE/f'reports/{run_id(sample,group,seed,attempt)}_VALIDATION.json'
            if validation.exists() and read_json(validation)['status']=='PASS':
                completed=True;break
            process_path=STAGE/f'reports/{run_id(sample,group,seed,attempt)}_PROCESS.json'
            if process_path.exists():
                previous=read_json(process_path)
                if previous['exit_code']==0:
                    raise RuntimeError('A completed training awaits validation; never retrain because validation failed')
                continue
            completed,retryable=launch(sample,group,seed,attempt)
            if completed:break
            if not retryable:raise RuntimeError('Non-infrastructure failure: repair stage code before continuing; preserve failed attempt')
        progress.append({'sample':sample,'group':group,'seed':seed,'completed':completed,'attempt':attempt})
        write_json(STAGE/'reports/MATRIX_PROGRESS.json',{'timestamp':now(),'completed':sum(r['completed'] for r in progress),'required':8,'runs':progress})
        if not completed:raise RuntimeError(f'Predeclared attempts exhausted: {sample}/{group}/seed{seed}')
        if index==1:
            write_json(STAGE/'reports/FIRST_PAIR_GATE.json',{'timestamp':now(),'status':'PASS','WanNianQing1_seed1_BC_completed_and_reloadable':True})
    write_json(STAGE/'reports/ACTIVE_TRAINING.json',{'status':'ALL_EIGHT_COMPLETED','timestamp':now()})
    print('ALL_EIGHT_TRAININGS_COMPLETED',flush=True)

if __name__=='__main__':main()
