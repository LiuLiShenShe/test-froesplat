"""Continue already frozen analysis only after all six formal trainings finish."""
from common import *
import time

def main():
    state={'status':'WAITING_FOR_SIX_FORMAL_TRAININGS','start_time':now()}
    write_json(STAGE/'reports/DEFERRED_POST.json',state)
    while True:
        history=read_json(STAGE/'reports/TRAINING_EXECUTION.json') if (STAGE/'reports/TRAINING_EXECUTION.json').exists() else []
        assert not any(r.get('status')=='FAILED' for r in history),'Training failure must be inspected before further processing'
        if len(history)==6 and all(r.get('status')=='PASS' for r in history):break
        time.sleep(30)
    state.update(status='RUNNING',training_completed_time=now(),steps=[]);write_json(STAGE/'reports/DEFERRED_POST.json',state)
    for script,cpu in [('execute_post.py',False),('summarize.py',True),('comparison_figures.py',True),('recompute_metrics.py',False)]:
        env=dict(os.environ)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        cmd=[str(PYTHON),str(STAGE/'scripts'/script)];log=STAGE/f'logs/DEFERRED_{Path(script).stem.upper()}.txt'
        item={'command':cmd,'cwd':str(ROOT),'env':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE']},'start_time':now(),'log_path':str(log)}
        with log.open('xb') as f:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        item.update(end_time=now(),exit_code=r.returncode,status='PASS' if r.returncode==0 else 'FAILED',log_sha256=sha(log))
        state['steps'].append(item);write_json(STAGE/'reports/DEFERRED_POST.json',state)
        assert r.returncode==0,f'Postprocessing fault retained: {script}'
        print('DEFERRED_COMPLETED',script,flush=True)
    state.update(status='PASS',end_time=now());write_json(STAGE/'reports/DEFERRED_POST.json',state)
if __name__=='__main__':main()
