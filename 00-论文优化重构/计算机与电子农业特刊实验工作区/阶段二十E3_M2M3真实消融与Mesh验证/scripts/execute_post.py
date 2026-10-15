"""Run the unchanged common evaluator and TSDF protocol after all six trainings."""
from common import *
import time

ENV_KEYS=['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE']

def run(script,sample,method=None,cpu=False):
    if not cpu:gpu_idle()
    cmd=[str(PYTHON),str(STAGE/'scripts'/script),'--sample',sample]
    if method:cmd+=['--method',method]
    label=f'{sample}_{method or "M0"}_{Path(script).stem.upper()}_POST'
    log=STAGE/f'logs/{label}.txt';env=dict(os.environ)
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    item={'label':label,'command':cmd,'cwd':str(ROOT),'env':{k:env[k] for k in ENV_KEYS},'start_time':now(),'log_path':str(log)}
    with log.open('xb') as f:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    item.update(end_time=now(),exit_code=result.returncode,status='PASS' if result.returncode==0 else 'FAILED',log_sha256=sha(log))
    history=read_json(STAGE/'reports/POST_EXECUTION.json') if (STAGE/'reports/POST_EXECUTION.json').exists() else []
    history.append(item);write_json(STAGE/'reports/POST_EXECUTION.json',history)
    assert result.returncode==0,f'Execution fault preserved: {label}'
    print('POST_COMPLETED',label,flush=True)

def main():
    verify_frozen()
    for sample in SCENES:
        for method in ['M1','M2']:
            r=read_json(STAGE/f'reports/{sample}_{method}_TRAIN_RUN.json')
            assert r['status']=='PASS' and r['completed_iterations']==30000
    for sample in SCENES:
        for method in ['M1','M2']:
            run('evaluate.py',sample,method)
            run('extract_mesh.py',sample,method)
            run('evaluate_mesh.py',sample,method,cpu=True)
            run('figures.py',sample,method,cpu=True)
            run('surface_views.py',sample,method,cpu=True)
    for sample in SCENES:run('sensitivity.py',sample)
    print('ALL_POSTPROCESSING_COMPLETED',flush=True)

if __name__=='__main__':main()
