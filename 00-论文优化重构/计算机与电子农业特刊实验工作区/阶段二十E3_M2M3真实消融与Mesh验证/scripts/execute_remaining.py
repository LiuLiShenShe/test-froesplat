"""Execution order frozen before formal quality results; stop only on faults."""
from common import *

def main():
    verify_frozen();events=[]
    for scene in SCENES:
        if scene=='WanNianQing1':
            assert read_json(STAGE/'reports/WanNianQing1_M0_MESH_RUN.json')['status']=='PASS';continue
        cmd=[str(PYTHON),str(STAGE/'scripts/extract_mesh.py'),'--sample',scene,'--method','M0']
        events.append(run(cmd,f'{scene}_M0_MESH'));write_json(STAGE/'reports/PREFLIGHT_EXECUTION.json',events)
    for method in ['M1','M2']:
        cmd=[str(PYTHON),str(STAGE/'scripts/train_observed.py'),'--sample','WanNianQing1','--method',method,'--smoke']
        events.append(run(cmd,f'WanNianQing1_{method}_SMOKE'));write_json(STAGE/'reports/PREFLIGHT_EXECUTION.json',events)
    cmd=[str(PYTHON),str(STAGE/'scripts/execute_training.py')]
    events.append(run(cmd,'FORMAL_TRAINING_CONTROLLER'));write_json(STAGE/'reports/PREFLIGHT_EXECUTION.json',events)

def run(cmd,label):
    gpu_idle();log=STAGE/f'logs/{label}.txt'
    item={'label':label,'command':cmd,'cwd':str(ROOT),'env':{k:os.environ[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE']},'start_time':now(),'status':'RUNNING','log_path':str(log)}
    write_json(STAGE/'reports/CURRENT_EXECUTION.json',item)
    with log.open('xb') as handle:r=subprocess.run(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
    item.update(end_time=now(),exit_code=r.returncode,status='PASS' if r.returncode==0 else 'FAILED',log_sha256=sha(log))
    write_json(STAGE/'reports/CURRENT_EXECUTION.json',item)
    assert r.returncode==0,f'Execution fault retained: {label}'
    print('COMPLETED',label,flush=True);return item

if __name__=='__main__':main()
