"""Finish frozen performance observations and external-asset recovery indexing."""
from common import *
import time

def main():
    record={'status':'WAITING_FOR_DEFERRED_POST','start_time':now(),'steps':[]}
    write_json(STAGE/'reports/FINAL_DATA_JOBS.json',record)
    while True:
        state=read_json(STAGE/'reports/DEFERRED_POST.json')
        assert not any(x.get('status')=='FAILED' for x in state.get('steps',[])),'Inspect retained postprocessing fault first'
        if state['status']=='PASS':break
        time.sleep(30)
    record.update(status='RUNNING');write_json(STAGE/'reports/FINAL_DATA_JOBS.json',record)
    for script,cpu in [('benchmark_inference.py',False),('capacity_report.py',True),('build_recovery.py',True)]:
        env=dict(os.environ)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        cmd=[str(PYTHON),str(STAGE/'scripts'/script)]
        log=STAGE/f'logs/FINAL_DATA_{Path(script).stem.upper()}.txt'
        item={'command':cmd,'cwd':str(ROOT),'env':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE']},'log_path':str(log),'start_time':now()}
        with log.open('xb') as out:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        item.update(end_time=now(),exit_code=r.returncode,status='PASS' if r.returncode==0 else 'FAILED',log_sha256=sha(log))
        record['steps'].append(item);write_json(STAGE/'reports/FINAL_DATA_JOBS.json',record)
        assert r.returncode==0,f'Data completion fault retained: {script}'
        print('FINAL_DATA_COMPLETED',script,flush=True)
    record.update(status='PASS',end_time=now());write_json(STAGE/'reports/FINAL_DATA_JOBS.json',record)
if __name__=='__main__':main()
