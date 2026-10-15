"""Finish independent CPU analysis of frozen M0 Meshes while GPU training runs."""
from common import *

def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    steps=[('figures.py','CaoMei2'),('evaluate.py','HongZhang'),('evaluate_mesh.py','HongZhang'),('figures.py','HongZhang')]
    steps += [('surface_views.py',s) for s in SCENES]
    history=[]
    for script,sample in steps:
        label=f'{sample}_M0_{Path(script).stem.upper()}_CPU'
        cmd=[str(PYTHON),str(STAGE/'scripts'/script),'--sample',sample,'--method','M0']
        log=STAGE/f'logs/{label}.txt'
        item={'label':label,'command':cmd,'cwd':str(ROOT),'start_time':now(),'env':{k:env.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE']},'log_path':str(log)}
        with log.open('xb') as f:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        item.update(end_time=now(),exit_code=r.returncode,log_sha256=sha(log),status='PASS' if r.returncode==0 else 'FAILED')
        history.append(item);write_json(STAGE/'reports/M0_CPU_EXECUTION.json',history)
        assert r.returncode==0,label
        print('CPU_COMPLETED',label,flush=True)
if __name__=='__main__':main()
