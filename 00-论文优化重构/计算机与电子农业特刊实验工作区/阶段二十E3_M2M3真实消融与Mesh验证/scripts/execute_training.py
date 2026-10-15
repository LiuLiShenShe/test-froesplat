"""Run predeclared fresh trainings serially on the single designated GPU."""
from common import *
import time

def main():
    frozen=verify_frozen();events=[]
    for sample in SCENES:
        for method in ['M1','M2']:
            report=STAGE/f'reports/{sample}_{method}_TRAIN_RUN.json'
            if report.exists():
                r=read_json(report);assert r['status']=='PASS' and r['completed_iterations']==30000
                events.append({'sample':sample,'method':method,'status':'ALREADY_COMPLETED','report':str(report)});continue
            gpu_idle()
            cmd=[str(PYTHON),str(STAGE/'scripts/train_observed.py'),'--sample',sample,'--method',method]
            log=STAGE/f'logs/{sample}_{method}_TRAIN.txt'
            item={'sample':sample,'method':method,'command':cmd,'cwd':str(ROOT),'env':{k:os.environ[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE']},
                  'start_time':now(),'status':'RUNNING','log_path':str(log)}
            events.append(item);write_json(STAGE/'reports/TRAINING_EXECUTION.json',events)
            with log.open('xb') as f:process=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            item.update(end_time=now(),exit_code=process.returncode,log_sha256=sha(log))
            item['status']='PASS' if process.returncode==0 else 'FAILED'
            write_json(STAGE/'reports/TRAINING_EXECUTION.json',events)
            assert process.returncode==0,f'Execution fault retained: {sample} {method}; stop for diagnosis'
            assert read_json(report)['reload_verified']
            print('TRAINING_COMPLETED',sample,method,flush=True)
    print('ALL_SIX_FORMAL_TRAININGS_COMPLETED',flush=True)

if __name__=='__main__':main()
