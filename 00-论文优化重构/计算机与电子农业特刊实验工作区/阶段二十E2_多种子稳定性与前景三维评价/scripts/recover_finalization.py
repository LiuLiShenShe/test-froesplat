"""Resume only the failed read-only finalization stages with separate evidence."""
from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path
from common import GPU_INDEX,GPU_UUID,PYTHON,ROOT,STAGE,now,read_json,sha256,write_json


def main():
    original_path=STAGE/'reports/FINALIZATION_PROGRESS.json'
    recovery_path=STAGE/'reports/FINALIZATION_RECOVERY.json'
    original=read_json(original_path)
    assert original['status']=='FAILED'
    failed=[step for step in original['steps'] if step['status']=='FAILED']
    assert len(failed)==1 and failed[0]['script']=='recompute_metrics.py'
    assert all(step['status']=='PASS' for step in original['steps'][:-1])
    assert not recovery_path.exists(), 'Preserve any earlier recovery evidence'
    original_log=Path(failed[0]['log_path'])
    original_progress_sha=sha256(original_path)
    original_log_sha=sha256(original_log)
    report={'status':'RUNNING','start_time':now(),'GPU_UUID':GPU_UUID,
            'original_finalization_progress_sha256':original_progress_sha,
            'original_failed_script':failed[0]['script'],
            'original_failed_log_path':str(original_log),
            'original_failed_log_sha256':original_log_sha,
            'original_failure_preserved':True,'steps':[]}
    write_json(recovery_path,report)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_INDEX,OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    sequence=('recompute_metrics.py','make_figures.py','acceptance.py','write_final_reports.py')
    for script in sequence:
        command=[str(PYTHON),str(STAGE/'scripts'/script)]
        log_path=STAGE/'logs'/f'RECOVERY_{script.removesuffix(".py")}.txt'
        step={'script':script,'command_argv':command,'start_time':now(),
              'script_sha256':sha256(STAGE/'scripts'/script),
              'log_path':str(log_path),'status':'RUNNING'}
        report['steps'].append(step)
        write_json(recovery_path,report)
        started=time.perf_counter()
        with log_path.open('xb') as handle:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        step.update({'end_time':now(),'wall_seconds':time.perf_counter()-started,
                     'exit_code':result.returncode,'log_sha256':sha256(log_path),
                     'status':'PASS' if result.returncode==0 else 'FAILED'})
        if result.returncode:
            report.update({'status':'FAILED','end_time':now()})
            write_json(recovery_path,report)
            raise RuntimeError(f'Preserved recovery failure: {script}: {log_path}')
        if script=='acceptance.py':
            acceptance=read_json(STAGE/'reports/FINAL_ACCEPTANCE.json')
            assert acceptance['status']=='PASS'
            report['acceptance_status']='PASS'
        write_json(recovery_path,report)
    assert sha256(original_path)==original_progress_sha
    assert sha256(original_log)==original_log_sha
    report.update({'status':'PASS','end_time':now(),
                   'recovery_scope':'float-metric recomputation, figures, acceptance and final reports only; no training or rendering'})
    write_json(recovery_path,report)
    print('FINALIZATION_RECOVERY_PASS',flush=True)


if __name__=='__main__':
    main()
