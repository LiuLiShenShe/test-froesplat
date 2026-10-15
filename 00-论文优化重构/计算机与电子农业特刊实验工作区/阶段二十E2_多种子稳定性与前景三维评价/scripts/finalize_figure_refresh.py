"""Run final acceptance/report generation after refreshed figures are reviewed."""
from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path
from common import GPU_INDEX,GPU_UUID,PYTHON,ROOT,STAGE,now,read_csv,read_json,sha256,write_json


def main():
    progress_path=STAGE/'reports/FIGURE_REFRESH_PROGRESS.json'
    report=read_json(progress_path)
    assert report['status']=='FIGURES_READY_FOR_VISUAL_QA'
    qa_path=STAGE/'reports/FIGURE_VISUAL_QA.md'
    assert qa_path.is_file() and 'directly inspected' in qa_path.read_text(encoding='utf-8')
    index=read_csv(STAGE/'reports/FIGURE_INDEX.csv')
    assert all(Path(row['path']).is_file() and sha256(row['path'])==row['sha256'] for row in index)
    assert sha256(STAGE/'reports/FIGURE_INDEX.csv')==report['figure_index_sha256']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_INDEX,OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    for script in ('acceptance.py','write_final_reports.py'):
        command=[str(PYTHON),str(STAGE/'scripts'/script)]
        log_path=STAGE/'logs'/f'FIGURE_REFRESH_{script.removesuffix(".py")}.txt'
        step={'script':script,'command_argv':command,'start_time':now(),
              'script_sha256':sha256(STAGE/'scripts'/script),
              'log_path':str(log_path),'status':'RUNNING'}
        report['steps'].append(step)
        report['status']='RUNNING'
        write_json(progress_path,report)
        started=time.perf_counter()
        with log_path.open('xb') as handle:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        step.update({'end_time':now(),'wall_seconds':time.perf_counter()-started,
                     'exit_code':result.returncode,'log_sha256':sha256(log_path),
                     'status':'PASS' if result.returncode==0 else 'FAILED'})
        if result.returncode:
            report.update({'status':'FAILED','end_time':now()})
            write_json(progress_path,report)
            raise RuntimeError(f'Figure refresh finalization failed: {script}: {log_path}')
        if script=='acceptance.py':
            acceptance=read_json(STAGE/'reports/FINAL_ACCEPTANCE.json')
            assert acceptance['status']=='PASS'
            report['final_acceptance_status']='PASS'
        write_json(progress_path,report)
    report.update({'status':'PASS','end_time':now(),
                   'visual_QA_sha256':sha256(qa_path),
                   'figure_index_sha256':sha256(STAGE/'reports/FIGURE_INDEX.csv')})
    write_json(progress_path,report)
    print('FIGURE_REFRESH_AND_FINAL_ACCEPTANCE_PASS',flush=True)


if __name__=='__main__':main()
