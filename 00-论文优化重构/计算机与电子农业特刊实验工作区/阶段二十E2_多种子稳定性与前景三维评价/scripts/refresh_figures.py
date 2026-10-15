"""Regenerate indexed scientific figures after the display-label QA fix."""
from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path
from common import GPU_INDEX,GPU_UUID,PYTHON,ROOT,STAGE,now,read_csv,read_json,sha256,write_json


def main():
    recovery=read_json(STAGE/'reports/FINALIZATION_RECOVERY.json')
    acceptance=read_json(STAGE/'reports/FINAL_ACCEPTANCE.json')
    assert recovery['status']=='PASS' and acceptance['status']=='PASS'
    progress_path=STAGE/'reports/FIGURE_REFRESH_PROGRESS.json'
    assert not progress_path.exists(), 'Preserve earlier visual-refresh evidence'
    script=STAGE/'scripts/make_figures.py'
    command=[str(PYTHON),str(script)]
    log_path=STAGE/'logs/FIGURE_REFRESH_make_figures.txt'
    report={'status':'RUNNING','start_time':now(),'GPU_UUID':GPU_UUID,
            'based_on_acceptance_sha256':sha256(STAGE/'reports/FINAL_ACCEPTANCE.json'),
            'steps':[{'script':'make_figures.py','command_argv':command,'start_time':now(),
                      'script_sha256':sha256(script),'log_path':str(log_path),'status':'RUNNING'}]}
    write_json(progress_path,report)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_INDEX,OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    started=time.perf_counter()
    with log_path.open('xb') as handle:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    step=report['steps'][0]
    step.update({'end_time':now(),'wall_seconds':time.perf_counter()-started,
                 'exit_code':result.returncode,'log_sha256':sha256(log_path),
                 'status':'PASS' if result.returncode==0 else 'FAILED'})
    if result.returncode:
        report.update({'status':'FAILED','end_time':now()})
        write_json(progress_path,report)
        raise RuntimeError(f'Figure refresh failed: {log_path}')
    index=read_csv(STAGE/'reports/FIGURE_INDEX.csv')
    assert index and all(Path(row['path']).is_file() and sha256(row['path'])==row['sha256'] for row in index)
    assert {'RGB','BC_ABSOLUTE_DIFFERENCE','ALPHA','DEPTH','NORMAL'} <= {row['figure_kind'] for row in index}
    report.update({'status':'FIGURES_READY_FOR_VISUAL_QA','end_time':now(),
                   'figure_count':len(index),'figure_index_sha256':sha256(STAGE/'reports/FIGURE_INDEX.csv'),
                   'all_indexed_figure_hashes_verified':True,
                   'display_change':'Shortened difference colorbar label; tensor values, color limits and metrics unchanged'})
    write_json(progress_path,report)
    print('FIGURE_REFRESH_READY_FOR_VISUAL_QA',len(index),flush=True)


if __name__=='__main__':main()
