"""Persist the authorized full E11 run through camera validation and 30K."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['MPLBACKEND']='Agg'
import json,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'
def run(stage,script):
    command=['uv','run','--no-project','--python',PY,'python',str(ROOT/'scripts'/script)]
    record=dict(stage=stage,command=command,cwd=str(ROOT),CUDA_VISIBLE_DEVICES='1',started_unix=time.time())
    with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    with (ROOT/'logs'/f'{stage}.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        (ROOT/'STATUS.json').write_text(json.dumps(dict(status='running',stage=stage,pid=process.pid,driver_pid=os.getpid(),physical_gpu=1),indent=2)+'\n')
        rc=process.wait()
    record.update(returncode=rc,ended_unix=time.time())
    with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if rc:raise RuntimeError(f'{stage} failed, see logs/{stage}.log')
    print(stage,'complete',flush=True)
try:
    # The one-time isolated source copy is already running in another cell.
    while not (ROOT/'camera_interface/2dgs_patch_complete.json').exists():time.sleep(3)
    run('E11_camera_validation','e11_camera_validation.py')
    run('E11_training_30K','train_e11.py')
    (ROOT/'STATUS.json').write_text(json.dumps(dict(status='running',stage='E11_complete_mesh_experiments_pending',physical_gpu=1),indent=2)+'\n')
except Exception as e:
    (ROOT/'STATUS.json').write_text(json.dumps(dict(status='failed',error=str(e),physical_gpu=1),indent=2)+'\n')
    raise
