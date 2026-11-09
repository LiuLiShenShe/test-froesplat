"""Sequential native CPU fusion, independent from the NKSR GPU worker."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='8'
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'
for method in ['M4_E10_GGGS_TSDF','M5_E11_2DGS_TSDF']:
    for matched in [False,True]:
        stage=method+('_alpha_filtered' if matched else '')
        command=['uv','run','--no-project','--python',PY,'python',str(ROOT/'scripts/extract_native_tsdf.py'),'--experiment',method]
        if matched:command.append('--alpha-filter')
        record=dict(stage=stage,command=command,cwd=str(ROOT),started_unix=time.time(),CUDA_VISIBLE_DEVICES='1',fusion_device='CPU:0 as official native TSDF')
        with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        (ROOT/'TSDF_STATUS.json').write_text(json.dumps(dict(status='running',stage=stage,pid=os.getpid()),indent=2)+'\n')
        with (ROOT/'logs'/f'{stage}.log').open('a') as log:rc=subprocess.call(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        record.update(returncode=rc,ended_unix=time.time())
        with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if rc:raise RuntimeError(f'{stage} failed; logs retained')
        print(stage,'complete',flush=True)
(ROOT/'TSDF_STATUS.json').write_text(json.dumps(dict(status='complete',matrices=4),indent=2)+'\n')
