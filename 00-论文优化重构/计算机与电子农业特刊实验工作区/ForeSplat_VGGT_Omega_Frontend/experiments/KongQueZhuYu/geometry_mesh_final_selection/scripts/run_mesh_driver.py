"""Durable full experiment execution, one GPU task at a time on GPU1."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['MPLBACKEND']='Agg';os.environ['OMP_NUM_THREADS']='8'
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY2D='/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'
PYGG=str(ROOT.parents[2]/'.venv/bin/python')
PYNK=str(ROOT/'.venv-nksr/bin/python')
BACKGROUND=[]
# One root driver; complete products are resumed without loading GPU runtimes.
import fcntl
_DRIVER_LOCK=(ROOT/'driver.lock').open('a')
fcntl.flock(_DRIVER_LOCK.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)

def cached(script,args):
    target=None
    if script=='render_surface.py':
        exp=args[1];name=f'{exp.lower()}_{"2dgs" if exp=="E11" else "gggs"}_surface'
        target=ROOT/'surface_inputs'/name/'render_complete.json'
    elif script=='prepare_rendered_inputs.py':
        exp=args[1];name=f'{exp.lower()}_{"2dgs" if exp=="E11" else "gggs"}_surface'
        target=ROOT/'surface_inputs'/name/'complete.json'
    elif script=='evaluate_e11.py':target=ROOT/'E11_2DGS_Corrected/evaluation_complete.json'
    elif script=='extract_nksr.py':
        method=args[1];folder=ROOT/'meshes'/method if method.startswith('M') else ROOT/'meshes/supplementary'/method
        target=folder/'complete.json'
    elif script=='evaluate_meshes.py':target=Path(args[1])/'evaluation/complete.json'
    elif script=='run_tsdf_worker.py':target=ROOT/'TSDF_STATUS.json'
    return target is not None and target.exists() and json.loads(target.read_text()).get('status')=='complete'


def status(stage,**kwargs):
    (ROOT/'MESH_STATUS.json').write_text(json.dumps(dict(status='running',stage=stage,driver_pid=os.getpid(),physical_gpu=1,**kwargs),indent=2)+'\n')

def run(stage,script,args=(),python=PYNK):
    if cached(script,args):
        print(stage,"already complete; retained",flush=True);return
    command=['uv','run','--no-project','--python',python,'python',str(ROOT/'scripts'/script),*map(str,args)]
    record=dict(stage=stage,command=command,cwd=str(ROOT),CUDA_VISIBLE_DEVICES='1',PYTORCH_ALLOC_CONF=os.environ.get('PYTORCH_ALLOC_CONF','native default'),started_unix=time.time())
    with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    status(stage)
    with (ROOT/'logs'/f'{stage}.log').open('a') as log:rc=subprocess.call(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
    record.update(returncode=rc,ended_unix=time.time())
    with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if rc:raise RuntimeError(f'{stage} failed; see logs/{stage}.log')
    print(stage,'complete',flush=True)

def wait(path,stage):
    status(stage,waiting_for=str(path))
    while not path.exists():time.sleep(5)

def start_background(stage,script,args=(),python=PYNK):
    if cached(script,args):
        print(stage,"already complete; retained",flush=True);return dict(done=True)
    command=['uv','run','--no-project','--python',python,'python',str(ROOT/'scripts'/script),*map(str,args)]
    record=dict(stage=stage,command=command,cwd=str(ROOT),CUDA_VISIBLE_DEVICES='1',PYTORCH_ALLOC_CONF=os.environ.get('PYTORCH_ALLOC_CONF','native default'),started_unix=time.time(),execution='Independent CPU worker; GPU reconstruction remains sequential')
    with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    with (ROOT/'logs'/f'{stage}.log').open('a') as log:process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
    job=dict(process=process,record=record,done=False);BACKGROUND.append(job);return job

def finish_background(job):
    if job['done']:return
    rc=job['process'].wait();job['done']=True;record=job['record'];record.update(returncode=rc,ended_unix=time.time())
    with (ROOT/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if rc:raise RuntimeError(f'{record["stage"]} failed; see logs/{record["stage"]}.log')
    print(record['stage'],'complete',flush=True)

try:
    wait(ROOT/'E11_2DGS_Corrected/training_complete.json','waiting_E11_30K')
    run('E11_surface_render','render_surface.py',['--experiment','E11'],PY2D)
    prep={'E11':start_background('E11_surface_input','prepare_rendered_inputs.py',['--experiment','E11'])}
    run('E11_common_evaluation','evaluate_e11.py',python=PYGG)
    run('E10_surface_render','render_surface.py',['--experiment','E10'],PYGG)
    prep['E10']=start_background('E10_surface_input','prepare_rendered_inputs.py',['--experiment','E10'])
    run('E8_surface_render','render_surface.py',['--experiment','E8'],PYGG)
    prep['E8']=start_background('E8_surface_input','prepare_rendered_inputs.py',['--experiment','E8'])
    tsdf=start_background('native_TSDF_matrix','run_tsdf_worker.py')
    wait(ROOT/'configs/nksr_environment_ready.json','waiting_NKSR_environment')
    for method in ['M1_Omega_NKSR','M2_Standard_NKSR','M3_E10_GGGS_NKSR','E8_GGGS_NKSR'] + ([] if (ROOT/'meshes/supplementary/E11_2DGS_NKSR/RESOURCE_LIMIT.json').exists() else ['E11_2DGS_NKSR']):
        dependency={'M3_E10_GGGS_NKSR':'E10','E8_GGGS_NKSR':'E8','E11_2DGS_NKSR':'E11'}.get(method)
        if dependency:finish_background(prep[dependency])
        run(method,'extract_nksr.py',['--experiment',method])
    for job in prep.values():finish_background(job)
    finish_background(tsdf)
    methods=['M1_Omega_NKSR','M2_Standard_NKSR','M3_E10_GGGS_NKSR','M4_E10_GGGS_TSDF','M5_E11_2DGS_TSDF','E8_GGGS_NKSR'] + (['E11_2DGS_NKSR'] if (ROOT/'meshes/supplementary/E11_2DGS_NKSR/complete.json').exists() else [])
    evaluations=[]
    for method in methods:
        folder=ROOT/'meshes'/method if method.startswith('M') else ROOT/'meshes/supplementary'/method
        evaluations.append(start_background('evaluate_'+method,'evaluate_meshes.py',['--mesh-dir',folder,'--method',method],PY2D))
        if len(evaluations)>=2:finish_background(evaluations.pop(0))
        if method.startswith(('M4','M5')):
            evaluations.append(start_background('evaluate_'+method+'_alpha_filtered','evaluate_meshes.py',['--mesh-dir',folder/'supplementary_alpha_filtered','--method',method+'/alpha_filtered'],PY2D))
            if len(evaluations)>=2:finish_background(evaluations.pop(0))
    for job in evaluations:finish_background(job)
    run('collect_comparisons','collect_comparisons.py',python=PY2D)
    run('build_visualizations','build_visualizations.py',python=PY2D)
    status('all_mesh_experiments_complete_analysis_pending')
except Exception as e:
    # Let already authorized independent CPU jobs finish; avoid orphaned outputs
    # and overlapping writers when the experiment is resumed after a fix.
    for job in BACKGROUND:
        try:finish_background(job)
        except Exception:pass
    (ROOT/'MESH_STATUS.json').write_text(json.dumps(dict(status='failed',error=str(e),physical_gpu=1),indent=2)+'\n');raise
