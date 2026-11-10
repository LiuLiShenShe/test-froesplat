"""Resumable physicalGPU1 queue. Each experiment validated by real output files."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MPLBACKEND='Agg')
import subprocess,json,time,sys,fcntl
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];BASE=OUT.parents[2]
PY=str(BASE/'.venv/bin/python')
from queue_io import write_status as write
lock=(OUT/'gpu_queue_execution.lock').open('a+')
fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
def wait_for(p):
    while not p.exists():
        write(OUT/'GPU_QUEUE_STATUS.json',dict(status='waiting_for_dependency',dependency=str(p),pid=os.getpid(),time=time.time()))
        time.sleep(5)
def run(name,cmd,required,cwd=BASE):
    if all(p.exists() for p in required):print(name,'already complete',flush=True);return
    write(OUT/'GPU_QUEUE_STATUS.json',dict(status='running',stage=name,command=cmd,pid=os.getpid(),time=time.time()))
    log=OUT/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True);start=time.time()
    rec=dict(stage=name,command=cmd,cwd=str(cwd),started=start)
    with (OUT/'actual_commands.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    with log.open('a') as f:
        child=subprocess.Popen(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=os.environ)
        write(OUT/'GPU_CHILD_PROCESS.json',dict(stage=name,pid=child.pid,log=str(log)))
        rc=child.wait()
    rec.update(returncode=rc,ended=time.time(),wall_seconds=time.time()-start)
    with (OUT/'actual_commands.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    if rc or not all(p.exists() for p in required):
        write(OUT/'GPU_QUEUE_STATUS.json',dict(status='failed',stage=name,returncode=rc,log=str(log),missing=[str(p) for p in required if not p.exists()]))
        raise SystemExit(rc or 1)
    print(name,'complete',flush=True)
def uv(cmd,py=PY):return ['uv','run','--no-project','--python',py,'python',*map(str,cmd)]
wait_for(OUT/'P1_GGGS_Tetrahedra/complete.json')
wait_for(OUT/'P2_GGGS_Cleaning/cleaning_complete.json')
p2=OUT/'P2_GGGS_Cleaning'
run('P2_render_cleaned',uv([OUT/'scripts/render_gggs.py','--ply',p2/'cleaned_gaussians.ply','--out',p2/'surface_inputs','--experiment','P2']),[p2/'surface_inputs/render_complete.json'])
run('P2_tetra',uv([OUT/'scripts/extract_tetra.py','--source-ply',p2/'cleaned_gaussians.ply','--out',p2/'Tetrahedra']),[p2/'Tetrahedra/complete.json'])
p3=OUT/'P3_GS2Mesh';p3py=str(p3/'.venv/bin/python')
run('P3_smoke',uv([p3/'scripts/run_p3_driver.py','--smoke-only'],p3py),[p3/'stereo_outputs/smoke_placeholder.json'] if False else [],cwd=p3) if False else None
run('P3_extract',uv([p3/'scripts/run_p3_driver.py','--extraction-only'],p3py),[p3/'fusion_complete.json',p3/'fusion_shared_depth_range_complete.json'],cwd=p3)
wait_for(OUT/'P4_2DGS_Depth_Mesh/complete.json')
e12=OUT/'E12_GGGS_EO'
cmd=json.loads((e12/'launch_command.json').read_text())['command']
run('E12_train',cmd,[e12/'checkpoints/chkpnt30000.pth',e12/'checkpoints/point_cloud/iteration_30000/point_cloud.ply',e12/'exposure_params/exposure_matrices.npz'])
run('E12_render',uv([OUT/'scripts/render_gggs.py','--ply',e12/'checkpoints/point_cloud/iteration_30000/point_cloud.ply','--out',e12/'surface_inputs','--experiment','E12','--exposure',e12/'exposure_params/exposure_matrices.npz','--repo',e12/'source']),[e12/'surface_inputs/render_complete.json'])
run('E12_tetra',uv([OUT/'scripts/extract_tetra.py','--source-ply',e12/'checkpoints/point_cloud/iteration_30000/point_cloud.ply','--out',e12/'Tetrahedra']),[e12/'Tetrahedra/complete.json'])
e13=OUT/'E13_QGS'
e13args=[e13/'scripts/run_e13.py']
recovery=e13/'RECOVERY_REQUEST.json'
preset_request=e13/'NATIVE_PRESET_REQUEST.json'
if preset_request.is_file():
    request=json.loads(preset_request.read_text())
    assert request['fresh_from_same_VGGT_initialization'] and not recovery.exists()
    e13args+=['--native-preset',request['native_preset'],'--allocator',request['allocator'],'--checkpoint-every',str(request['checkpoint_every'])]
elif recovery.is_file() and not (e13/'training_complete.json').is_file():
    request=json.loads(recovery.read_text());checkpoint=Path(request['resume_checkpoint']).resolve()
    assert checkpoint.is_file()
    e13args+=['--resume',checkpoint,'--allocator',request['allocator'],'--checkpoint-every',str(request['rolling_checkpoint_every'])]
run('E13_native',uv(e13args,str(e13/'.venv/bin/python')),[e13/'training_complete.json',e13/'evaluation_complete.json',e13/'meshes/native_TSDF/complete.json'],cwd=e13)
write(OUT/'GPU_QUEUE_STATUS.json',dict(status='prepared_stages_complete_E14_pending',pid=os.getpid()))

