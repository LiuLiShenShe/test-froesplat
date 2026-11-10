"""Queue E14 only after preceding authorized formal stages and actual readiness."""
import json,time,subprocess,os,fcntl
from pathlib import Path
from queue_io import read_status,write_status
OUT=Path(__file__).resolve().parents[1];E14=OUT/'E14_2D_SuGaR';E13=OUT/'E13_QGS'
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MPLBACKEND='Agg')
lock=(OUT/'e14_queue_execution.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
    mesh=read_status(E13/'meshes/native_TSDF/complete.json')
    ready=read_status(E14/'environment_ready.json')
    upstream=read_status(E13/'queue_status.json')
    summary=read_status(E13/'execution_summary_complete.json')
    if mesh.get('status')=='complete' and upstream.get('status')=='complete' and summary.get('status')=='complete' and ready.get('status') in ['ready','passed','complete']:break
    if upstream.get('status') in ['failed','partial','failed_dependency']:
        write_status(OUT/'E14_QUEUE_STATUS.json',dict(status='failed_dependency',dependency=str(E13/'queue_status.json'),upstream=upstream,pid=os.getpid()))
        raise SystemExit(1)
    write_status(OUT/'E14_QUEUE_STATUS.json',dict(status='waiting_for_E13_all_native_stages_and_environment',pid=os.getpid(),requires_native_viewer_and_summary=True));time.sleep(5)
cmd=['uv','run','--no-project','--python',str(E14/'.venv/bin/python'),'python',str(E14/'scripts/run_all_e14.py')]
write_status(OUT/'E14_QUEUE_STATUS.json',dict(status='running',pid=os.getpid(),command=cmd));start=time.time()
with (OUT/'logs/E14_all_stages.log').open('a') as f:
    p=subprocess.Popen(cmd,cwd=E14,env=os.environ,stdout=f,stderr=subprocess.STDOUT)
    write_status(OUT/'E14_CHILD_PROCESS.json',dict(pid=p.pid,command=cmd));rc=p.wait()
status=read_status(E14/'STATUS.json')
write_status(OUT/'E14_QUEUE_STATUS.json',dict(status='complete' if rc==0 and status.get('status')=='complete' else 'failed',returncode=rc,wall_seconds=time.time()-start,native_status=status))
raise SystemExit(rc or (0 if status.get('status')=='complete' else 1))

