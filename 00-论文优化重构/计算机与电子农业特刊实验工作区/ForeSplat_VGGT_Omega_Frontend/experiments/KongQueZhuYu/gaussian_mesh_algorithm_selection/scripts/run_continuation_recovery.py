"""Resume real outputs after the reproduced E14 Open3D dependency failure."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parents[1]
E14 = OUT / 'E14_2D_SuGaR'
STATUS = OUT / 'CONTINUATION_RECOVERY_STATUS.json'

def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def write(**values):
    values.update(pid=os.getpid(), updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    temp=STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(values,ensure_ascii=False,indent=2))
    temp.replace(STATUS)

def run(name, script):
    cmd=['python3',str(OUT/'scripts'/script)]
    write(status='running',stage=name,command=cmd)
    start=time.time()
    with (OUT/'logs'/('continuation_'+name+'.log')).open('a') as log:
        child=subprocess.Popen(cmd,cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
        write(status='running',stage=name,command=cmd,child_pid=child.pid)
        rc=child.wait()
    with (OUT/'actual_commands.jsonl').open('a') as journal:
        journal.write(json.dumps(dict(stage='continuation_'+name,command=cmd,returncode=rc,wall_seconds=time.time()-start))+'\n')
    if rc:
        write(status='failed',stage=name,returncode=rc,log=str(log.name))
        raise SystemExit(rc)

def main():
    with (OUT/'continuation_recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            result=read(E14/'native_render_mesh_complete.json')
            meshes=result.get('mesh_results',[])
            if result.get('open3d')=='0.19.0' and len(meshes)==3 and all(r.get('status')=='complete' and r.get('post_status')=='complete' for r in meshes):
                break
            process=read(E14/'recovery_open3d019/process.json')
            try:
                os.kill(process['pid'],0)
                stat=Path('/proc')/str(process['pid'])/'stat'
                if stat.read_text().split(') ',1)[1].startswith('Z '):raise ProcessLookupError()
            except (KeyError,ProcessLookupError,FileNotFoundError):
                write(status='failed',stage='cached_mesh_fusion',reason='Process stopped before three nonempty Mesh stages completed',evidence=result)
                raise SystemExit(1)
            write(status='waiting',stage='cached_mesh_fusion',process_pid=process['pid'])
            time.sleep(5)
        run('E14_native_remaining_stages','run_e14_queue.py')
        run('historical208_renders','run_history_queue.py')
        run('all_native_reflection_correspondences','run_analysis_queue.py')
        run('full_model_metric_visual_delivery','run_finalization_queue.py')
        write(status='review_ready',stage='scientific_review',validation=str(OUT/'artifact_index/FULL_DELIVERY_VALIDATION.json'))

if __name__=='__main__':
    main()
