"""Sequential, resumable P3 execution. Start only when GPU stage order permits."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["OMP_NUM_THREADS"]="8"
os.environ["MPLBACKEND"]="Agg"
import argparse,json,subprocess,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from render_e10_stereo import write

def run(stage,args):
    log=ROOT/"logs"/(stage+".log")
    command=[str(ROOT/".venv/bin/python"),*args]
    record=dict(stage=stage,command=command,cwd=str(ROOT),started_unix=time.time(),physical_gpu=1)
    with (ROOT/"actual_commands.jsonl").open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
    write(ROOT/"STATUS.json",dict(status="running",stage=stage,pid=os.getpid(),command=command))
    with log.open("a") as f:
        p=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
        write(ROOT/"CHILD_PROCESS.json",dict(stage=stage,pid=p.pid,parent_pid=os.getpid(),log=str(log)))
        stop=threading.Event()
        samples=[]
        def observe():
            with (ROOT/"logs"/(stage+"_gpu_observations.jsonl")).open("a") as observation:
                while not stop.is_set():
                    try:
                        apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True,timeout=10)
                        memory=max([float(line.split(',')[1]) for line in apps.splitlines() if line.split(',')[0].strip()==str(p.pid)] or [0.])
                        gpu=subprocess.check_output(["nvidia-smi","--id=1","--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw","--format=csv,noheader,nounits"],text=True,timeout=10).strip()
                        sample=dict(unix=time.time(),pid=p.pid,child_gpu_memory_mib=memory,physical_gpu1_aggregate=gpu,aggregate_includes_other_processes=True)
                        samples.append(sample);observation.write(json.dumps(sample)+"\n");observation.flush()
                    except Exception as e:
                        observation.write(json.dumps(dict(unix=time.time(),observation_error=str(e)))+"\n");observation.flush()
                    stop.wait(2)
        monitor=threading.Thread(target=observe,daemon=True);monitor.start()
        code=p.wait()
        stop.set();monitor.join(timeout=12)
    record.update(returncode=code,ended_unix=time.time(),elapsed_seconds=time.time()-record["started_unix"])
    record.update(peak_child_gpu_memory_mib_sampled=max([s["child_gpu_memory_mib"] for s in samples] or [0]),observations=len(samples))
    with (ROOT/"actual_commands.jsonl").open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
    if code:
        write(ROOT/"STATUS.json",dict(status="failed",stage=stage,returncode=code,log=str(log)))
        raise RuntimeError(stage+" failed; inspect "+str(log))
    print(stage,"complete",flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--extraction-only",action="store_true")
    a=parser.parse_args()
    if a.smoke_only:
        run("render_smoke",["scripts/render_e10_stereo.py","--limit","1"])
        run("stereo_smoke",["scripts/run_stereo_fusion.py","--stage","stereo","--limit","1"])
        write(ROOT/"STATUS.json",dict(status="smoke_complete",formal_views_remaining=207))
        return
    if not (ROOT/"render_complete.json").exists():run("render",["scripts/render_e10_stereo.py"])
    if not (ROOT/"stereo_complete.json").exists():run("stereo",["scripts/run_stereo_fusion.py","--stage","stereo"])
    print("STEREO_GPU_COMPLETE; remaining stages use CPU",flush=True)
    if not (ROOT/"fusion_complete.json").exists():run("fusion",["scripts/run_stereo_fusion.py","--stage","fusion"])
    if not (ROOT/"fusion_shared_depth_range_complete.json").exists():
        run("fusion_shared_depth_range",["scripts/run_stereo_fusion.py","--stage","fusion","--shared-depth-range"])
    run("artifact_index",["scripts/index_p3.py"])
    write(ROOT/"STATUS.json",dict(status="model_extraction_complete",views=208,evaluation_status="pending"))
    if a.extraction_only:return
    if not (ROOT/"meshes/evaluation/complete.json").exists():run("evaluation",["scripts/evaluate_p3.py"])
    if not (ROOT/"meshes_shared_depth_range/evaluation/complete.json").exists():
        run("evaluation_shared_depth_range",["scripts/evaluate_p3.py","--shared-depth-range"])
    if not (ROOT/"visualizations/protocol.json").exists():run("visualizations",["scripts/build_p3_views.py"])
    write(ROOT/"STATUS.json",dict(status="complete",views=208,mesh_routes=2,raw_post_preserved=True))

if __name__=="__main__":main()

