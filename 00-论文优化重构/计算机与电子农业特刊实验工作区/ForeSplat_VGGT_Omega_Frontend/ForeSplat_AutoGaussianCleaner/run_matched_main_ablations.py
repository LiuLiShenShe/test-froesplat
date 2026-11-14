#!/usr/bin/env python3
"""Serialize explicit same-source/config/input main ablations, with honest process wall costs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,json,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrypoint',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--methods',nargs='+',choices=['C5-A','C5-B','C5-D'],required=True)
    args=parser.parse_args();out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=True);records=[]
    for method in args.methods:
        command=['uv','run','--no-project','--python',sys.executable,'python','-u',str(args.entrypoint.resolve()),
            '--scene',str(args.scene.resolve()),'--method',method,'--config',str(args.config.resolve()),
            '--output',str(out/method),'--resume']
        start=time.perf_counter();log=out/(method+'_process.log')
        with log.open('a') as handle:
            child=subprocess.Popen(command,cwd=args.entrypoint.resolve().parent.parent,env=os.environ.copy(),stdout=handle,stderr=subprocess.STDOUT)
            write_json(out/'MATCHED_ABLATION_PROGRESS.json',dict(method=method,driver_pid=os.getpid(),uv_pid=child.pid,command=command,log=str(log),started=time.time()))
            code=child.wait()
        records.append(dict(method=method,exit_code=code,child_wall_seconds=time.perf_counter()-start,log=str(log)))
        write_json(out/'MATCHED_ABLATION_PROCESS_COSTS.json',records)
        if code:raise SystemExit(code)

if __name__=='__main__':main()
