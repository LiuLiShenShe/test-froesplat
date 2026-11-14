#!/usr/bin/env python3
"""Prepare missing 30K GGGS checkpoints from existing bundles, without inference.

Training is independent of cleanup development/test selection. Existing live
training is observed, never duplicated; partial checkpoints are not accepted.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import argparse
import csv
import fcntl
import json
import subprocess
import sys
import time
import re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json


def live_training(events):
    matches=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:args=(process/'cmdline').read_bytes().split(b'\0')
        except (OSError,PermissionError):continue
        words=[a.decode(errors='replace') for a in args]
        if str(events) in words and any(w.endswith(('train_observed.py','train_observed_resume.py')) for w in words):
            matches.append(int(process.name))
    return matches


def validate(spec):
    path=Path(spec['model'])
    if 'iteration_30000' not in path.parts:
        raise ValueError('A final iteration_30000 native checkpoint is required')
    from plyfile import PlyData
    p=PlyData.read(str(path),mmap='r')['vertex'].data
    required=['x','y','z','opacity','filter_3D']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    if not all(k in p.dtype.names for k in required):raise ValueError('Incomplete GGGS native fields')
    return dict(gaussians=len(p),native_ply=str(path),sha256=sha(path))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--plant',action='append')
    p.add_argument('--recover-interrupted',action='store_true',help='Explicitly restore an owned missing-model native checkpoint, preserving interrupted logs and recording repeated work')
    a=p.parse_args();base=Path(__file__).resolve().parent;root=base.parent
    out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads(a.manifest.read_text());status={}
    observer=root/'experiments/KongQueZhuYu/geometry_rgb_dual_frontend/scripts/train_observed.py'
    jobs=[j for j in manifest['plants'] if not a.plant or j['plant'] in a.plant]
    # Follow already-started training before creating another job on GPU1.
    jobs.sort(key=lambda j:(not bool(live_training(Path(j['training_dir'])/'densification_events.jsonl')),j['plant']))
    def persist():
        write_json(out/'TRAINING_STATUS.json',status)
        rows=[dict(plant=j['plant'],group=j.get('group','production'),
            **status.get(j['plant'],dict(status='pending',native_ply=json.loads(Path(j['scene']).read_text())['model']))) for j in jobs]
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with (out/'MODEL_PREPARATION.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    for j in jobs:
        spec=json.loads(Path(j['scene']).read_text());training=Path(j['training_dir']);events=training/'densification_events.jsonl'
        complete=events.with_suffix('.complete.json');beg=time.perf_counter()
        try:
            if Path(spec['model']).exists():
                status[j['plant']]=dict(status='reused_final_gggs_checkpoint',**validate(spec))
                if complete.exists():status[j['plant']]['training_seconds']=json.loads(complete.read_text())['wall_seconds']
                persist();continue
            live=live_training(events)
            if live:
                status[j['plant']]=dict(status='observing_existing_training',pids=live,native_ply=spec['model']);persist()
                while live_training(events) and not complete.exists():time.sleep(10)
                if not complete.exists():raise RuntimeError('Existing training ended without completion marker; checkpoint recovery required')
                status[j['plant']]=dict(status='completed_existing_training',**validate(spec),
                    training_seconds=json.loads(complete.read_text())['wall_seconds'],observation_seconds=time.perf_counter()-beg)
                persist();continue
            recovery=None
            if events.exists() and not complete.exists():
                if not a.recover_interrupted:
                    raise RuntimeError('Interrupted GGGS training: preserve partial files and recover checkpoint; no silent restart')
                previous=json.loads((training/'actual_training_parameters.json').read_text())
                params=previous['dataset']
                if (Path(params['source_path']).resolve()!=Path(spec['dataset']).resolve() or params['sh_degree']!=3
                        or params['sg_degree']!=0 or params['resolution']!=2 or previous['opt']['iterations']!=30000):
                    raise ValueError('Interrupted checkpoint belongs to another dataset or training configuration')
                if any(sha(Path(spec['gggs_repo'])/name)!=digest for name,digest in previous['source_sha256'].items()):
                    raise ValueError('Native GGGS training sources changed since the interrupted checkpoint')
                candidates=[]
                for path in (training/'checkpoints').glob('chkpnt*.pth'):
                    match=re.fullmatch(r'chkpnt(\d+)\.pth',path.name)
                    if match and 0<int(match[1])<30000:candidates.append((int(match[1]),path))
                if not candidates:raise ValueError('No recoverable owned native GGGS optimizer checkpoint')
                iteration,checkpoint=max(candidates)
                log=out/'logs'/f"{j['plant']}_train30K.log"
                tail=log.read_bytes()[-32768:].decode(errors='replace') if log.exists() else ''
                observed=re.findall(r'(\d+)/30000 \[([0-9:]+)<',tail)
                last,elapsed=(int(observed[-1][0]),observed[-1][1]) if observed else (iteration,'0')
                seconds=sum(int(value)*60**position for position,value in enumerate(reversed(elapsed.split(':'))))
                recovery=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),restored_iteration=iteration,
                    previous_last_observed_iteration=last,previous_observed_seconds_lower_bound=seconds,
                    resumed_native_restore=True,uninterrupted_trajectory_claimed=False)
                write_json(training/'RECOVERY_REQUEST.json',recovery)
                previous_protocol=training/'ADAPTIVE_TRAINING_PROTOCOL.json'
                if previous_protocol.exists():
                    write_json(training/f'ADAPTIVE_TRAINING_PROTOCOL.interrupted_{int(time.time())}.json',json.loads(previous_protocol.read_text()))
            dataset=Path(spec['dataset'])
            if not (dataset/'sparse/0').is_dir() or not (dataset/'images').is_dir() or not (dataset/'masks').is_dir():
                raise FileNotFoundError('Existing corrected camera/RGB/mask dataset missing; frontend inference is disabled')
            training.mkdir(parents=True,exist_ok=True)
            chosen_observer=observer.with_name('train_observed_resume.py') if recovery else observer
            command=['uv','run','--no-project','--python',sys.executable,'python','-u',str(chosen_observer),
                '--repo',spec['gggs_repo'],'--events',str(events)]
            if recovery:
                command+=['--checkpoint',recovery['checkpoint'],'--checkpoint-sha256',recovery['checkpoint_sha256'],
                    '--previous-observed-seconds',str(recovery['previous_observed_seconds_lower_bound']),
                    '--previous-last-iteration',str(recovery['previous_last_observed_iteration'])]
            command+=['--','--source_path',spec['dataset'],
                '--model_path',str(training/'checkpoints'),'--iterations','30000','--resolution','2',
                '--sh_degree','3','--sg_degree','0','--test_iterations','10000','20000','30000',
                '--save_iterations','7000','15000','30000','--checkpoint_iterations','7000','15000','30000',
                '--metrics_csv',str(training/'train_metrics.csv'),'--metrics_interval','100']
            protocol=dict(command=command,seed=0,iterations=30000,physical_gpu=1,frontend=spec['frontend'],
                existing_bundle_reused=True,new_frontend_inference=False,observer_sha256=sha(chosen_observer),recovery=recovery,
                gggs_training_sha256=sha(Path(spec['gggs_repo'])/'train_foresplat.py'),
                purpose='Missing reconstruction inputs only; no cleaner parameters or test cleaning metrics used')
            write_json(training/'ADAPTIVE_TRAINING_PROTOCOL.json',protocol)
            log=out/'logs'/f"{j['plant']}_train30K.log";log.parent.mkdir(exist_ok=True)
            status[j['plant']]=dict(status='training',native_ply=spec['model'],log=str(log),recovery=recovery);persist()
            with log.open('a') as f:
                f.write(json.dumps(protocol)+'\n');f.flush()
                code=subprocess.call(command,cwd=spec['gggs_repo'],env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
            if code or not complete.exists():raise RuntimeError(f'GGGS training failed ({code}); see {log}')
            status[j['plant']]=dict(status='trained_gggs_30K',**validate(spec),
                training_seconds=json.loads(complete.read_text())['wall_seconds'],log=str(log))
        except Exception as e:
            status[j['plant']]=dict(status='failed',reason=str(e),native_ply=spec['model'])
        persist()
    if any(s['status']=='failed' for s in status.values()):raise SystemExit(1)


if __name__=='__main__':main()
