#!/usr/bin/env python3
"""Replay real KongQueZhuYu development groups with complete and early rejection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import yaml
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import native_subset
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.evaluation import Evaluator
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import choose
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--visibility',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups',type=int,nargs='+',default=[1,8,19])
    a=p.parse_args();base=Path(__file__).resolve().parent;torch.set_num_threads(8)
    cfg=yaml.safe_load((base/'configs/default.yaml').read_text());cfg.update(yaml.safe_load((base/'configs/adaptive_surface.yaml').read_text()))
    spec=json.loads(a.scene.read_text());scene=Scene(spec,cfg);model,rows=scene.load_model(spec['model'])
    import lpips
    lp=lpips.LPIPS(net='alex').cuda().eval();surface=external_surface(scene,scene.evidence_indices())
    evaluator=Evaluator(scene,a.output,a.baseline,lp,surface)
    sources={str(f.relative_to(base)):sha(f) for f in [base/'cleaner/adaptive/evaluation.py',base/'cleaner/adaptive/selection.py',Path(__file__).resolve()]}
    initial=json.loads((a.run/'comparison/initial/metrics.json').read_text())
    evaluator.initial_records={v['index']:v for v in initial['views']}
    with np.load(a.visibility) as z:evaluator.visibility={i:x.copy() for i,x in enumerate(z['visibility'])}
    current=initial;keep=np.ones(len(rows),bool);results=[]
    for i in range(1,20):
        decision=json.loads((a.run/f'rounds/01/group_{i:03d}/decision.json').read_text())
        group=np.asarray(decision['removed_ids'],np.int64)
        if i in a.groups:
            candidate_keep=keep.copy();candidate_keep[group]=False
            candidate=native_subset(scene,model,candidate_keep)
            cfg['rejection_preflight']=False;start=time.perf_counter()
            full,affected=evaluator.evaluate_group(candidate,a.output/f'group_{i:03d}/complete',group,current)
            full_seconds=time.perf_counter()-start;accepted_full,_,_,reasons_full=choose(full,current,initial,cfg)
            cfg['rejection_preflight']=True;start=time.perf_counter()
            probe,_=evaluator.evaluate_group(candidate,a.output/f'group_{i:03d}/preflight',group,current)
            probe_seconds=time.perf_counter()-start;accepted_probe,_,_,reasons_probe=choose(probe,current,initial,cfg)
            assert accepted_probe==accepted_full==decision['accepted']
            measured=set(probe.get('rerendered_views',affected))
            reference={v['index']:v for v in full['views']}
            maximum=0.
            for v in probe['views']:
                if v['index'] not in measured:continue
                for k,value in v.items():
                    if isinstance(value,(int,float)) and k!='index':maximum=max(maximum,abs(value-reference[v['index']][k]))
            assert maximum==0.
            if not probe.get('metrics_complete',True):
                assert set(reasons_probe).issubset(set(reasons_full))
                first_complete=next(r for r in reasons_full if r.split(':',1)[0] in {v['name'] for v in full['views']})
                assert reasons_probe[0]==first_complete
                assert probe['canonical_first_rejection_preserved']
            results.append(dict(group=i,candidates=len(group),accepted=accepted_probe,
                complete_views=len(affected),preflight_measured_views=len(measured),metrics_complete=probe.get('metrics_complete',True),
                maximum_measured_metric_delta=maximum,full_seconds=full_seconds,preflight_seconds=probe_seconds,
                complete_reasons=reasons_full,preflight_reasons=reasons_probe,
                timing_note='Same GPU and actual candidates; native view count is exact. Replayed metrics use warm RGB cache, wall-time improvement is not an isolated throughput estimate'))
            del candidate;print(results[-1],flush=True)
        if decision['accepted']:
            keep[group]=False;current=json.loads(Path(decision['metrics']).read_text())
    write_json(a.output/'PREFLIGHT_EQUIVALENCE.json',dict(cases=results,all_decisions_identical=True,
        measured_metrics_identical=True,acceptance_requires_complete_metrics=True,canonical_first_rejection_identical=True,
        sources=sources,scene=spec))


if __name__=='__main__':main()
