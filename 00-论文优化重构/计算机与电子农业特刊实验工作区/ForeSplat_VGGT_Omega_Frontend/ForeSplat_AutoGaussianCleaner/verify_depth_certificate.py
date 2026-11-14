#!/usr/bin/env python3
"""Native verification of the new recovery certificate, without accepting a proposal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import json
import numpy as np
import torch
import yaml
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import native_subset
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.evaluation import Evaluator
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.depth_recovery import certified,potential_tradeoff
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays


def main():
    base=Path(__file__).resolve().parent;c=base/'experiments/GGGS_Cleaner_Comparison'
    cfg=yaml.safe_load((base/'configs/default.yaml').read_text());cfg.update(yaml.safe_load((base/'configs/adaptive_recovery.yaml').read_text()))
    spec=json.loads((base/'configs/scenes/KongQueZhuYu.json').read_text());scene=Scene(spec,cfg);torch.set_num_threads(8)
    original,rows=scene.load_model(spec['model']);legacy=base/'experiments/all_19_plants_v3/development/KongQueZhuYu/C'
    item=next(e for e in json.loads((legacy/'edit_manifest.json').read_text())['edits'] if e['action']=='prune' and len(e['removed_ids'])==31955)
    keep=np.ones(len(rows),bool);keep[np.asarray(item['removed_ids'],np.int64)]=False
    proposal=native_subset(scene,original,keep)
    initial_metrics=json.loads((legacy/'comparison/initial/metrics.json').read_text());proposal_metrics=json.loads(Path(item['metrics']).read_text())
    names={r.split(':',1)[0] for r in item['reasons'] if 'depth' in r and ':' in r};reports=[]
    evaluator=object.__new__(Evaluator);evaluator.scene=scene;evaluator.geometry_cache={};evaluator.cache_hits=dict(geometry=0);evaluator.regions={}
    for index,v in enumerate(scene.views):
        if v['name'] not in names:continue
        camera,_,mask,*_=scene.get_view(index);evaluator.regions[index]=regions(mask,cfg['boundary_relative_width'])
        with np.load(legacy/f'comparison/initial/{index:04d}.npz') as z:old={k:z[k] for k in ['alpha','depth','normal']}
        with torch.no_grad():a=arrays(scene.render(camera,original));b=arrays(scene.render(camera,proposal))
        iv=next(x.copy() for x in initial_metrics['views'] if x['index']==index);cv=next(x.copy() for x in proposal_metrics['views'] if x['index']==index)
        iv.update(evaluator.geometry_cached(index,a,old,mask));cv.update(evaluator.geometry_cached(index,b,old,mask))
        certificate=certified(cv,iv,iv,cfg)
        depth_only=potential_tradeoff([v['name']+':depth_changed_fraction'],[cv],[iv],[iv],cfg)
        normal_failure=potential_tradeoff([v['name']+':depth_changed_fraction,normal_angle_p95_deg'],[cv],[iv],[iv],cfg)
        full_local=potential_tradeoff([r for r in item['reasons'] if r.startswith(v['name']+':')],[cv],[iv],[iv],cfg)
        assert not normal_failure
        reports.append(dict(view=v['name'],original=iv,candidate=cv,depth_certificate=certificate,
            depth_only_may_require_complete_review=depth_only,normal_failure_never_excused=not normal_failure,
            complete_local_reason_set_may_require_review=full_local))
    write_json(c/'comparisons/NATIVE_DEPTH_RECOVERY_CERTIFICATE_VERIFICATION.json',dict(views=reports,
        input_sha256=sha(spec['model']),config=cfg,
        scope='Actual native rejected proposal; depth certificate alone does not accept the full model or override color/structure constraints'))
    print(json.dumps([{k:v for k,v in r.items() if k not in ['original','candidate']} for r in reports],indent=2))


if __name__=='__main__':main()
