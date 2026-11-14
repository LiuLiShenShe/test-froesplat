#!/usr/bin/env python3
"""Verify cached FG geometry/contours against independent completed GGGS metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import json
import time
import numpy as np
import torch
import yaml
from scipy import ndimage
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import native_subset
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.evaluation import Evaluator
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import contour_metrics
from ForeSplat_AutoGaussianCleaner.cleaner.foreground_protection import regions
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays


def main():
    base=Path(__file__).resolve().parent;c=base/'experiments/GGGS_Cleaner_Comparison'
    cfg=yaml.safe_load((base/'configs/default.yaml').read_text());spec=json.loads((base/'configs/scenes/KongQueZhuYu.json').read_text())
    torch.set_num_threads(8);scene=Scene(spec,cfg);original,rows=scene.load_model(spec['model'])
    group=json.loads((c/'C5_Fused_GGGS/development_v3/C5-C/rounds/01/group_001/decision.json').read_text())
    keep=np.ones(len(rows),bool);keep[np.asarray(group['removed_ids'],np.int64)]=False
    safe=native_subset(scene,original,keep);history,_=scene.load_model(c/'baselines/C3/cleaned_gaussians.ply')
    completed=[json.loads((c/'baselines/C0/cleanup_metrics.json').read_text())['metrics'],
        json.loads(Path(group['metrics']).read_text()),json.loads((c/'baselines/C3/cleanup_metrics.json').read_text())['metrics']]
    evaluator=object.__new__(Evaluator);evaluator.scene=scene;evaluator.geometry_cache={};evaluator.contour_cache={};evaluator.cache_hits=dict(geometry=0,contour=0)
    results=[]
    for index in [0,45,100]:
        camera,target,mask,*_=scene.get_view(index);r=regions(mask,cfg['boundary_relative_width'])
        r['contour_context']=ndimage.distance_transform_edt(~mask)<=max(2,3*r['radius'])
        with np.load(base/f'experiments/all_19_plants_v3/development/KongQueZhuYu/C/comparison/initial/{index:04d}.npz') as z:old={k:z[k] for k in ['alpha','depth','normal']}
        for label,model,reference in zip(['C0','safe_group_3461','C3_changed_geometry'],[original,safe,history],completed):
            with torch.no_grad():a=arrays(scene.render(camera,model))
            start=time.perf_counter();actual={**evaluator.geometry_cached(index,a,old,mask),**evaluator.contour_cached(index,a['alpha'],old['alpha'],mask,r)};first=time.perf_counter()-start
            expected=next(v for v in reference['views'] if v['name']==camera.image_name)
            contour_expected=contour_metrics(a['alpha'],old['alpha'],mask,r['radius'])
            comparison={**contour_expected,**expected}
            maximum=max(abs(value-comparison[key]) for key,value in actual.items())
            assert maximum==0.,(label,camera.image_name,maximum)
            hits_before=dict(evaluator.cache_hits);start=time.perf_counter();again={**evaluator.geometry_cached(index,a,old,mask),**evaluator.contour_cached(index,a['alpha'],old['alpha'],mask,r)};second=time.perf_counter()-start
            assert again==actual and all(evaluator.cache_hits[k]>hits_before[k] for k in hits_before)
            results.append(dict(case=label,view=camera.image_name,max_independent_metric_delta=maximum,
                first_seconds=first,cached_seconds=second,depth_changed_fraction=actual['depth_changed_fraction'],
                hits_before_repeat=hits_before,hits_after_repeat=dict(evaluator.cache_hits)))
    assert any(r['depth_changed_fraction']>0 for r in results if r['case']=='C3_changed_geometry')
    write_json(c/'comparisons/GEOMETRY_CONTOUR_CACHE_EQUIVALENCE.json',dict(cases=results,
        changed_real_geometry_not_hidden=True,independent_geometry_source='Completed common C0/C3 evaluation and completed native group-001 evaluation',
        contour_source='Uncached contour function; old C0/C3 evaluations predate this additional foreground-local metric',
        evaluator_source_sha256=sha(base/'cleaner/adaptive/evaluation.py')))
    print('All nine real native renders match independent completed metrics exactly; changed C3 geometry remains nonzero')


if __name__=='__main__':main()
