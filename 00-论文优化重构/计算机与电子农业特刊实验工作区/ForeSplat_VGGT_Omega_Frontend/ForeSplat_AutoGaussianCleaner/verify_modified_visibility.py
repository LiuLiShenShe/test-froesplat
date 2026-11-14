#!/usr/bin/env python3
"""Verify unchanged primitive footprint reuse after local covariance refinement."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
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
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha


def main():
    base=Path(__file__).resolve().parent;c=base/'experiments/GGGS_Cleaner_Comparison'
    cfg=yaml.safe_load((base/'configs/default.yaml').read_text())
    spec=json.loads((base/'configs/scenes/KongQueZhuYu.json').read_text())
    scene=Scene(spec,cfg);torch.set_num_threads(8)
    original,rows=scene.load_model(spec['model'])
    fixture=c/'comparisons/compact_checkpoint_verify'
    changed,changed_rows=scene.load_model(fixture/'materialized.ply')
    assert len(changed_rows)==len(rows)
    fields=[k for k in rows.dtype.names if k.startswith(('scale_','rot_')) or k=='opacity']
    modified=np.flatnonzero(np.any(np.column_stack([rows[k]!=changed_rows[k] for k in fields]),axis=1))
    assert len(modified)>0
    group=np.asarray(json.loads((c/'C5_Fused_GGGS/development_v3/C5-C/rounds/01/group_001/decision.json').read_text())['removed_ids'],np.int64)
    assert not np.intersect1d(group,modified).size
    evaluator=object.__new__(Evaluator)
    evaluator.scene=scene;evaluator.force_full=True;evaluator.modified_original_ids=set(modified.tolist())
    with np.load(c/'comparisons/initial_native_visibility.npz') as z:
        assert str(z['original_model_sha256'])==sha(spec['model'])
        packed=z['visibility'];evaluator.visibility={i:v for i,v in enumerate(packed)}
    affected=evaluator.affected_views(group)
    assert len(affected)<len(scene.views)
    assert evaluator.affected_views(modified[:1])==list(range(len(scene.views)))
    keep=np.ones(len(rows),bool);keep[group]=False
    candidate=native_subset(scene,changed,keep)
    records=[]
    for index in sorted(set(range(len(scene.views)))-set(affected)):
        camera,*_=scene.get_view(index)
        with torch.no_grad():
            before=scene.render(camera,changed);after=scene.render(camera,candidate)
        deltas={k:float((before[k]-after[k]).abs().max()) for k in ['render','mask','median_depth','normal']}
        assert max(deltas.values())==0.,(camera.image_name,deltas)
        records.append(dict(index=index,view=camera.image_name,max_native_deltas=deltas))
    write_json(c/'comparisons/MODIFIED_PRIMITIVE_VISIBILITY_CACHE_VERIFICATION.json',dict(
        original_model_sha256=sha(spec['model']),refined_fixture_sha256=sha(fixture/'materialized.ply'),
        modified_gaussians=len(modified),unmodified_prune_group=len(group),rerendered_views=len(affected),
        all_unaffected_views_checked=records,groups_touching_modified_members_rerender_all_views=True,
        evaluator_source_sha256=sha(base/'cleaner/adaptive/evaluation.py'),
        conclusion='All four native render outputs remain exactly unchanged in every reused view after pruning unchanged members from a model with local covariance edits'))
    print('Verified',len(records),'unchanged views after local covariance edits; modified-member groups require all views',flush=True)


if __name__=='__main__':main()
