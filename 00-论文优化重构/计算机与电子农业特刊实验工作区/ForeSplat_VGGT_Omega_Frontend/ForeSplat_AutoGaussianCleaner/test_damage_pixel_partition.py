#!/usr/bin/env python3
"""Development-only native trial: split from actual damaged pixels, never certify derivatives."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,json,sys,importlib.util
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.geometry_consistency import external_surface
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import native_subset
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.evaluation import Evaluator
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.selection import choose
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draft',type=Path)
    parser.add_argument('--max-depth',type=int,default=1)
    args=parser.parse_args();source=args.source_output.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'comparison/initial').mkdir(parents=True)
    ref=json.loads((source/'original_reference.json').read_text());cfg=ref['config']
    candidates=[]
    for path in sorted((source/'rounds/01').glob('group_*/decision.json')):
        decision=json.loads(path.read_text())
        if decision['accepted']:break
        if (not decision['accepted'] and 1<decision['candidates']<=512
                and any('unsupported_local_photo' in reason for reason in decision['reasons'])):
            candidates.append((decision.get('predicted_bg',0),path,decision))
    if not candidates:raise ValueError('No eligible automatically logged rejection')
    _,path,parent=max(candidates,key=lambda item:(item[0],str(item[1])))
    scene=Scene(ref['scene'],cfg);torch.set_num_threads(8)
    model,rows=scene.load_model(scene.spec['model']);ids=np.arange(len(rows))
    assert sha(scene.spec['model'])==ref['input_sha256']
    import lpips
    lp=lpips.LPIPS(net='alex').cuda().eval()
    evaluator=Evaluator(scene,out,None,lp,external_surface(scene,scene.evidence_indices()),original_model=model)
    initial=evaluator.evaluate(model,out/'comparison/initial')
    scene.original_reference_rgb=evaluator.reference_rgb
    local=np.asarray(parent['removed_ids'],np.int64);keep=np.ones(len(rows),bool);keep[local]=False
    rejected=native_subset(scene,model,keep)
    with np.load(source/'rounds/01/evidence.npz') as z:benefits=z['bg'][local].copy()
    draft=args.draft or Path(__file__).resolve().parent/'experiments/GGGS_Cleaner_Comparison/development_drafts/rejection_split_pixel_draft.py'
    spec=importlib.util.spec_from_file_location('ForeSplat_AutoGaussianCleaner.cleaner.adaptive.rejection_split_pixel_draft',draft)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    record=dict(source_parent=str(path),parent_count=len(local),parent_reasons=parent['reasons'],
        input_sha256=ref['input_sha256'],prototype_sha256=sha(draft),automatic_selection='Largest predicted background contribution among first-round logged rejected groups before the first accepted edit, with 2..512 members and a local excluded-photo failure',
        scope='Development diagnostic only; actual damage pixels from the recorded native rejection, no manual ROI/IDs, no relaxed quality rule, no production modification',trials=[])
    reasons=parent['reasons']
    for number in range(args.max_depth):
        partition=out/f'partition_{number}';partition.mkdir()
        with np.load(source/'rounds/01/evidence.npz') as z:benefits=z['bg'][local].copy()
        pieces=module.split_rejected(scene,model,rejected,local,ids,benefits,reasons,initial,initial,cfg,partition)
        if not pieces:break
        group=pieces[0]
        keep=np.ones(len(rows),bool);keep[group]=False;trial=native_subset(scene,model,keep)
        metrics=evaluator.evaluate(trial,out/f'child_{number}')
        accepted,gain,local_gain,reasons=choose(metrics,initial,initial,cfg)
        record['trials'].append(dict(candidates=len(group),original_ids=group,accepted=accepted,gain=gain,local_gain=local_gain,reasons=reasons,metrics=str(out/f'child_{number}/metrics.json')))
        print('damage pixel child',len(group),'accepted',accepted,'gain',gain,'reasons',reasons,flush=True)
        if accepted:break
        local=group;rejected=trial
        write_json(out/'RESULT_IN_PROGRESS.json',record)
    write_json(out/'RESULT.json',record)

if __name__=='__main__':main()
