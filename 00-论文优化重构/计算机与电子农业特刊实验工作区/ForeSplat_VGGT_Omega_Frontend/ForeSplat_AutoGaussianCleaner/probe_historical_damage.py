#!/usr/bin/env python3
"""Test whole-round harm protection using measured native RGB and depth support."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.historical.runner import HistoricalEvaluator,quality_reasons,score,load_rows
from ForeSplat_AutoGaussianCleaner.cleaner.historical.evidence import camera_indices
from ForeSplat_AutoGaussianCleaner.cleaner.historical.damage_protection import damage_evidence,protected_members
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply,write_json,save_ply


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--round',type=int,default=5)
    a=p.parse_args();root=a.experiment.resolve();stage=root/'stage1_generalized'
    execution=json.loads((root/'execution.json').read_text())['reference']
    config=execution['config'];scene=Scene(execution['scene'],config);torch.set_num_threads(8)
    dest=root/'comparison'/f'damage_protection_round_{a.round:02d}';dest.mkdir(parents=True,exist_ok=True)
    original,_=scene.load_model(scene.spec['model'])
    ev=HistoricalEvaluator(scene,dest,original_model=original)
    indices=camera_indices(scene,24)
    initial=ev.evaluate(original,dest/'initial',indices)
    folder=stage/'rounds'/f'{a.round:02d}'
    prior=stage/'rounds'/f'{a.round-1:02d}'
    rows=read_ply(prior/'accepted.ply');ids=np.load(prior/'accepted_original_ids.npy')
    rejected=read_ply(folder/'prune_1/candidate.ply')
    # Native rows remain exact here, so model row IDs can be recovered via XYZ
    # only after verifying structured row equality for every recovered member.
    dtype=np.dtype([(k,rows.dtype[k]) for k in ['x','y','z']])
    xyz=np.empty(len(rows),dtype=dtype);other=np.empty(len(rejected),dtype=dtype)
    for k in xyz.dtype.names:xyz[k]=rows[k];other[k]=rejected[k]
    order=np.argsort(xyz,order=['x','y','z']);loc=np.searchsorted(xyz[order],other)
    kept=order[loc]
    if not np.array_equal(rows[kept],rejected):raise ValueError('Candidate lineage mismatch')
    remove=np.ones(len(rows),bool);remove[kept]=False;members=np.flatnonzero(remove)
    current,_=scene.load_model(prior/'accepted.ply')
    candidate,_=scene.load_model(folder/'prune_1/candidate.ply')
    evidence,records=damage_evidence(scene,current,candidate,indices,ev.reference_rgb,ev.reference)
    np.savez(dest/'damage.npz',evidence=evidence,original_ids=ids,delete_members=members)
    results=[]
    for coverage in [.9,.98,1.]:
        protect=protected_members(evidence,members,coverage)
        delete=members[~protect]
        keep=np.ones(len(rows),bool);keep[delete]=False
        trial=dest/f'protect_{coverage:g}'
        model,_=load_rows(scene,rows[keep],trial)
        metrics=ev.evaluate(model,trial/'quick',indices)
        reasons=quality_reasons(initial,metrics,config)
        result=dict(coverage=coverage,protected=int(protect.sum()),deleted=len(delete),gaussians=int(keep.sum()),
            score=score(metrics,initial),quality_reasons=reasons,metrics=metrics)
        write_json(trial/'result.json',result);results.append(result)
        np.save(trial/'original_gaussian_ids.npy',ids[keep])
        print({k:v for k,v in result.items() if k!='metrics'},flush=True)
    write_json(dest/'probe.json',dict(attribution=records,proposals=results))


if __name__=='__main__':main()
