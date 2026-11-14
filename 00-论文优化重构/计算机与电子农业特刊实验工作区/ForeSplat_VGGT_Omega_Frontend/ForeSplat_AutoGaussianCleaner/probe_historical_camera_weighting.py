#!/usr/bin/env python3
"""Use immutable historical per-view attribution to audit camera aggregation bias."""
from pathlib import Path
import argparse,json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();root=a.experiment.resolve()
    scene=json.loads((root/'execution.json').read_text())['reference']['scene']
    hist=Path(scene['baseline_models']['R5']).parent.parent
    old=hist.parent.parent.parent/'geometry_mesh_final_selection/surface_inputs/e10_gggs_surface/camera_runtime.json'
    # The native source lives in the same plant experiment tree; this is an
    # evaluation-only comparison against historical caches, not pipeline input.
    if not old.exists():
        old=hist.parents[2]/'geometry_mesh_final_selection/surface_inputs/e10_gggs_surface/camera_runtime.json'
    records=json.loads(old.read_text())['records']
    by_name={r['name']:i for i,r in enumerate(records)}
    E=np.array([np.asarray(r['world_view_transform']).T for r in records])
    centers=np.linalg.inv(E)[:,:3,3];extent=max(np.linalg.norm(np.ptp(centers,axis=0)),1e-9)
    features=np.column_stack([centers/extent,E[:,2,:3]])
    stage=root/'stage1_generalized'
    mapping={v['index']:v['name'] for v in json.loads((stage/'initial_confirmation/metrics.json').read_text())['views']}
    selections={24:json.loads((stage/'rounds/01/evidence/identity.json').read_text())['indices'],
                64:json.loads((stage/'original_foreground/identity.json').read_text())['indices']}
    with np.load(hist/'diagnosis/actual_contribution_evidence.npz') as z:full=z['fg_contribution']/len(records)
    estimates={};weight_records={}
    for count,indices in selections.items():
        chosen=np.array([by_name[mapping[i]] for i in indices])
        distances=((features[:,None,:]-features[chosen][None,:,:])**2).sum(2)
        assignment=distances.argmin(1);weights=np.bincount(assignment,minlength=len(chosen))/len(records)
        simple=np.zeros_like(full);weighted=np.zeros_like(full)
        for j,index in enumerate(chosen):
            f=hist/'diagnosis/contribution_views'/f'{records[index]["source_id"]:04d}.npz'
            with np.load(f) as z:g=z['contribution'][:,0].astype(float)
            simple+=g/len(chosen);weighted+=g*weights[j]
        estimates[f'equal_{count}']=simple;estimates[f'voronoi_{count}']=weighted
        weight_records[str(count)]=[dict(name=records[i]['name'],weight=float(w)) for i,w in zip(chosen,weights)]
        print('cached camera aggregation',count,flush=True)
    progress=json.loads((stage/'progress.json').read_text())
    ids=np.load(stage/'rounds'/f'{progress["round"]:02d}'/'accepted_original_ids.npy')
    history_ids=np.load(hist/'additional_rounds/round_05/original_gaussian_ids.npy')
    missed=np.setdiff1d(ids,history_ids);missed=missed[full[missed]>.001]
    qs=[.01,.25,.5,.75,.99]
    summary={label:np.quantile(value[missed]/full[missed],qs).tolist() for label,value in estimates.items()}
    current=np.load(stage/'original_foreground/foreground.npy')
    valid=estimates['equal_64']>.001
    validation=np.quantile(current[valid]/estimates['equal_64'][valid],[.01,.5,.99]).tolist()
    result=dict(sample='CurrentStage1-kept but historicalRound5-with-marker-removed rows; historical originalFG mean>.001',
        count=len(missed),quantiles=qs,ratios_to_historical_full208_foreground=summary,
        cached_equal64_vs_live_equal64_validation=validation,weights=weight_records,
        caveat='This audits original-model representative-view aggregation only. Any weighting change must reattribute every changed model and rerun quality/visual gates.')
    dest=root/'comparison/camera_weighting';dest.mkdir(parents=True,exist_ok=True)
    (dest/'CAMERA_WEIGHTING_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(dest/'original_foreground_estimates.npz',historical_full208=full,**estimates)
    print(json.dumps({k:v for k,v in result.items() if k!='weights'},indent=2))


if __name__=='__main__':main()
