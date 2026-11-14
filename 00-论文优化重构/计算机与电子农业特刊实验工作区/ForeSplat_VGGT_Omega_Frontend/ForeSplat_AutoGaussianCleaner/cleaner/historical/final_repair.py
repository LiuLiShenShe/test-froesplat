"""One bounded restoration after final all-camera quality detects missed harm.

The full-camera failure selects at most24 diagnostic views. Native compositor
attribution identifies edited original rows contributing to lost thin coverage,
worsened photographs or depth-supported loss. Restore those rows exactly, then
use24/64 checks before one final all-camera evaluation. No per-row full loop.
"""
import numpy as np
import torch
from ..contribution_analysis import contribution
from ..foreground_protection import regions
from ..io import write_json
from ...evaluation.metrics import arrays


def edited_members(input_rows,input_ids,rows,ids):
    positions=np.searchsorted(input_ids,ids)
    if not np.array_equal(input_ids[positions],ids):
        raise ValueError('Candidate lineage is not a stage-input subset')
    edited=np.ones(len(input_rows),bool)
    changed=np.zeros(len(rows),bool)
    for key in rows.dtype.names:
        changed|=input_rows[key][positions]!=rows[key]
    edited[positions]=changed
    return np.flatnonzero(edited)


def restore_members(input_rows,input_ids,rows,ids,members):
    members=np.unique(np.asarray(members,dtype=np.int64))
    new_ids=np.union1d(ids,input_ids[members])
    result=input_rows[np.searchsorted(input_ids,new_ids)].copy()
    result[np.searchsorted(new_ids,ids)]=rows
    result[np.searchsorted(new_ids,input_ids[members])]=input_rows[members]
    return result,new_ids


def repair_view_indices(initial,selected,limit=24):
    old={v['index']:v for v in initial['views']}
    ranked=[]
    for value in selected['views']:
        before=old[value['index']]
        severity=0.
        for key,tolerance in [('thin_coverage',.003),('skeleton_coverage',.002),
                ('supported_boundary_coverage',.006),('supported_fg_alpha',.003),
                ('supported_dark_foreground_alpha',.003)]:
            if before.get(key) is not None and value.get(key) is not None:
                severity+=max(0.,before[key]-value[key])/tolerance
        severity+=max(0.,before['fg_psnr']-value['fg_psnr'])/.12
        severity+=value.get('edge_photo_worse_alpha_loss',0.)/.002
        severity+=value.get('depth_supported_alpha_loss',0.)/.002
        ranked.append((severity,value['index']))
    return sorted(i for _,i in sorted(ranked,key=lambda v:(-v[0],v[1]))[:limit])


def restoration_evidence(scene,baseline,candidate,indices,evaluator,members,output):
    weights=np.zeros(len(baseline.get_xyz),np.float64);records=[]
    for index in indices:
        c,target,mask,_,_=scene.get_view(index)
        with torch.no_grad():after=arrays(scene.render(c,candidate))
        before=evaluator.reference[index];rgb=evaluator.reference_rgb[index]
        reg=regions(mask,scene.config['boundary_relative_width'])
        _,geometry=scene.original_foreground_support[index]
        lost=(before['alpha']>.5)&(after['alpha']<=.5)
        sensitive=reg['thin']|reg['skeleton']|reg['boundary']
        old_error=((rgb-target)**2).mean(2);new_error=((after['rgb']-target)**2).mean(2)
        worsening=sensitive&(np.sqrt(new_error)-np.sqrt(old_error)>1/255)
        roi=np.zeros(mask.shape,float)
        # Region-normalized weights correspond to coverage, including raw thin
        # pixels even when external depth/photometric support is uncertain.
        for key in ['thin','skeleton']:
            roi+=(lost&reg[key])/max(reg[key].sum(),1)
        roi+=np.maximum(before['alpha']-after['alpha'],0)*geometry/max(geometry.sum(),1)
        roi+=np.where(worsening,np.maximum(new_error-old_error,0),0)/max(sensitive.sum(),1)
        g,_,err=contribution(c,baseline,np.stack([roi,roi,roi]))
        weights+=g[:,0]/len(indices)
        records.append(dict(index=index,lost_thin_pixels=int((lost&reg['thin']).sum()),
            lost_skeleton_pixels=int((lost&reg['skeleton']).sum()),worsening_edge_pixels=int(worsening.sum()),
            conservation_error=float(err.max())))
        print(f'final restoration attribution {len(records)}/{len(indices)}',flush=True)
    eligible=np.asarray(members,dtype=np.int64)
    positive=weights[eligible]>max(float(weights[eligible].sum())*1e-9,1e-12)
    restore=eligible[positive]
    write_json(output/'evidence.json',dict(indices=indices,views=records,edited_members=len(members),
        restore_members=len(restore),policy='Restore all edited stage-input rows with positive masked native coverage/photo harm; no relaxed final threshold'))
    np.save(output/'restored_stage_input_rows.npy',restore)
    return restore
