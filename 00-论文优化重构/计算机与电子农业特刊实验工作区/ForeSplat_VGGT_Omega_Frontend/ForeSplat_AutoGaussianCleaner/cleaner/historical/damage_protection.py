"""One masked reattribution protects harmful foreground rows in a broad prune.

This is a whole-round correction at representative views. No fixed pixel ROI,
Gaussian ID, per-point full-camera validation or C5 recursive splitter is used.
"""
import numpy as np
import torch
from ..contribution_analysis import contribution
from ..foreground_protection import regions
from ...evaluation.metrics import arrays


def damage_evidence(scene, model, candidate, indices, reference_rgb, references):
    n=len(model.get_xyz)
    evidence=np.zeros((n,3),np.float64)
    records=[]
    for vi in indices:
        c,target,mask,_,_=scene.get_view(vi)
        with torch.no_grad():
            after=arrays(scene.render(c,candidate))
        initial=references[vi]
        rgb=reference_rgb[vi]
        reg=regions(mask,scene.config['boundary_relative_width'])
        support,geometry=scene.original_foreground_support[vi]
        old_error=((rgb-target)**2).mean(2)
        error=((after['rgb']-target)**2).mean(2)
        increase=np.maximum(error-old_error,0)
        dark=mask & (target@[.2126,.7152,.0722]<.12)
        sensitive=dark|reg['boundary']|reg['thin']|geometry
        # Weights are actual worsening photometry and supported lost coverage,
        # measured against the immutable stage input rather than the last edit.
        bad=sensitive & (np.sqrt(error)-np.sqrt(old_error)>1/255)
        damage=np.where(bad,increase,0)
        coverage=np.maximum(initial['alpha']-after['alpha'],0)*geometry
        improving=np.maximum(old_error-error,0)*mask
        g,_,err=contribution(c,model,np.stack([damage,coverage,improving]))
        evidence+=g/len(indices)
        records.append(dict(index=vi,damaged_pixels=int(bad.sum()),damage_weight=float(damage.sum()),conservation_error=float(err.max())))
        print(f'foreground damage attribution {len(records)}/{len(indices)}',flush=True)
    return evidence,records


def protected_members(evidence,members,coverage=.98):
    members=np.asarray(members,np.int64)
    # Any depth-supported alpha loss matters; photometric protection ranks
    # by the share of measured harm. Improvement cannot cancel a dark leaf.
    weights=evidence[members,0]+evidence[members,1]
    positive=weights>max(float(weights.sum())*1e-9,1e-12)
    ranked=np.flatnonzero(positive)
    if not len(ranked):
        return np.zeros(len(members),bool)
    ranked=ranked[np.argsort(-weights[ranked])]
    cumulative=np.cumsum(weights[ranked])
    end=min(len(ranked),int(np.searchsorted(cumulative,coverage*cumulative[-1]))+1)
    protect=np.zeros(len(members),bool)
    protect[ranked[:end]]=True
    return protect
