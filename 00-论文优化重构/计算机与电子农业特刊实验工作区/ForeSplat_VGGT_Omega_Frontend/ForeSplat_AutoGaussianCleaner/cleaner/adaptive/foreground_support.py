"""Fixed original-model support; raw photographs remain the primary loss guard."""
import numpy as np
from scipy import ndimage
from ..foreground_protection import regions
from ..contribution_analysis import contribution


def support_masks(before, target, mask, depth, confidence, config):
    noise = config.get('foreground_support_noise', 1/255)
    error = ((before['rgb']-target)**2).mean(2)
    energy = (target**2).mean(2)+noise**2
    appearance = mask & (before['alpha']>.5) & (error/np.maximum(energy,1e-12)<=config['foreground_support_relative_error'])
    geometry = np.zeros_like(mask)
    if depth is not None and confidence is not None:
        valid=mask & np.isfinite(depth) & (depth>0) & np.isfinite(confidence)
        if valid.any():
            valid &= confidence>=np.quantile(confidence[valid],.2)
            unit=max(float(np.median(depth[valid])),1e-9)
            geometry=valid & (before['alpha']>.5) & (before['depth']>0) & (np.abs(before['depth']-depth)/unit<=.03)
    supported=appearance|geometry
    if supported.sum()<config.get('foreground_support_minimum_fraction',.5)*mask.sum():
        supported=mask.copy()
    return supported, geometry


def support_metrics(after, before, reference_rgb, target, mask, r, supported, geometry, config):
    def mean(value,selection):return float(value[selection].mean()) if selection.any() else 1.
    def coverage(selection):return mean((after['alpha']>.5).astype(np.float32),selection)
    dark=mask & ((target@np.array([.2126,.7152,.0722]))<.12)
    selections={}
    for name,selection in [('boundary',r['boundary']),('thin',r['thin']),('skeleton',r['skeleton']),('dark_foreground',dark)]:
        subset=selection&supported
        selections[name]=subset if subset.sum()>=config.get('foreground_support_minimum_fraction',.5)*selection.sum() else selection
    output=dict(supported_fg_alpha=mean(after['alpha'],supported),supported_fg_coverage=coverage(supported),
        supported_boundary_coverage=coverage(selections['boundary']),supported_thin_coverage=coverage(selections['thin']),
        supported_skeleton_coverage=coverage(selections['skeleton']),supported_dark_foreground_alpha=mean(after['alpha'],selections['dark_foreground']),
        original_supported_pixels=int(supported.sum()),original_unsupported_pixels=int((mask&~supported).sum()))
    for name,selection in [('boundary',r['boundary']),('thin',r['thin']),('skeleton',r['skeleton']),('dark_foreground',dark)]:
        output['raw_'+name+'_support_pixels']=int(selection.sum())
        output['photo_depth_'+name+'_support_pixels']=int((selection&supported).sum())
        output['guard_'+name+'_support_pixels']=int(selections[name].sum())
        output[name+'_support_fallback']=bool(selections[name].sum()>(selection&supported).sum())
    if geometry.sum()<config.get('foreground_geometry_minimum_fraction',.1)*mask.sum():
        geometry=supported&(before['alpha']>.5)&(before['depth']>0)
        output['geometry_support_fallback']=True
    else:output['geometry_support_fallback']=False
    output['guard_geometry_support_pixels']=int(geometry.sum())
    valid=geometry&(after['alpha']>.5)&(after['depth']>0)
    unit=max(float(np.median(before['depth'][geometry])),1e-9) if geometry.any() else 1.
    delta=np.abs(before['depth'][valid]-after['depth'][valid])/unit
    no,nn=np.linalg.norm(before['normal'],axis=2),np.linalg.norm(after['normal'],axis=2)
    normals=valid&(no>.1)&(nn>.1)
    cosine=(before['normal'][normals]*after['normal'][normals]).sum(1)/np.maximum(no[normals]*nn[normals],1e-9)
    angles=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    output.update(supported_depth_relative_p95=float(np.quantile(delta,.95)) if len(delta) else 0.,
        supported_depth_changed_fraction=float((delta>config['quality']['depth_relative_p95']).mean()) if len(delta) else 0.,
        supported_new_depth_invalid_fraction=float((geometry&~valid).sum()/max(geometry.sum(),1)),
        supported_normal_angle_p95_deg=float(np.quantile(angles,.95)) if len(angles) else 0.)
    # Guard EVERY excluded-support pixel, including small dark leaves. No
    # cancellation across a large ROI or dilution by a full-image average.
    initial_error=np.sqrt(((reference_rgb-target)**2).mean(2))
    changed_error=np.sqrt(((after['rgb']-target)**2).mean(2))
    increase=changed_error-initial_error
    excluded=mask&~supported
    worse=excluded&(increase>config.get('unsupported_local_rmse_cap',1/255))
    _,components=ndimage.label(worse)
    output.update(unsupported_photo_worse_pixels=int(worse.sum()),unsupported_photo_worse_components=int(components),
        unsupported_photo_max_rmse_increase=float(increase[excluded].max()) if excluded.any() else 0.,
        unsupported_new_alpha_holes=int((excluded&(before['alpha']>.5)&(after['alpha']<=.5)).sum()),
        supported_new_alpha_holes=int((supported&(before['alpha']>.5)&(after['alpha']<=.5)).sum()))
    return output


STRUCTURAL_FIELDS=['fg_alpha','fg_coverage','boundary_coverage','thin_coverage','skeleton_coverage',
    'dark_foreground_alpha','depth_relative_p95','depth_changed_fraction','new_depth_invalid_fraction','normal_angle_p95_deg']


def project_record(record,config):
    if config.get('foreground_support_mode')!='photo_relative_or_depth':return record
    result=record.copy()
    for key in STRUCTURAL_FIELDS:
        if 'supported_'+key not in record:raise ValueError('Missing fixed-original supported quality measurement: '+key)
        result[key]=record['supported_'+key]
    return result


def project_metrics(metrics,config):
    if config.get('foreground_support_mode')!='photo_relative_or_depth':return metrics
    return {**metrics,'mean':project_record(metrics['mean'],config),
        'views':[project_record(v,config) for v in metrics['views']]}


def add_supported_contribution(scene,model,evidence,indices,config):
    n=len(model.get_xyz);sums=np.zeros((n,3),np.float64)
    for count,index in enumerate(indices):
        if index not in scene.original_foreground_support:raise ValueError('Support must be fixed on the original GGGS model first')
        c,_,mask,*_=scene.get_view(index);r=regions(mask,config['boundary_relative_width'])
        support=scene.original_foreground_support[index][0]
        value,_,_=contribution(c,model,np.stack([support,r['boundary']&support,r['thin']&support]))
        sums+=value/max(mask.sum(),1)
        if (count+1)%16==0 or count+1==len(indices):print('supported contribution',count+1,'/',len(indices),flush=True)
    for dimension,key in enumerate(['fg','boundary','thin']):
        evidence['raw_'+key]=evidence[key].copy();evidence[key]=sums[:,dimension]/max(len(indices),1)
    return evidence
