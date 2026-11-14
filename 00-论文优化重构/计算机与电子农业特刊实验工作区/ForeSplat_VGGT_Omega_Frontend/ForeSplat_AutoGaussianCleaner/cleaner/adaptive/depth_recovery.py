"""Certify corrective depth changes against independent existing point-map depth."""
import numpy as np


def metrics(scene,index,a,old,mask,regions):
    _,_,_,depth,confidence=scene.get_view(index)
    before=mask&(old['alpha']>.5)&(old['depth']>0)
    shared=before&(a['alpha']>.5)&(a['depth']>0)
    original_unit=max(float(np.median(old['depth'][before])),1e-9) if before.any() else 1.
    changed=shared&(np.abs(a['depth']-old['depth'])/original_unit>scene.config['quality']['depth_relative_p95'])
    result=dict(external_depth_reference_pixels=0,external_depth_mae=None,external_depth_p95=None,
        geometry_foreground_pixels=int(shared.sum()),recovery_changed_pixels=int(changed.sum()),
        recovery_supported_pixels=0,recovery_toward_pixels=0,recovery_harmful_pixels=0,
        recovery_supported_fraction=0. if changed.any() else 1.,
        recovery_toward_fraction=0. if changed.any() else 1.,recovery_harmful_fraction=0.,
        recovery_edge_harmful_fraction=0.,
        recovery_changed_mae_before=None,recovery_changed_mae_after=None)
    if depth is None:return result
    valid=mask&np.isfinite(depth)&(depth>0)&np.isfinite(confidence)
    if not valid.any():return result
    valid &= confidence>=np.quantile(confidence[valid],.2)
    unit=max(float(np.median(depth[valid])),1e-9)
    supported=shared&valid;supported_changed=changed&valid
    eb=np.abs(old['depth']-depth)/unit;ea=np.abs(a['depth']-depth)/unit
    harmful=supported_changed&(ea>eb+scene.config['quality']['depth_relative_p95'])
    edges=(regions['boundary']|regions['thin'])&shared
    result.update(recovery_supported_pixels=int(supported_changed.sum()),
        recovery_toward_pixels=int((supported_changed&(ea<eb)).sum()),recovery_harmful_pixels=int(harmful.sum()),
        recovery_edge_harmful_fraction=float((harmful&edges).sum()/max(edges.sum(),1)))
    result['external_depth_reference_pixels']=int(supported.sum())
    if supported.any():
        result.update(external_depth_mae=float(ea[supported].mean()),external_depth_p95=float(np.quantile(ea[supported],.95)))
    if changed.any():
        result['recovery_supported_fraction']=float(supported_changed.sum()/changed.sum())
        result['recovery_harmful_fraction']=float(harmful.sum()/max(shared.sum(),1))
    if supported_changed.any():
        result.update(recovery_toward_fraction=float((ea[supported_changed]<eb[supported_changed]).mean()),
            recovery_changed_mae_before=float(eb[supported_changed].mean()),recovery_changed_mae_after=float(ea[supported_changed].mean()))
    return result


def certified(view,initial,current,config):
    if not config.get('certify_depth_recovery',False):return False
    if view.get('external_depth_mae') is None or any(r.get('external_depth_mae') is None for r in [initial,current]):return False
    tolerance=config['depth_recovery_metric_tolerance']
    return (view.get('recovery_changed_pixels',0)>0 and
        view['recovery_supported_fraction']>=config['depth_recovery_minimum_support'] and
        view['recovery_toward_fraction']>=config['depth_recovery_minimum_toward'] and
        view['recovery_harmful_fraction']<=config['depth_recovery_maximum_harmful_fraction'] and
        view['recovery_edge_harmful_fraction']<=config['depth_recovery_maximum_edge_harmful_fraction'] and
        view['external_depth_mae']<=min(initial['external_depth_mae'],current['external_depth_mae'])+tolerance and
        view['external_depth_p95']<=min(initial['external_depth_p95'],current['external_depth_p95'])+tolerance and
        view['recovery_changed_mae_after']<view['recovery_changed_mae_before'])


def certified_global(views,config):
    changed=sum(v.get('recovery_changed_pixels',0) for v in views)
    supported=sum(v.get('recovery_supported_pixels',0) for v in views)
    toward=sum(v.get('recovery_toward_pixels',0) for v in views)
    harmful=sum(v.get('recovery_harmful_pixels',0) for v in views)
    total=sum(v.get('geometry_foreground_pixels',0) for v in views)
    return (changed>0 and supported/changed>=config['depth_recovery_minimum_support'] and
        toward/max(supported,1)>=config['depth_recovery_minimum_toward'] and
        harmful/max(total,1)<=config['depth_recovery_maximum_harmful_fraction'] and
        all(v.get('recovery_edge_harmful_fraction',1)<=config['depth_recovery_maximum_edge_harmful_fraction'] for v in views))


def potential_tradeoff(reasons,views,initial_views,current_views,config):
    if not reasons:return False
    by_name={v['name']:v for v in views};old={v['name']:v for v in initial_views};previous={v['name']:v for v in current_views}
    allowed={'depth_relative_p95','depth_changed_fraction'}
    for reason in reasons:
        if reason in allowed:
            if not config.get('certify_depth_recovery',False):return False
            # Full global recovery still needs a full quality/photo review.
            failed=[v for v in views if v[reason]>config['quality']['view_'+reason]]
            if not certified_global(views,config) or any(not certified(v,old[v['name']],previous[v['name']],config) for v in failed):return False
            continue
        if ':' not in reason:return False
        name,rules=reason.split(':',1)
        if name not in by_name:return False
        for rule in rules.split(','):
            if rule in allowed:
                if not certified(by_name[name],old[name],previous[name],config):return False
            elif rule=='dark_leaf_mse' and config.get('structure_constrained_dark_tradeoff',False):
                increase=np.sqrt(by_name[name]['dark_foreground_mse'])-np.sqrt(min(old[name]['dark_foreground_mse'],previous[name]['dark_foreground_mse']))
                if increase>config['dark_rmse_increase_cap']:return False
            else:return False
    return True
