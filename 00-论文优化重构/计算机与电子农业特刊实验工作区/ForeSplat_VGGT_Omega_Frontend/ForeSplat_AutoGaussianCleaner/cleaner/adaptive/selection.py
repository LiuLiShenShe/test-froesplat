"""Foreground-local silhouette safety and quantitative local/aggregate gains.

The old global contour F1 mixes distant background artifact contours with
the plant contour. It cannot veto a deletion whose plant pixels are identical.
Its raw value stays in the reports. Target-band contour F1 replaces this veto;
alpha, photometry, thin structure and depth/normal budgets remain unchanged.
"""
import numpy as np
from scipy import ndimage
from ..automatic_selection import quality_reasons
from ...evaluation.foreground_metrics import boundary_f1
from .depth_recovery import potential_tradeoff
from .foreground_support import project_metrics, project_record


def contour_metrics(alpha, reference_alpha, mask, radius):
    near = ndimage.distance_transform_edt(~mask) <= max(2, 3*radius)
    # Closing the prediction at the same target-band boundary adds the same
    # artificial clipping contour to both versions; exclude those endpoints.
    p = (alpha > .5) & near
    m = mask
    a = p ^ ndimage.binary_erosion(p)
    b = m ^ ndimage.binary_erosion(m)
    a &= ndimage.binary_erosion(near)
    if not a.any() or not b.any():
        f1 = float(not a.any() and not b.any())
    else:
        precision = (ndimage.distance_transform_edt(~b)[a] <= radius).mean()
        recall = (ndimage.distance_transform_edt(~a)[b] <= radius).mean()
        f1 = float(2*precision*recall/max(precision+recall, 1e-12))
    return dict(fg_local_boundary_f1=f1,
        fg_alpha_loss_pixels=int((mask & (reference_alpha > .5) & (alpha <= .5)).sum()),
        fg_alpha_loss_mass=float(np.maximum(reference_alpha[mask]-alpha[mask], 0).sum()/max(mask.sum(), 1)))


def choose(candidate, current, initial, config):
    complete = candidate.get('metrics_complete', True)
    if complete:
        projected = project_metrics(candidate, config)
        reasons = quality_reasons(projected, project_metrics(initial, config), config)
        reasons += [r for r in quality_reasons(projected, project_metrics(current, config), config) if r not in reasons]
        if config.get('foreground_support_mode') == 'photo_relative_or_depth':
            reasons += [v['name']+':unsupported_local_photo' for v in candidate['views'] if v['unsupported_photo_worse_pixels']]
        reasons = [r for r in reasons if not r.startswith('boundary_f1:')]
        for reference in [initial, current]:
            if candidate['mean']['fg_local_boundary_f1'] < reference['mean']['fg_local_boundary_f1']-config['quality']['boundary_f1_drop']:
                reasons.append('foreground-local boundary F1 regression')
        tradeoff=[]
        recovery=[]
        if (potential_dark_tradeoff(reasons,candidate['views'],initial['views'],current['views'],config) or
                potential_tradeoff(reasons,candidate['views'],initial['views'],current['views'],config)):
            photo_better=candidate['mean']['fg_mse']<=min(current['mean']['fg_mse'],initial['mean']['fg_mse'])+1e-8
            dark_values=[current['mean']['dark_foreground_mse'],initial['mean']['dark_foreground_mse']]
            dark_limit=min(dark_values)*(1+config['quality']['view_foreground_mse_relative_increase'])+config['quality']['foreground_mse_absolute_increase'] if all(v is not None for v in dark_values) else None
            depth_reasons=any('depth' in r for r in reasons)
            external_values=[r['mean'].get('external_depth_mae') for r in [candidate,current,initial]]
            external_better=(not depth_reasons or (all(v is not None for v in external_values) and external_values[0]<=min(external_values[1:])+config['depth_recovery_metric_tolerance']))
            if photo_better and external_better and (dark_limit is None or candidate['mean']['dark_foreground_mse']<=dark_limit):
                by_name={v['name']:v for v in candidate['views']}
                old={v['name']:v for v in initial['views']};previous={v['name']:v for v in current['views']}
                for reason in reasons:
                    name=reason.split(':',1)[0]
                    if 'depth' in reason:
                        recovery.append(dict(view=name,original_rule=reason,
                            evidence=({k:v for k,v in by_name[name].items() if k.startswith(('external_depth_','recovery_'))} if name in by_name else
                                dict(global_external_depth_mae=external_values[0],reference_external_depth_mae=min(external_values[1:]))),
                            reference='Existing confidence-filtered VGGT depth; proxy, not botanical ground truth'))
                    if 'dark_leaf_mse' not in reason:continue
                    tradeoff.append(dict(view=name,original_rule=reason,
                        dark_rmse_increase=float(np.sqrt(by_name[name]['dark_foreground_mse'])-np.sqrt(min(old[name]['dark_foreground_mse'],previous[name]['dark_foreground_mse']))),
                        cap=config['dark_rmse_increase_cap'],
                        foreground_mse_change=candidate['mean']['fg_mse']-current['mean']['fg_mse'],
                        global_dark_mse_limit=dark_limit,
                        condition='All structural/geometry/other photometric constraints pass; aggregate FG MSE does not worsen; every flagged dark-region RMSE increase is at most the fixed input intensity step'))
                reasons=[]
        candidate['dark_mse_tradeoff_analysis']=tradeoff
        candidate['depth_recovery_analysis']=recovery
    else:
        # Only a measured hard per-view failure can shorten a rejected trial.
        # Incomplete metrics never certify a candidate, regardless of benefit.
        reasons = candidate.get('preflight_rejection_reasons', []) or ['incomplete candidate evaluation']
    before = current['mean']['bg_alpha_mass_per_fg']
    gain = (before-candidate['mean']['bg_alpha_mass_per_fg'])/max(before, 1e-12)
    local = max((b['bg_alpha_mass_per_fg']-a['bg_alpha_mass_per_fg'])/max(b['bg_alpha_mass_per_fg'], 1e-12)
                for a, b in zip(candidate['views'], current['views']))
    # Small safe groups can accumulate a measurable full-model improvement.
    # Each needs positive aggregate benefit AND material affected-view benefit.
    free_before=current.get('free_background',{}).get('alpha_mass',0)
    free_after=candidate.get('free_background',{}).get('alpha_mass',free_before)
    free_gain=(free_before-free_after)/max(free_before,1e-12)
    free_local=max([(b['alpha_mass']-a['alpha_mass'])/max(b['alpha_mass'],1e-12)
        for a,b in zip(candidate.get('free_background',{}).get('views',[]),current.get('free_background',{}).get('views',[]))] or [0])
    training_benefit=gain>=config['group_minimum_global_gain'] and local>=config['group_minimum_view_gain']
    free_benefit=free_gain>=config['group_minimum_global_gain'] and free_local>=config['group_minimum_view_gain'] and gain>=-1e-7
    if complete and not (training_benefit or free_benefit):
        reasons.append('insufficient measured group benefit')
    return complete and not reasons, max(gain,free_gain), max(local,free_local), reasons


def potential_dark_tradeoff(reasons,views,initial_views,current_views,config):
    """Only dark-MSE-only failures below a fixed RMSE cap may need full review."""
    if not config.get('structure_constrained_dark_tradeoff',False) or not reasons:
        return False
    by_name={v['name']:v for v in views};old={v['name']:v for v in initial_views};previous={v['name']:v for v in current_views}
    for reason in reasons:
        if ':' not in reason:return False
        name,rules=reason.split(':',1)
        if rules!='dark_leaf_mse' or name not in by_name:return False
        increase=np.sqrt(by_name[name]['dark_foreground_mse'])-np.sqrt(min(old[name]['dark_foreground_mse'],previous[name]['dark_foreground_mse']))
        if increase>config['dark_rmse_increase_cap']:return False
    return True


def view_quality_reasons(candidate, reference, config):
    """Reuse the unchanged hard per-view rules without a partial global test."""
    # Global values agree and are discarded; only the actual view's rules count.
    means = dict(fg_alpha=1., boundary_f1=1., thin_coverage=1., boundary_coverage=1.,
        fg_coverage=1., skeleton_coverage=1., fg_mse=0., depth_relative_p95=0.,
        normal_angle_p95_deg=0., new_depth_invalid_fraction=0., depth_changed_fraction=0.,
        fg_ssim=1., fg_masked_lpips=None)
    reasons = quality_reasons(dict(mean=means, views=[project_record(candidate,config)]),
        dict(mean=means, views=[project_record(reference,config)]), config)
    if config.get('foreground_support_mode') == 'photo_relative_or_depth' and candidate['unsupported_photo_worse_pixels']:
        reasons.append(candidate['name']+':unsupported_local_photo')
    return [r for r in reasons if r.startswith(candidate['name'] + ':')]
