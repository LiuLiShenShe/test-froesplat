"""Two-stage native GGGS runner with 24/64/full evaluation and lossless rollback."""
from pathlib import Path
import csv
import gc
import json
import shutil
import time
import numpy as np
import torch
from scipy.special import expit, logit
from ..runtime import Scene, camera
from ..io import read_ply, save_ply, bake_viewer, write_json, sha
from ..geometry_consistency import external_surface
from ..adaptive.evaluation import Evaluator, summarize
from ..adaptive.legacy import features, rules
from ..adaptive.artifacts import export, parity, visuals
from ...evaluation.metrics import arrays
from .evidence import camera_indices, measure, component_features, historical_candidates, adaptive_floor, foreground_envelope
from .damage_protection import damage_evidence, protected_members


class HistoricalEvaluator(Evaluator):
    def __init__(self,scene,*args,**kwargs):
        # Retiring an evaluator must release its reference arrays. Avoid a
        # chain of old evaluator closures on Scene.get_view across stages.
        if hasattr(scene,'_historical_get_view'):
            scene.get_view=scene._historical_get_view
        else:
            scene._historical_get_view=scene.get_view
        super().__init__(scene,*args,**kwargs)

    def close(self):
        self.scene.get_view=self.scene._historical_get_view

    def geometry_cached(self,index,a,old,mask):
        result=super().geometry_cached(index,a,old,mask)
        if not self.scene.config.get('historical_supported_quality',False):
            return result
        support,geometry=self.scene.original_foreground_support[index]
        if geometry.sum()<.1*mask.sum():
            geometry=support
        before=geometry&(old['alpha']>.5)&(old['depth']>0)
        after=geometry&(a['alpha']>.5)&(a['depth']>0)
        valid=before&after
        unit=max(float(np.median(old['depth'][before])),1e-9) if before.any() else 1.
        delta=np.abs(a['depth'][valid]-old['depth'][valid])/unit
        nv=valid&(np.linalg.norm(old['normal'],axis=2)>.1)&(np.linalg.norm(a['normal'],axis=2)>.1)
        cosine=(old['normal'][nv]*a['normal'][nv]).sum(1)/np.maximum(np.linalg.norm(old['normal'][nv],axis=1)*np.linalg.norm(a['normal'][nv],axis=1),1e-9)
        angles=np.degrees(np.arccos(np.clip(cosine,-1,1)))
        result.update(supported_depth_relative_p95=float(np.quantile(delta,.95)) if len(delta) else 0.,
            supported_depth_changed_fraction=float((delta>self.scene.config['quality']['depth_relative_p95']).mean()) if len(delta) else 0.,
            supported_new_depth_invalid_fraction=float((before&~after).sum()/max(before.sum(),1)),
            supported_normal_angle_p95_deg=float(np.quantile(angles,.95)) if len(angles) else 0.)
        return result

    def photometry_cached(self, index, rgb, target, mask, r, alpha):
        photo, lp = super().photometry_cached(index, rgb, target, mask, r, alpha)
        for name in ['boundary', 'thin']:
            roi = r[name]
            photo[name+'_rgb_mse'] = float(((rgb[roi]-target[roi])**2).mean()) if roi.any() else 0.
        if self.scene.config.get('historical_supported_quality',False):
            from ..adaptive.foreground_support import support_masks
            if index not in self.reference_rgb:
                self.reference_rgb[index]=rgb.copy()
            before=self.reference[index]
            if index not in self.scene.original_foreground_support:
                _,_,_,depth,confidence=self.scene.get_view(index)
                support,geometry=support_masks(dict(before,rgb=self.reference_rgb[index]),target,mask,depth,confidence,self.scene.config)
                self.scene.original_foreground_support[index]=(support,geometry)
            support,geometry=self.scene.original_foreground_support[index]
            for key,region in [('fg_alpha',mask),('fg_coverage',mask),('boundary_coverage',r['boundary']),
                               ('thin_coverage',r['thin']),('skeleton_coverage',r['skeleton']),
                               ('dark_foreground_alpha',mask & (target@[.2126,.7152,.0722]<.12))]:
                qualified=region&support
                # A silhouette boundary straddles the RAP mask. Its outside
                # half cannot have foreground support; falling back to the
                # whole band when support is below50% protects black exterior
                # tails. Only evidence-supported edge pixels define this
                # coverage; all unsupported edges still receive the immutable
                # RGB-worsening alpha-loss guard below, and raw thin/skeleton
                # coverage remains mandatory.
                if key!='boundary_coverage' and qualified.sum()<self.scene.config.get('foreground_support_minimum_fraction',.5)*region.sum():
                    qualified=region
                values=alpha if key.endswith('alpha') else alpha>.5
                photo['supported_'+key]=float(values[qualified].mean()) if qualified.any() else (1. if key=='boundary_coverage' else None)
                if key=='boundary_coverage':photo['qualified_boundary_pixels']=int(qualified.sum())
            dark=mask&support&(target@[.2126,.7152,.0722]<.12)
            photo['supported_dark_foreground_mse']=float(((rgb[dark]-target[dark])**2).mean()) if dark.any() else None
            old_err=np.sqrt(((self.reference_rgb[index]-target)**2).mean(2))
            new_err=np.sqrt(((rgb-target)**2).mean(2))
            # A discarded outer compositor tail is not a lost real edge if it
            # improves the observed photograph. Geometry-supported alpha loss
            # remains measured separately and protected.
            edge=r['boundary']|r['thin']
            worsened=edge&((new_err-old_err)>1/255)
            lost=np.maximum(before['alpha']-alpha,0)
            photo['edge_photo_worse_alpha_loss']=float(lost[worsened].sum()/max(edge.sum(),1))
            photo['depth_supported_alpha_loss']=float(lost[geometry].sum()/max(geometry.sum(),1))
            if self.scene.config.get('allow_unsupported_nonworsening_tails',False):
                # An opaque black tail can cross a RAP thin region despite
                # having neither correct observed color nor valid depth. Count
                # its raw alpha loss separately; preserve all supported pixels
                # and every pixel whose observed RGB gets worse.
                holes=(before['alpha']>.5)&(alpha<=.5)
                unchanged_or_better=(new_err<=old_err+1e-10)
                unresolved=holes&~support&~geometry&unchanged_or_better
                for name in ['thin','skeleton']:
                    region=r[name]
                    photo['unsupported_nonworsening_'+name+'_loss']=float((unresolved&region).sum()/max(region.sum(),1))
        return photo, lp

    def free_background(self, model):
        from scipy import ndimage
        records = []
        for uid, s in enumerate(self.free_views):
            v = s['view']
            c = camera(v['K'], v['E'], v['width'], v['height'], v['name'], uid)
            with torch.no_grad():
                a = arrays(self.scene.render(c, model))
            outside = s['outside_surface']
            occupied = outside & (a['alpha'] > .5)
            labels, _ = ndimage.label(occupied)
            sizes = np.bincount(labels.ravel())[1:]
            gray = a['rgb']+(1-a['alpha'][..., None])*.5
            records.append(dict(name=v['name'], alpha_mass=float(a['alpha'][outside].sum()),
                high_alpha_pixels=int(occupied.sum()),
                black_pixels=int((occupied & (gray @ [.2126, .7152, .0722] < .12)).sum()),
                largest_component=int(sizes.max()) if len(sizes) else 0,
                aabb_alpha_mass=float(a['alpha'][s['outside_aabb']].sum())))
        return dict(views=records, **{k: sum(r[k] for r in records) for k in
            ['alpha_mass', 'high_alpha_pixels', 'black_pixels', 'largest_component', 'aabb_alpha_mass']},
            protocol='automatic_depth_trajectory_landscape_v2' if self.scene.config.get('free_view_aspect') else 'automatic_depth_trajectory_v1',
            viewport_aspect=self.scene.config.get('free_view_aspect'),
            reference='Masked independent depth projection; auxiliary silhouette proxy, not mesh or ground truth')


def subset_metrics(result, indices):
    wanted = set(indices)
    out = summarize([r for r in result['views'] if r['index'] in wanted], result['resolution_max_side'])
    out['free_background'] = result['free_background']
    return out


def quality_reasons(before, after, config):
    """Stage-relative and input-relative safeguards, matched camera populations."""
    q = config['quality']
    def guarded(record):
        if not config.get('historical_supported_quality',False):
            return record
        fields=['fg_alpha','fg_coverage','boundary_coverage','thin_coverage','skeleton_coverage','dark_foreground_alpha','dark_foreground_mse',
                'depth_relative_p95','depth_changed_fraction','new_depth_invalid_fraction','normal_angle_p95_deg']
        return {**record,**{k:record['supported_'+k] for k in fields if record.get('supported_'+k) is not None}}
    old, new = guarded(before['mean']), guarded(after['mean'])
    reasons = []
    floors = ['fg_alpha', 'fg_coverage', 'boundary_coverage', 'thin_coverage', 'skeleton_coverage', 'dark_foreground_alpha']
    limits = ['foreground_alpha_drop', 'foreground_coverage_drop', 'boundary_coverage_drop', 'thin_coverage_drop', 'skeleton_coverage_drop', 'dark_foreground_alpha_drop']
    if new['fg_psnr'] < old['fg_psnr']-q['foreground_psnr_drop_db']:
        reasons.append('foreground_psnr')
    for name, limit in zip(floors, limits):
        if old.get(name) is not None and new.get(name) is not None and new[name] < old[name]-q[limit]:
            reasons.append(name)
    if old.get('dark_foreground_mse') is not None and new.get('dark_foreground_mse') is not None:
        if new['dark_foreground_mse'] > old['dark_foreground_mse']*(1+q['dark_foreground_mse_relative_increase'])+1e-5:
            historical_tradeoff=(config.get('active_stage')==1 and config.get('stage1_historical_photo_tradeoff',False)
                and not reasons)
            # Stage1 follows the historical photo/background tradeoff while
            # preserving measured structure. Stage2 retains the strict guard.
            if not historical_tradeoff:reasons.append('dark_foreground_rgb')
    for k in ['boundary_rgb_mse', 'thin_rgb_mse']:
        if new[k] > old[k]+q['local_rgb_mse_increase']:
            reasons.append(k)
    for k in ['depth_changed_fraction', 'normal_angle_p95_deg', 'new_depth_invalid_fraction']:
        if new[k] > q[k]:
            reasons.append(k)
    if config.get('historical_supported_quality',False):
        if after['mean']['edge_photo_worse_alpha_loss']>config.get('edge_photo_worse_alpha_loss',.002):
            reasons.append('edge_photo_worse_alpha_loss')
        if after['mean']['depth_supported_alpha_loss']>config.get('depth_supported_alpha_loss',.002):
            reasons.append('depth_supported_alpha_loss')
        # Raw thin/skeleton coverage stays a second guard even where optical
        # support is incomplete; no density/color rule can erase a dark leaf.
        for key,limit in [('thin_coverage',.003),('skeleton_coverage',.002)]:
            allowance=0.
            if config.get('allow_unsupported_nonworsening_tails',False):
                allowance=after['mean'].get('unsupported_nonworsening_'+key.removesuffix('_coverage')+'_loss',0.)
            if after['mean'][key]+allowance<before['mean'][key]-limit:
                reasons.append('raw_'+key)
    # Averages alone can hide one damaged leaf or pot rim. Use the 95th
    # percentile of matched-view drops, without a per-candidate full208 loop.
    previous = {r['index']: guarded(r) for r in before['views']}
    for key, limit in [('thin_coverage', .01), ('skeleton_coverage', .005), ('fg_alpha', .008)]:
        drops = [previous[r['index']][key]-guarded(r)[key] for r in after['views']]
        if np.quantile(drops, .95) > limit:
            reasons.append('view_p95_'+key)
    return reasons


def score(metrics, reference):
    def gain(old, new):
        return float(np.clip((old-new)/max(old, 1e-8), -1, 1)) if old > 1e-8 else (0. if new < 1e-8 else -1.)
    a, b = reference['free_background'], metrics['free_background']
    return (.35*gain(a['black_pixels'], b['black_pixels'])+
            .25*gain(a['high_alpha_pixels'], b['high_alpha_pixels'])+
            .15*gain(a['alpha_mass'], b['alpha_mass'])+
            .10*gain(a['largest_component'], b['largest_component'])+
            .15*gain(reference['mean']['bg_alpha_mass_per_fg'], metrics['mean']['bg_alpha_mass_per_fg']))


def covariance_proposal(rows, selected, spatial, surface, factor):
    """Shrink a candidate axis towards local surface; preserve effective opacity.

    Native quaternion eigenvectors determine the spatial axis. XYZ, SH, rotation
    and filter remain exact. Infeasible opacity compensation leaves a row exact.
    """
    from scipy.spatial.transform import Rotation
    out = rows.copy()
    if not len(selected):
        return out, np.array([], dtype=np.int64)
    selected = np.asarray(selected, np.int64)
    sig = np.exp(np.column_stack([rows[f'scale_{k}'][selected] for k in range(3)]).astype(float))
    quat = np.column_stack([rows[f'rot_{k}'][selected] for k in range(4)])
    # GGGS stores w,x,y,z; scipy expects x,y,z,w.
    rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
    axis = sig.argmax(1)
    if surface is not None and len(surface['points']) >= 8:
        xyz = np.column_stack([rows[k][selected] for k in ['x', 'y', 'z']])
        _, ni = surface['tree'].query(xyz, k=8, workers=8)
        p = surface['points'][ni]
        p -= p.mean(1, keepdims=True)
        _, vec = np.linalg.eigh(np.einsum('nki,nkj->nij', p, p)/8)
        normal = vec[:, :, 0]
        projected = np.abs(np.einsum('nji,nj->ni', rotation, normal))*sig
        axis = projected.argmax(1)
    new_sig = sig.copy()
    new_sig[np.arange(len(sig)), axis] *= factor
    fil = rows['filter_3D'][selected, None]
    old_coef = (sig/np.hypot(sig, fil)).prod(1)
    new_coef = (new_sig/np.hypot(new_sig, fil)).prod(1)
    desired = expit(rows['opacity'][selected].astype(float))*old_coef
    raw = desired/np.maximum(new_coef, 1e-30)
    valid = (raw > 0) & (raw < 1-1e-7)
    ids = selected[valid]
    for k in range(3):
        out[f'scale_{k}'][ids] = np.log(new_sig[valid, k])
    out['opacity'][ids] = logit(raw[valid])
    return out, ids


def filtered_covariance_proposal(rows, selected, surface, factor, mode='surface_normal'):
    """Bake only selected filters, then constrain their actual spatial axes.

    Baking the selected row's filter is algebraically equivalent before the
    axis edit. This avoids impossible opacity compensation for saturated raw
    opacity. Effective center opacity, XYZ, SH and rotation remain preserved;
    all unselected native rows remain bitwise exact.
    """
    from scipy.spatial.transform import Rotation
    selected=np.asarray(selected,np.int64);out=rows.copy()
    if not len(selected):return out,selected
    sig=np.hypot(np.exp(np.column_stack([rows[f'scale_{k}'][selected] for k in range(3)]).astype(float)),rows['filter_3D'][selected,None])
    raw=np.exp(np.column_stack([rows[f'scale_{k}'][selected] for k in range(3)]).astype(float))
    alpha=expit(rows['opacity'][selected].astype(float))*(raw/sig).prod(1)
    axis=sig.argmax(1)
    if mode=='surface_normal' and surface is not None and len(surface['points'])>=8:
        xyz=np.column_stack([rows[k][selected] for k in ['x','y','z']])
        _,ni=surface['tree'].query(xyz,k=8,workers=8)
        p=surface['points'][ni];p=p-p.mean(1,keepdims=True)
        _,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',p,p)/8)
        quat=np.column_stack([rows[f'rot_{k}'][selected] for k in range(4)])
        rotation=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
        projection=np.abs(np.einsum('nji,nj->ni',rotation,vec[:,:,0]))*sig
        axis=projection.argmax(1)
    elif mode!='longest' and mode!='surface_normal':
        raise ValueError('Unknown local covariance axis mode: '+mode)
    sig[np.arange(len(sig)),axis]*=factor
    for k in range(3):out[f'scale_{k}'][selected]=np.log(sig[:,k])
    out['opacity'][selected]=logit(np.clip(alpha,np.finfo(np.float32).tiny,1-np.finfo(np.float32).eps))
    out['filter_3D'][selected]=0
    return out,selected


def load_rows(scene, rows, folder):
    path = Path(folder)/'candidate.ply'
    save_ply(path, rows)
    model, _ = scene.load_model(path)
    return model, sha(path)


def run_stage(scene, input_rows, input_ids, config, output, stage, surface, provenance, lp, attribution_scene=None, attribution_free=None):
    config={**config,'active_stage':stage}
    begin = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, ids = input_rows.copy(), input_ids.copy()
    model, digest = load_rows(scene, rows, output/'input')
    quick = camera_indices(scene, config['evidence_views'])
    confirm = camera_indices(scene, config['confirmation_views'])
    # FPS is deterministic and nested; keep this invariant explicit.
    if not set(quick) <= set(confirm):
        raise ValueError('Quick cameras must be a subset of confirmation cameras')
    evaluator = HistoricalEvaluator(scene, output, lp=None, surface=surface, original_model=model)
    initial_confirm = evaluator.evaluate(model, output/'initial_confirmation', confirm)
    current_confirm = initial_confirm
    current_quick = subset_metrics(initial_confirm, quick)
    original_fg = np.zeros(int(input_ids.max())+1)
    original_fg_ready = False
    attribution_scene=attribution_scene or scene
    attribution_free=evaluator.free_views if attribution_free is None else attribution_free
    if config.get('original_foreground_views',0):
        envelope_indices=camera_indices(attribution_scene,config['original_foreground_views'])
        original_fg[ids]=foreground_envelope(attribution_scene,model,envelope_indices,output/'original_foreground',
            dict(provenance,model_sha256=digest))
        original_fg_ready=True
    edits, rejected = [], []
    max_rounds = config[f'stage{stage}_max_rounds']
    plateau, stop = 0, 'safety_round_limit'
    for iteration in range(max_rounds):
        folder = output/'rounds'/f'{iteration+1:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        identity = dict(provenance, model_sha256=digest)
        e = measure(attribution_scene, model, rows, ids, quick, attribution_free, folder/'evidence', identity)
        if not original_fg_ready:
            original_fg[ids] = e['fg']
            original_fg_ready = True
        spatial = component_features(rows, e, surface)
        spatial['sigma'] = np.sort(np.hypot(np.exp(np.column_stack([rows[f'scale_{k}'] for k in range(3)])), rows['filter_3D'][:, None]), axis=1)
        if stage == 1:
            remove, rationale = historical_candidates(e, original_fg, ids, iteration, spatial if iteration else None)
        else:
            fg = np.maximum(e['fg'], original_fg[ids])
            bg = e['bg']+e['free_bg']
            ratio = bg/np.maximum(fg+bg, 1e-12)
            floor = adaptive_floor(bg)
            # Disconnection and depth are supporting evidence; true dark
            # foreground and thin structures still require high harm ratio.
            unsupported = (e['depth_support'] < 2) & (e['bg_views'] >= 2)
            separate = spatial['isolated'] & unsupported & (ratio > .95)
            revealed = (ratio > .99) & (e['bg_views'] >= 2)
            remove = (bg > floor) & (separate | revealed)
            rationale = dict(component_candidates=int(separate.sum()), newly_revealed=int(revealed.sum()),
                contribution_floor=floor if np.isfinite(floor) else None)
        # One whole-round proposal is tested at 24 then 64 cameras. Rank-based
        # fractions are fallbacks, never a full208 recursive candidate splitter.
        proposals = []
        if remove.any():
            ranking = np.flatnonzero(remove)
            ranking = ranking[np.argsort(-(e['bg'][ranking]+e['free_bg'][ranking])/np.maximum(e['fg'][ranking], 1e-8))]
            for fraction in [1., .5, .25]:
                delete = ranking[:max(1, int(len(ranking)*fraction))]
                keep = np.ones(len(rows), bool)
                keep[delete] = False
                if keep.any():
                    proposals.append((f'prune_{fraction:g}', rows[keep], ids[keep], delete))
        if stage == 2:
            priors, bits = np.zeros(len(rows), bool), np.zeros(len(rows), np.uint16)
            if config.get('v3_discovery', True):
                f = features(rows)
                priors, bits, stats = rules(f, 3, True)
                write_json(folder/'v3_discovery.json', {**stats, 'role': 'candidate discovery only; no direct V3 deletion'})
            bg = e['bg']+e['free_bg']
            fg = np.maximum(e['fg'], original_fg[ids])
            mixed = (bg > adaptive_floor(bg)) & (fg > 0) & ((e['black']+e['free_black'] > adaptive_floor(e['black']+e['free_black'])) | priors)
            # Boundary/geometry-supported candidates can be adjusted, never
            # removed by the appearance prior. Test their local RGB explicitly.
            mixed &= (e['max_radius'] > .01)
            selected = np.flatnonzero(mixed)
            importance=(.7*e['free_black']/max(e['free_black'].sum(),1e-12)+.3*e['black']/max(e['black'].sum(),1e-12))
            selected = selected[np.argsort(-importance[selected])][:config['refine_max_candidates']]
            np.savez(folder/'residual_discovery.npz', original_ids=ids, candidate=mixed, v3_bits=bits,
                components=spatial['components'], spatially_isolated=spatial['isolated'])
            for mode in config.get('refine_axis_modes',['surface_normal']):
                for factor in config['refine_axis_factors']:
                    if config.get('refine_filtered_covariance',False):
                        candidate,changed=filtered_covariance_proposal(rows,selected,surface,factor,mode)
                    else:
                        candidate, changed = covariance_proposal(rows, selected, spatial, surface, factor)
                    if len(changed):
                        proposals.append((f'covariance_{mode}_{factor:g}', candidate, ids.copy(), changed))
        if not proposals:
            if stage == 1 and iteration == 0:
                # Round1 may find no near-pure background; still test the
                # historical Round2 .95 rule before declaring convergence.
                continue
            stop = 'no_supported_candidate'
            break
        best = None
        for proposal_position,(name, trial_rows, trial_ids, members) in enumerate(proposals):
            trial = folder/name
            candidate, trial_sha = load_rows(scene, trial_rows, trial)
            qmetrics = evaluator.evaluate(candidate, trial/'quick', quick)
            qreasons = quality_reasons(subset_metrics(initial_confirm, quick), qmetrics, config)
            qreasons += quality_reasons(current_quick, qmetrics, config)
            gain = score(qmetrics, current_quick)
            record = dict(name=name, round=iteration+1, stage=stage, changed=len(members),
                quick_gain=gain, quick_reasons=sorted(set(qreasons)), rationale=rationale,
                full_camera_candidate_evaluation=False)
            if qreasons or gain <= 0:
                rejected.append(record)
                write_json(trial/'decision.json', {**record, 'accepted': False})
                covariance_damage=(stage==2 and config.get('protect_covariance_damage',False)
                    and name==f'covariance_longest_{min(config["refine_axis_factors"]):g}')
                if (name=='prune_1' or covariance_damage) and qreasons and config.get('protect_measured_damage',False):
                    damage,damage_records=damage_evidence(scene,model,candidate,quick,evaluator.reference_rgb,evaluator.reference)
                    np.savez(trial/'foreground_damage.npz',evidence=damage,original_ids=ids,members=members)
                    write_json(trial/'foreground_damage.json',damage_records)
                    # A fixed number of whole-round corrections, each 24/64,
                    # protects the actual harmful contributors. No recursive
                    # small-group208 validation or plant-specific ROI.
                    corrected=[]
                    for coverage in config.get('damage_protection_fractions',[.98,1.]):
                        protect=protected_members(damage,members,coverage)
                        changed=members[~protect]
                        if len(changed) and protect.any():
                            if name=='prune_1':
                                keep=np.ones(len(rows),bool);keep[changed]=False
                                corrected.append((f'protected_prune_{coverage:g}',rows[keep],ids[keep],changed))
                            else:
                                corrected_rows=trial_rows.copy();corrected_rows[members[protect]]=rows[members[protect]]
                                corrected.append((f'protected_{name}_{coverage:g}',corrected_rows,ids.copy(),changed))
                    proposals[proposal_position+1:proposal_position+1]=corrected
                del candidate
                continue
            confirmed = evaluator.evaluate(candidate, trial/'confirmation', confirm)
            reasons = quality_reasons(initial_confirm, confirmed, config)+quality_reasons(current_confirm, confirmed, config)
            gain = score(confirmed, current_confirm)
            record.update(confirmation_gain=gain, confirmation_reasons=sorted(set(reasons)))
            feasible = not reasons and gain > 0
            write_json(trial/'decision.json', {**record, 'accepted': False, 'feasible': feasible})
            if feasible and (best is None or gain > best[0]):
                best = (gain, trial_rows, trial_ids, trial_sha, confirmed, qmetrics, record, trial)
            elif not feasible:
                rejected.append(record)
            del candidate
            # Broad prune accepted at64 needs no weaker fractional trials.
            if feasible and name == 'prune_1':
                break
        if best is None:
            stop = 'no_quality_preserving_gain'
            break
        gain, rows, ids, digest, current_confirm, current_quick, record, trial = best
        write_json(trial/'decision.json', {**record, 'accepted': True})
        edits.append({**record, 'accepted': True})
        save_ply(folder/'accepted.ply', rows)
        np.save(folder/'accepted_original_ids.npy', ids)
        model, _ = scene.load_model(folder/'accepted.ply')
        digest = sha(folder/'accepted.ply')
        # Next loop ALWAYS remeasures compositing for this exact changed model.
        plateau = plateau+1 if gain < config['minimum_relative_gain'] else 0
        write_json(output/'progress.json', dict(stage=stage, round=iteration+1, gaussians=len(rows), accepted=edits, rejected=len(rejected)))
        print(f'Stage {stage} round {iteration+1}: {len(rows)} Gaussians, gain {gain:.6f}', flush=True)
        if plateau >= config['plateau_patience']:
            stop = 'measured_gain_plateau'
            break
    # Only selected outputs get full-camera PSNR/SSIM/LPIPS/depth/normal checks.
    # Create a complete stage-input reference, including views unused at24/64.
    evaluator.close()
    del evaluator
    # Support masks belong to this stage's explicit input, not to another
    # stage or an earlier evaluator's photometric/geometry references.
    scene.original_foreground_support={}
    gc.collect()
    torch.cuda.empty_cache()
    baseline_model, _ = load_rows(scene, input_rows, output/'input')
    final_eval = HistoricalEvaluator(scene, output, lp=lp, surface=surface, original_model=baseline_model)
    initial = final_eval.evaluate(baseline_model, output/'comparison/initial')
    visuals(scene, baseline_model, output/'visualizations_input', surface)
    candidate, _ = load_rows(scene, rows, output/'selected')
    final = final_eval.evaluate(candidate, output/'comparison/selected')
    reasons = quality_reasons(initial, final, config)
    net_gain = score(final, initial)
    repair_record=None
    if edits and reasons and config.get('full_view_restoration',False):
        from .final_repair import edited_members,restore_members,repair_view_indices,restoration_evidence
        repair_folder=output/'final_restoration';repair_folder.mkdir(parents=True,exist_ok=True)
        members=edited_members(input_rows,input_ids,rows,ids)
        repair_views=repair_view_indices(initial,final,config['evidence_views'])
        restore=restoration_evidence(scene,baseline_model,candidate,repair_views,final_eval,members,repair_folder)
        repair_record=dict(trigger_reasons=reasons,restored_rows=len(restore),accepted=False)
        if len(restore):
            repaired_rows,repaired_ids=restore_members(input_rows,input_ids,rows,ids,restore)
            repaired,_=load_rows(scene,repaired_rows,repair_folder/'proposal')
            # Diagnostics target newly discovered failure views. Confirmation
            # retains automatic camera coverage and stays within64 views.
            confirm_repair=sorted(set(repair_views)|set(i for i in confirm if i not in repair_views) )
            if len(confirm_repair)>config['confirmation_views']:
                fill=[i for i in confirm if i not in repair_views]
                confirm_repair=sorted(repair_views+fill[:config['confirmation_views']-len(repair_views)])
            fast=HistoricalEvaluator(scene,repair_folder,original_model=baseline_model)
            repair_initial=fast.evaluate(baseline_model,repair_folder/'initial_confirmation',confirm_repair)
            repair_quick=fast.evaluate(repaired,repair_folder/'quick',repair_views)
            quick_reasons=quality_reasons(subset_metrics(repair_initial,repair_views),repair_quick,config)
            repair_record['quick_reasons']=quick_reasons
            full_repair_ready=False
            if not quick_reasons and score(repair_quick,subset_metrics(repair_initial,repair_views))>0:
                repair_confirm=fast.evaluate(repaired,repair_folder/'confirmation',confirm_repair)
                confirmation_reasons=quality_reasons(repair_initial,repair_confirm,config)
                repair_record['confirmation_reasons']=confirmation_reasons
                if not confirmation_reasons and score(repair_confirm,repair_initial)>0:
                    full_repair_ready=True
            fast.close()
            del fast
            gc.collect()
            if full_repair_ready:
                # Only the selected restored whole-stage output receives one
                # additional full evaluation, never each restored row.
                repaired_final=final_eval.evaluate(repaired,repair_folder/'selected_full')
                repaired_reasons=quality_reasons(initial,repaired_final,config)
                repair_record['full_view_reasons']=repaired_reasons
                if not repaired_reasons and score(repaired_final,initial)>0:
                    save_ply(output/'rejected_before_restoration.ply',rows)
                    rows,ids,candidate,final=repaired_rows,repaired_ids,repaired,repaired_final
                    reasons=[];net_gain=score(final,initial);repair_record['accepted']=True
                    edits.append(dict(name='final_coverage_restoration',restored_original_ids=input_ids[restore].tolist()))
        write_json(repair_folder/'decision.json',repair_record)
    rolled_back = bool(edits) and (bool(reasons) or net_gain <= 0)
    if rolled_back:
        save_ply(output/'rejected_selected.ply', rows)
        rows, ids, model, final = input_rows.copy(), input_ids.copy(), baseline_model, initial
        stop = 'full_view_rollback'
    else:
        model = candidate
    export(output, input_rows, rows, np.searchsorted(input_ids, ids))
    # export() writes IDs relative to stage input; keep both row namespaces.
    np.save(output/'stage_input_row_ids.npy', np.searchsorted(input_ids, ids))
    np.save(output/'original_gaussian_ids.npy', ids)
    removed = ~np.isin(input_ids, ids)
    np.save(output/'removed_original_gaussian_ids.npy', input_ids[removed])
    parity(scene, model, output)
    visuals(scene, model, output/'visualizations', surface)
    write_changes(output/'parameter_changes.csv', input_rows, input_ids, rows, ids)
    result = dict(stage=stage, input_gaussians=len(input_rows), final_gaussians=len(rows),
        removed_gaussians=int(removed.sum()), refined_gaussians=count_changes(input_rows, input_ids, rows, ids),
        seconds=time.perf_counter()-begin, initial=initial, final=final,
        score=score(final, initial), full_view_rejection_reasons=reasons, rolled_back=rolled_back,
        final_restoration=repair_record,
        foreground_acceptance_policy=('Historical Stage1: maintained structure and bounded FG-PSNR loss may accept measured dark-shadow RGB tradeoff'
            if stage==1 and config.get('stage1_historical_photo_tradeoff',False) else 'Strict local and whole-foreground photo/structure guards'),
        stop=stop, accepted=edits, rejected=rejected, quick_camera_names=[scene.views[i]['name'] for i in quick],
        confirmation_camera_names=[scene.views[i]['name'] for i in confirm],
        metrics_complete=True, native_model=str(output/'cleaned_gaussians.ply'),
        supersplat_model=str(output/'cleaned_gaussians_supersplat.ply'))
    write_json(output/'cleanup_metrics.json', result)
    write_json(output/'COMPLETE.json', dict(native_sha256=sha(output/'cleaned_gaussians.ply'), metrics_sha256=sha(output/'cleanup_metrics.json')))
    final_eval.close()
    del final_eval, model, baseline_model, candidate
    gc.collect()
    torch.cuda.empty_cache()
    return rows, ids, result


def count_changes(original, original_ids, rows, ids):
    before = original[np.searchsorted(original_ids, ids)]
    changed = np.zeros(len(rows), bool)
    for key in rows.dtype.names:
        changed |= before[key] != rows[key]
    return int(changed.sum())


def write_changes(path, original, original_ids, rows, ids):
    before = original[np.searchsorted(original_ids, ids)]
    with Path(path).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['original_gaussian_id', 'field', 'old', 'new'])
        for key in rows.dtype.names:
            for i in np.flatnonzero(before[key] != rows[key]):
                w.writerow([int(ids[i]), key, float(before[key][i]), float(rows[key][i])])


def link(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    if target.exists():
        raise FileExistsError(target)
    target.symlink_to(source)


def run(spec, config, output, stage2_input=None):
    from ...execution_provenance import scene_input_fingerprint
    begin = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = Scene(spec, config)
    original = read_ply(spec['model'])
    source_sha = sha(spec['model'])
    scene_digest = config['scene_input_fingerprint']
    provenance = dict(scene_input_fingerprint=scene_digest, config=config)
    write_json(output/'original_reference.json', dict(scene=spec, config=config, input_sha256=source_sha, physical_gpu=1))
    indices = camera_indices(scene, config['confirmation_views'])
    surface = external_surface(scene, indices)
    if surface is None:
        raise ValueError('Two-stage cleaning requires the supplied front-end depth/Camera/Mask geometry support')
    import lpips
    lp = lpips.LPIPS(net='alex').cuda().eval()
    attribution_scene=scene
    attribution_free=None
    if config.get('attribution_max_side',config['evaluation_max_side'])!=config['evaluation_max_side']:
        from ..adaptive.free_support import free_support
        attribution_scene=Scene(spec,{**config,'evaluation_max_side':config['attribution_max_side']})
        attribution_free=free_support(attribution_scene,surface)
    stage1_rows, stage1_ids, s1 = run_stage(scene, original, np.arange(len(original)), config,
        output/'stage1_generalized', 1, surface, provenance, lp,attribution_scene,attribution_free)
    # A supplied historical reference is used as requested for the development
    # Stage2 comparison. Other scenes automatically consume Stage1 output.
    historical = stage2_input or spec.get('baseline_models', {}).get('R5')
    if historical:
        historical = Path(historical).resolve()
        stage2_rows = read_ply(historical)
        stage2_ids = np.load(historical.parent/'original_gaussian_ids.npy')
        if (len(stage2_ids) != len(stage2_rows) or np.any(np.diff(stage2_ids) <= 0) or
                stage2_ids.min() < 0 or stage2_ids.max() >= len(original) or
                not np.array_equal(original[stage2_ids], stage2_rows)):
            raise ValueError('Historical baseline lacks exact unchanged native row lineage to this scene input')
        link(historical, output/'historical_baseline/cleaned_gaussians.ply')
        viewer = historical.with_name(historical.stem+'_supersplat.ply')
        if not viewer.exists():
            raise FileNotFoundError(viewer)
        link(viewer, output/'historical_baseline/cleaned_gaussians_supersplat.ply')
        # Read old residual evidence ONLY with matching model SHA and row IDs.
        old_evidence = historical.parent/'remaining_actual_contribution.npz'
        old_meta = historical.parent/'remaining_dependencies.json'
        if old_evidence.exists() and old_meta.exists():
            meta = json.loads(old_meta.read_text())
            with np.load(old_evidence) as z:
                matches = meta.get('final_ply_sha256') == sha(historical) and np.array_equal(z['original_ids'], stage2_ids)
            write_json(output/'historical_baseline/residual_evidence_audit.json', dict(valid=matches,
                path=str(old_evidence), candidate_discovery_only=True,
                attribution_protocol='historical fixed8; remeasured under automatic views before any edit',
                historical_dark_contributors=meta['gaussians_with_remaining_black_weight_gt005']))
            if matches:
                link(old_evidence, output/'historical_baseline/remaining_actual_contribution.npz')
                link(historical.parent/'remaining_artifact_dependencies.csv', output/'historical_baseline/remaining_artifact_dependencies.csv')
    else:
        stage2_rows, stage2_ids = stage1_rows, stage1_ids
    final_rows, final_ids, s2 = run_stage(scene, stage2_rows, stage2_ids, config,
        output/'stage2_residual', 2, surface, provenance, lp,attribution_scene,attribution_free)
    for stage, folder in [(1, 'stage1_generalized'), (2, 'stage2_residual')]:
        for src, dst in [('cleaned_gaussians.ply', f'stage{stage}_cleaned_gaussians.ply'),
                         ('cleaned_gaussians_supersplat.ply', f'stage{stage}_supersplat.ply'),
                         ('removed_gaussians.ply', f'removed_stage{stage}.ply')]:
            link(output/folder/src, output/dst)
    # Stage2 rollback exports its unchanged input, including the historical
    # best when provided. It never silently reverts to the poorer C5-C asset.
    for name in ['cleaned_gaussians.ply', 'cleaned_gaussians_supersplat.ply', 'original_gaussian_ids.npy']:
        link(output/'stage2_residual'/name, output/name)
    all_removed = np.ones(len(original), bool)
    all_removed[final_ids] = False
    save_ply(output/'removed_gaussians.ply', original[all_removed])
    removed_viewer, _ = bake_viewer(original[all_removed])
    save_ply(output/'removed_gaussians_supersplat.ply', removed_viewer)
    np.save(output/'removed_original_gaussian_ids.npy', np.flatnonzero(all_removed))
    link(output/'stage2_residual/viewer_baking.json', output/'viewer_baking.json')
    models = dict(original=str(Path(spec['model']).resolve()), historical=str(historical) if historical else None,
        stage1=str(output/'stage1_cleaned_gaussians.ply'), stage2=str(output/'stage2_cleaned_gaussians.ply'),
        final=str(output/'cleaned_gaussians.ply'), supersplat=str(output/'cleaned_gaussians_supersplat.ply'))
    result = dict(plant=spec['plant'], method='historical_two_stage', scene=spec, config=config,
        input_sha256=source_sha, stage2_input=str(historical) if historical else s1['native_model'],
        stage1=s1, stage2=s2, models=models, seconds=time.perf_counter()-begin,
        source_protocol='historical native rounds1/2/4/5; no global Round3 scaling, mesh edits, ROI or retraining',
        visual_review='native renders and viewer parity exported; interactive SuperSplat review pending')
    write_json(output/'cleanup_metrics.json', result)
    write_json(output/'model_index.json', models)
    if sha(spec['model']) != source_sha:
        raise RuntimeError('Native GGGS input changed during two-stage execution')
    report = [f'# {spec["plant"]} — Historical Two-Stage Cleaner', '',
        f'Stage1: {s1["input_gaussians"]} → {s1["final_gaussians"]}; {s1["seconds"]:.1f}s; stop={s1["stop"]}.',
        f'Stage2: {s2["input_gaussians"]} → {s2["final_gaussians"]}; adjusted={s2["refined_gaussians"]}; {s2["seconds"]:.1f}s; stop={s2["stop"]}.',
        'Final models received full observed-camera evaluation. Automatic depth silhouettes are proxies.',
        'Interactive SuperSplat visual review remains separate from quantitative acceptance.', '',
        *[f'- {k}: `{v}`' for k, v in models.items() if v], '',
        'Sources: actual local native GGGS renders and recorded edit manifests. Bocha search unavailable (HTTP403 quota).']
    (output/'CLEANING_REPORT.md').write_text('\n'.join(report)+'\n')
    artifacts = ['cleaned_gaussians.ply','cleaned_gaussians_supersplat.ply','removed_gaussians.ply',
        'removed_gaussians_supersplat.ply','original_gaussian_ids.npy','removed_original_gaussian_ids.npy','viewer_baking.json']
    write_json(output/'COMPLETE.json', dict(input_sha256=source_sha, scene=spec, config=config, models=models,
        artifact_hashes={name:sha(output/name) for name in artifacts},
        native_sha256=sha(output/'cleaned_gaussians.ply'), full_view_evaluation=True))
    return result
