"""Grouped multiview GGGS suppression; only verified edits become final rows."""
import time
import json
from pathlib import Path
import numpy as np
import torch
from ..runtime import Scene
from ..io import save_ply, write_json, sha
from ..contribution_analysis import analyze, contribution
from ..geometry_consistency import external_surface
from ..gaussian_refinement import refine
from .legacy import features, rules
from .grouping import groups, split_group
from .evaluation import Evaluator
from .selection import choose
from .artifacts import export, parity, visuals
from .mixed_refinement import refine_spatial_groups
from .free_support import free_support
from .rejection_split import split_rejected
from .foreground_support import add_supported_contribution
from .center_support import center_support


def native_subset(scene, model, keep):
    candidate = scene.Model(model.active_sh_degree, model.active_sg_degree)
    candidate.active_sh_degree = model.active_sh_degree
    candidate.active_sg_degree = model.active_sg_degree
    index = torch.as_tensor(np.flatnonzero(keep), device='cuda')
    for key in ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation',
                '_sg_axis','_sg_sharpness','_sg_color','filter_3D']:
        if hasattr(model,key):
            # Advanced indexing already materializes independent row storage.
            setattr(candidate,key,getattr(model,key).detach()[index])
    return candidate


def reattribute(scene, model, indices, config, surface):
    e = analyze(scene, model, indices, config)
    return add_free_evidence(scene,model,e,surface)


def add_free_evidence(scene,model,e,surface):
    from ..runtime import camera
    e['free_bg'] = np.zeros(len(model.get_xyz))
    e['free_aabb_bg'] = np.zeros(len(model.get_xyz))
    e['free_visible_views'] = np.zeros(len(model.get_xyz),np.int32)
    definitions = free_support(scene,surface)
    for uid,support in enumerate(definitions):
        v=support['view']
        c=camera(v['K'],v['E'],v['width'],v['height'],v['name'],uid)
        denominator=support['normalizer']
        g,_,_=contribution(c,model,np.stack([support['outside'],support['surface_mask'],support['outside_aabb']]))
        e['free_bg'] += g[:,0]/denominator/max(len(definitions),1)
        e['free_aabb_bg'] += g[:,2]/denominator/max(len(definitions),1)
        e['free_visible_views'] += g[:,0]>denominator*1e-7
    return e


def background_selection_score(metrics,initial):
    training=(initial['mean']['bg_alpha_mass_per_fg']-metrics['mean']['bg_alpha_mass_per_fg'])/max(initial['mean']['bg_alpha_mass_per_fg'],1e-12)
    before=initial.get('free_background',{}).get('alpha_mass',0)
    free=(before-metrics.get('free_background',{}).get('alpha_mass',before))/max(before,1e-12)
    training=float(np.clip(training,-1,1));free=float(np.clip(free,-1,1))
    def gain(key):
        old=initial['mean'].get(key,0);new=metrics['mean'].get(key,old)
        return float(np.clip((old-new)/old,-1,1)) if old>1e-12 else (0. if new<=1e-12 else -1.)
    composite=.5*training+.2*gain('bg_high_alpha_pixels')+.15*gain('largest_bg_component_fraction')+.05*gain('small_bg_pixels')+.1*free
    return composite,training


def require_final_quality(verified,reasons,modified):
    # An unchanged original may honestly report no accepted modification.
    # Any edited publication needs complete quality AND actual net benefit.
    if (modified and not verified) or any('benefit' not in reason for reason in reasons):
        raise ValueError('Full-camera final quality/benefit validation failed; output is not certified')


def validate_native_input(rows):
    required=['x','y','z','opacity','filter_3D']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_dc_{i}' for i in range(3)]
    if not len(rows) or any(name not in rows.dtype.names for name in required):
        raise ValueError('Requires a nonempty native GGGS checkpoint with three scales, rotation, SH, opacity and stored 3D filter')
    if sorted(name for name in rows.dtype.names if name.startswith('scale_'))!=['scale_0','scale_1','scale_2']:
        raise ValueError('Native GGGS requires exactly three encoded spatial scales')
    if np.any(rows['filter_3D']<0) or not np.any(rows['filter_3D']>0):
        raise ValueError('Expected the trained native GGGS 3D filter; a baked viewing PLY is not a native input checkpoint')
    for offset in range(0,len(rows),250000):
        chunk=rows[offset:offset+250000]
        if not all(np.isfinite(chunk[name]).all() for name in rows.dtype.names):
            raise ValueError('Nonfinite native GGGS parameters')


def run(spec,config,output,variant='C5-D',baseline=None,evidence_cache=None):
    begin=time.perf_counter()
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    scene=Scene(spec,config)
    model,original=scene.load_model(spec['model'])
    validate_native_input(original)
    rows=original.copy();ids=np.arange(len(rows))
    original_sha=sha(spec['model'])
    write_json(output/'original_reference.json',dict(scene=spec,config=config,variant=variant,input_sha256=original_sha,
        original_gaussians=len(rows),physical_gpu=1))
    import lpips
    lp=lpips.LPIPS(net='alex').cuda().eval()
    indices=scene.evidence_indices()
    surface=external_surface(scene,indices)
    evaluator=Evaluator(scene,output,baseline,lp,surface,original_model=model)
    initial=evaluator.evaluate(model,output/'comparison/initial')
    current=initial
    original_effective_opacity=model.get_opacity_with_3D_filter.detach().cpu().numpy().flatten() if variant=='C5-D' else None
    strict_centers=None
    if variant!='C5-A' and config.get('protect_supported_centers',False):
        strict_centers=center_support(scene,model,config)
        np.save(output/'original_center_depth_support.npy',strict_centers)
        write_json(output/'center_protection.json',dict(protected=int((strict_centers>=config.get('center_minimum_views',2)).sum()),
            definition='Original Gaussian center agrees with foreground high-confidence VGGT depth within 3% of median foreground depth in at least two supplied cameras; independent of Gaussian scale/footprint. These centers are retained and mixed contributors may be locally refined.',proxy_not_semantic_ground_truth=True))
    selected_score=(0.,0.);selected_round=0
    f=features(rows)
    raw_removed,bits,rule_summary=rules(f,3,True)
    np.savez(output/'discovery_priors.npz',original_ids=ids,appearance_candidates=raw_removed,
        reason_bits=bits,brightness=f['brightness'],effective_scales=f['effective'],knn_dark=f['knn_normalized'])
    write_json(output/'appearance_priors.json',rule_summary)
    edits=[];stop='safety_round_limit';original_fg=None;refined_ids=set()
    for iteration in range(config['max_rounds']):
        dest=output/'rounds'/f'{iteration+1:02d}';dest.mkdir(parents=True,exist_ok=True)
        if variant=='C5-A':
            # The appearance-only decision does not compute or consume GGGS
            # attribution. Shared native rendering is used for evaluation.
            prior=raw_removed[ids]
            keep=~prior
            rows,ids=rows[keep].copy(),ids[keep]
            model=native_subset(scene,model,keep)
            current=evaluator.evaluate(model,dest/'appearance_only')
            edits.append(dict(round=iteration+1,action='appearance_only_unconstrained_ablation',candidates=int(prior.sum()),
                accepted=True,certified=False,removed_ids=np.flatnonzero(prior).tolist()))
            stop='appearance_only_ablation';break
        if iteration==0 and evidence_cache:
            with np.load(evidence_cache) as z:
                e={k:z[k].copy() for k in z.files if k not in ['original_ids','labels','harm','appearance_candidates','prune','mixed']}
            for key in ['fg','boundary','thin']:
                if 'raw_'+key in e:e[key]=e['raw_'+key].copy()
            if len(e['fg'])!=len(rows):
                raise ValueError('Original evidence cache row mismatch')
            write_json(dest/'evidence_cache_provenance.json',dict(path=str(evidence_cache),sha256=sha(evidence_cache),
                input_model_sha256=original_sha,scope='Native all-208 training-view evidence; no historical free ROI'))
            e=add_free_evidence(scene,model,e,surface)
        else:
            first_count=config.get('initial_evidence_views',config['evidence_views'])
            first_indices=list(range(len(scene.views))) if first_count<=0 or first_count>=len(scene.views) else indices
            e=reattribute(scene,model,first_indices if iteration==0 else indices,config,surface)
        if config.get('foreground_support_mode')=='photo_relative_or_depth':
            first_count=config.get('initial_evidence_views',config['evidence_views'])
            support_indices=list(range(len(scene.views))) if iteration==0 and (first_count<=0 or first_count>=len(scene.views)) else indices
            e=add_supported_contribution(scene,model,e,support_indices,config)
        if original_fg is None:
            original_fg=e['fg'].copy()
        e['fg']=np.maximum(e['fg'],original_fg[ids])
        records=e.pop('records',[])
        bg=e['bg']+e['free_bg']
        fraction=bg/np.maximum(e['fg']+bg,1e-15)
        sensitive=(e['boundary']+e['thin'])/np.maximum(e['fg'],1e-15)>.2
        prior=raw_removed[ids]
        appearance_supported=prior & (fraction>config['mixed_candidate_background_fraction'])
        render_supported=fraction>config['candidate_background_fraction']
        valid=(bg>1e-7)&((e['visible_views']>=2)|(e['free_visible_views']>=2))
        safe=valid&(appearance_supported|render_supported)&~sensitive
        if strict_centers is not None:
            e['strict_center_geometry_views']=strict_centers[ids]
            safe &= strict_centers[ids]<config.get('center_minimum_views',2)
        if not config.get('depth_proxy_as_hint',False):
            safe &= (e['geometry_views']<2)&(e['occlusion_views']==0)
        prune=np.flatnonzero(safe)
        # Order by predicted benefit at bounded FG risk, not stable zero-FG ties.
        priority=bg[prune]/(1+e['fg'][prune]/1e-8)
        prune=prune[np.argsort(-priority,kind='stable')]
        budget=config['candidate_foreground_budget']*e['fg'].sum()
        prune=prune[np.cumsum(e['fg'][prune])<=budget]
        mixed=np.flatnonzero(valid&(e['fg']>1e-7)&(fraction>.10)&~safe)
        np.savez(dest/'evidence.npz',original_ids=ids,**e,appearance_candidates=prior,prune=prune,mixed=mixed)
        write_json(dest/'discovery_summary.json',dict(appearance=int(prior.sum()),render_candidates=int(render_supported.sum()),
            prune=len(prune),mixed=len(mixed),records=records,knn_candidates=int(f['knn_normalized'].sum()),
            knn_additional_to_other_priors=int(((bits[ids]==(1<<7)) & f['knn_normalized']).sum())))
        round_before=current['mean']['bg_alpha_mass_per_fg'];round_free_before=current.get('free_background',{}).get('alpha_mass',0);accepted_round=False
        if variant=='C5-B':
            grouped=[prune] if len(prune) else []
        else:
            grouped=groups(prune,f,{**e,'bg':bg},config)
        queue=[(ids[g].copy(),0,False) for g in grouped]
        evidence_positions=np.full(len(original),-1,np.int64)
        evidence_positions[ids]=np.arange(len(ids))
        def pending_benefit(entry):
            position=evidence_positions[entry[0]]
            return float(bg[position].sum())
        trials=0;gradient_splits=0;gradient_attempts=0
        while queue and trials<config['group_max_trials']:
            if config.get('benefit_priority_queue',False):
                # Splitting one difficult region must not exhaust the trial
                # budget before independent high-benefit groups are tested.
                queue.sort(key=lambda entry:(entry[2],pending_benefit(entry)),reverse=True)
            original_group,level,gradient_proposal=queue.pop(0)
            local=np.flatnonzero(np.isin(ids,original_group))
            if not len(local):continue
            trials+=1
            keep=np.ones(len(rows),bool);keep[local]=False
            candidate=native_subset(scene,model,keep)
            folder=dest/f'group_{trials:03d}'
            metrics,affected=evaluator.evaluate_group(candidate,folder,original_group,current)
            accepted,gain,local_gain,reasons=choose(metrics,current,initial,config)
            # Current evidence indices map back to round-start IDs after earlier group edits.
            evid_ids=evidence_positions[ids[local]]
            item=dict(round=iteration+1,action='group_prune',group=trials,split_depth=level,candidates=len(local),
                removed_ids=ids[local].tolist(),accepted=accepted,relative_gain=gain,local_view_gain=local_gain,reasons=reasons,
                gain_definition=('max of actual training-background and configured auxiliary free-background relative improvements' if metrics.get('metrics_complete',True) else 'max of partial measured training-background and exact auxiliary free-background improvements; rejected diagnostic only'),
                training_relative_gain=(current['mean']['bg_alpha_mass_per_fg']-metrics['mean']['bg_alpha_mass_per_fg'])/max(current['mean']['bg_alpha_mass_per_fg'],1e-12),
                free_relative_gain=(current.get('free_background',{}).get('alpha_mass',0)-metrics.get('free_background',{}).get('alpha_mass',0))/max(current.get('free_background',{}).get('alpha_mass',0),1e-12),
                predicted_bg=float(bg[evid_ids].sum()),predicted_fg=float(e['fg'][evid_ids].sum()),
                affected_views=[scene.views[i]['name'] for i in affected],
                metrics_complete=metrics.get('metrics_complete',True),
                rerendered_views=metrics.get('rerendered_views',affected),
                background_gain_scope=metrics.get('background_gain_scope','Complete actual affected-view evaluation plus proven unchanged views'),
                dark_mse_tradeoff_analysis=metrics.get('dark_mse_tradeoff_analysis',[]),
                depth_recovery_analysis=metrics.get('depth_recovery_analysis',[]),
                metrics=str((folder/'metrics.json').resolve()))
            edits.append(item);write_json(output/'progress.json',item)
            write_json(folder/'decision.json',item)
            print('group',trials,'count',len(local),'accepted',accepted,'gain',gain,'reasons',reasons,flush=True)
            if accepted:
                rows,ids,model,current=rows[keep].copy(),ids[keep],candidate,metrics
                accepted_round=True
                save_ply(output/'best_so_far.ply',rows)
                np.save(output/'original_gaussian_ids.npy',ids)
            else:
                if variant!='C5-B' and level<config['group_max_split_depth'] and len(local)>1 and any('benefit' not in r for r in reasons):
                    native_pieces=[]
                    if (config.get('native_gradient_subdivision',False) and gradient_attempts<config.get('gradient_splits_per_round',8)
                            and len(local)<=config.get('gradient_split_max_gaussians',512)):
                        gradient_attempts+=1
                        native_pieces=split_rejected(scene,model,candidate,local,ids,bg[evid_ids],reasons,current,initial,config,folder)
                        if native_pieces:gradient_splits+=1
                    focused=evaluator.split_for_rejection(original_group,reasons) if not native_pieces else []
                    if native_pieces:
                        queue[0:0]=[(native_pieces[0],level+1,True),(native_pieces[1],level+1,False)]
                    elif focused:
                        queue[0:0]=[(g.copy(),level+1,False) for g in focused]
                    else:
                        pieces=split_group(local,np.column_stack([rows[k] for k in ['x','y','z']]))
                        queue[0:0]=[(ids[g].copy(),level+1,False) for g in pieces]
                del candidate
        write_json(dest/'group_scheduling.json',dict(trials=trials,native_gradient_analyses_attempted=gradient_attempts,native_gradient_partitions=gradient_splits,untried_groups=len(queue),
            untried_predicted_bg=sum(pending_benefit(entry) for entry in queue),
            order='A native-gradient proposed child gets one immediate nonlinear test; other groups and split children follow descending native predicted background contribution' if config.get('benefit_priority_queue',False) else 'Recursive split-first historical development schedule',
            untried_groups_are_not_claimed_safe=True))
        # Reattribute after accepted pruning before mixed local optimization.
        if variant=='C5-D' and config['refinement']:
            if accepted_round:
                e=reattribute(scene,model,indices,config,surface)
                if config.get('foreground_support_mode')=='photo_relative_or_depth':
                    e=add_supported_contribution(scene,model,e,indices,config)
                bg=e['bg']+e['free_bg']
                fraction=bg/np.maximum(e['fg']+bg,1e-15)
                mixed=np.flatnonzero((e['fg']>1e-7)&(fraction>.10)&(e['visible_views']>=2))
            if len(mixed):
                model,rows,current,refined,changed=refine_spatial_groups(
                    scene,model,rows,ids,mixed,e,config,evaluator,current,initial,dest,edits,iteration)
                refined_ids.update(changed)
                evaluator.modified_original_ids.update(changed)
                accepted_round |= refined
        if variant!='C5-A' and accepted_round:
            eligible,_,_,_=choose(current,initial,initial,config)
            score=background_selection_score(current,initial)
            if eligible and score>selected_score:
                selected_score,selected_round=score,iteration+1
                save_ply(output/'quality_selected_gaussians.ply',rows)
                np.save(output/'quality_selected_original_ids.npy',ids)
        write_json(output/'edit_manifest.json',dict(edits=edits,input_sha256=original_sha,current_gaussians=len(rows)))
        export(output,original,rows,ids)
        if not accepted_round:
            stop='no_verified_improving_group';break
        if variant=='C5-B':stop='single_batch_ablation';break
        round_gain=(round_before-current['mean']['bg_alpha_mass_per_fg'])/max(round_before,1e-12)
        round_gain=max(round_gain,(round_free_before-current.get('free_background',{}).get('alpha_mass',round_free_before))/max(round_free_before,1e-12))
        if round_gain<config['round_minimum_relative_gain']:
            stop='automatic_background_plateau';break
        f=features(rows,knn=False)
        # Original appearance identities are stable; local geometric features recomputed.
        f['knn_normalized']=np.isin(ids,np.flatnonzero(raw_removed & (bits & (1<<7) != 0)))
    final=evaluator.evaluate(model,output/'comparison/final')
    parameter_fields=['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    final_eligible=True if variant=='C5-A' else choose(final,initial,initial,config)[0]
    if variant!='C5-A' and (background_selection_score(final,initial)<selected_score or (not final_eligible and selected_round>0)):
        path=output/'quality_selected_gaussians.ply' if selected_round else Path(spec['model'])
        model,rows=scene.load_model(path)
        ids=np.load(output/'quality_selected_original_ids.npy') if selected_round else np.arange(len(original))
        final=evaluator.evaluate(model,output/'comparison/final')
        write_json(output/'MODEL_SELECTION.json',dict(selected_round=selected_round,selected_score=selected_score,
            criterion='Highest complete quality-certified composite of training alpha mass (0.5), opaque background area (0.2), largest cloud area (0.15), small-component pixels (0.05) and auxiliary free alpha (0.1); training alpha gain breaks ties. Component gains are bounded to [-1,1].',restored=True))
    else:
        write_json(output/'MODEL_SELECTION.json',dict(selected_round=iteration+1,selected_score=background_selection_score(final,initial),restored=False))
    final_refined=np.logical_or.reduce([rows[k]!=original[k][ids] for k in parameter_fields])
    opacity_summary=dict(refined=0,scope='Per-call local trust may compound; these ratios compare final retained primitives to their untouched original effective opacity')
    if final_refined.any() and original_effective_opacity is not None:
        before_opacity=original_effective_opacity[ids[final_refined]]
        after_opacity=model.get_opacity_with_3D_filter.detach().cpu().numpy().flatten()[final_refined]
        ratios=np.divide(after_opacity,before_opacity,out=np.ones_like(after_opacity),where=before_opacity>0)
        opacity_summary.update(refined=int(final_refined.sum()),minimum_ratio=float(ratios.min()),median_ratio=float(np.median(ratios)),
            maximum_ratio=float(ratios.max()),zero_original_opacity_rows=int((before_opacity==0).sum()))
    modified=(len(rows)!=len(original)) or bool(final_refined.any())
    if variant!='C5-A':
        verified,_,_,verification_reasons=choose(final,initial,initial,config)
        if not verified and (modified or any('benefit' not in r for r in verification_reasons)):
            write_json(output/'FULL_VALIDATION_FAILURE.json',dict(reasons=verification_reasons,proposed=final))
        require_final_quality(verified,verification_reasons,modified)
    export(output,original,rows,ids)
    parity(scene,model,output)
    visuals(scene,model,output/'visualizations',surface)
    gain=(initial['mean']['bg_alpha_mass_per_fg']-final['mean']['bg_alpha_mass_per_fg'])/max(initial['mean']['bg_alpha_mass_per_fg'],1e-12)
    free_before=initial.get('free_background',{}).get('alpha_mass',0)
    free_gain=(free_before-final.get('free_background',{}).get('alpha_mass',free_before))/max(free_before,1e-12)
    result=dict(plant=spec['plant'],method=variant,original_gaussians=len(original),cleaned_gaussians=len(rows),
        removed_gaussians=len(original)-len(rows),refined_gaussians=int(final_refined.sum()),
        ever_refined_gaussians=len(refined_ids),artifact_reduction=gain,
        cumulative_effective_opacity=opacity_summary,
        free_artifact_reduction=free_gain,
        initial=initial,final=final,stop_reason=stop,rounds=iteration+1,seconds=time.perf_counter()-begin,
        outcome=('uncertified_appearance_only_ablation' if variant=='C5-A' else
                 'measured_improvement_with_residuals' if modified and max(gain,free_gain)>0 else 'no_accepted_modification'),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        native_ply=str(output/'cleaned_gaussians.ply'),supersplat_ply=str(output/'cleaned_gaussians_supersplat.ply'),
        removed_ply=str(output/'removed_gaussians.ply'),input_sha256=original_sha,config=config,
        selection_objective={'training_alpha':.5,'opaque_background_area':.2,'largest_cloud_area':.15,'small_component_pixels':.05,'free_alpha':.1},
        semantic_certification='Binary-mask/depth/optional fixed original photo-depth support proxies; full-mask dark photometry and excluded-region local loss measured; no leaf-vein semantic GT',
        closest_rejected_candidates=sorted([x for x in edits if not x['accepted']],key=lambda x:-x['relative_gain'])[:5],
        closest_rejected_rank_scope='Measured relative gain; preflight-rejected trials have partial training gains explicitly marked, not full global measurements',
        largest_predicted_rejected_candidates=sorted([x for x in edits if not x['accepted']],key=lambda x:-x.get('predicted_bg',0))[:5])
    write_json(output/'cleanup_metrics.json',result)
    from ForeSplat_AutoGaussianCleaner.execution_provenance import ARTIFACT_FILES
    write_json(output/'COMPLETE.json',dict(input_sha256=original_sha,native_sha256=sha(output/'cleaned_gaussians.ply'),config=config,scene=spec,variant=variant,
        artifact_hashes={name:sha(output/name) for name in ARTIFACT_FILES}))
    return result
