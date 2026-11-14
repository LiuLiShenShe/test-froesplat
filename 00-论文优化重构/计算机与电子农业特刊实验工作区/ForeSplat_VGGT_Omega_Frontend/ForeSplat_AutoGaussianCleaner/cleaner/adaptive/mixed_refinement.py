"""Independent constrained covariance optimization of spatial mixed groups."""
import numpy as np
from ..gaussian_refinement import refine
from ..io import save_ply,write_json,sha
from .legacy import features
from .grouping import groups
from .selection import choose
from .free_support import free_support


def refine_spatial_groups(scene,model,rows,ids,mixed,e,config,evaluator,current,initial,dest,edits,iteration):
    import torch
    write_json(dest/'runtime_mixed_source.json',dict(source_file=__file__,sha256=sha(__file__),
        note='Runtime helper source recorded as well as startup source snapshots'))
    local_config={**config,'group_max_gaussians':config['refine_group_max'],'group_population_fraction':0}
    f=features(rows,knn=False)
    scheduled=groups(mixed,f,{**e,'bg':e['bg']+e.get('free_bg',0)},local_config)[:config['refine_groups_per_round']]
    accepted_any=False;changed_ids=set()
    np.save(dest/'post_prune_original_ids.npy',ids)
    for group_id,local in enumerate(scheduled):
        base_rows=rows.copy();best=dict(rows=rows,metrics=current,accepted=False)
        base_opacity=model._opacity.detach().clone()
        folder=dest/f'mixed_group_{group_id+1:02d}';folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/'mixed_discovery.npz',original_ids=ids[local],**{k:v[local] for k,v in e.items()
            if isinstance(v,np.ndarray) and len(v)==len(ids)})
        def checkpoint(step,subset,parameters):
            patch=base_rows[subset].copy()
            for axis in range(3):patch[f'scale_{axis}']=parameters['scale'][:,axis]
            for axis in range(4):patch[f'rot_{axis}']=parameters['rotation'][:,axis]
            patch['opacity']=parameters['opacity'][:,0]
            path=folder/f'refine_{step:03d}_parameters.npz'
            np.savez(path,original_ids=ids[subset],old=base_rows[subset],new=patch)
            write_json(path.with_suffix('.json'),dict(parameters=str(path.resolve()),
                retained_ids=str((dest/'post_prune_original_ids.npy').resolve()),
                original_model=scene.spec['model'],
                previous_accepted_states=[str(p.resolve()) for p in sorted(dest.parent.rglob('accepted_parameter_changes.npz'))],
                materialization='Apply accepted states then this patch to original rows, retain listed original IDs; full native fields preserved'))
            # The optimizer materializes scale/rotation at the checkpoint.
            # Preserve exact raw opacity of all unselected rows as PLY-based
            # candidates do; avoid a sigmoid/logit round-trip outside subset.
            index=torch.as_tensor(subset,device='cuda')
            with torch.no_grad():
                model._opacity=base_opacity.index_copy(0,index,torch.as_tensor(parameters['opacity'],device='cuda'))
            metrics=evaluator.evaluate(model,folder/f'refine_{step:03d}')
            ok,gain,local_gain,reasons=choose(metrics,best['metrics'],initial,config)
            item=dict(round=iteration+1,action='local_covariance_refine',group=group_id+1,step=step,
                candidates=len(subset),changed_ids=ids[subset].tolist(),accepted=ok,relative_gain=gain,
                local_view_gain=local_gain,reasons=reasons,metrics=str((folder/f'refine_{step:03d}/metrics.json').resolve()),
                parameters=str(path.resolve()),predicted_fg=float(e['fg'][subset].sum()),
                predicted_bg=float((e['bg'][subset]+e.get('free_bg',np.zeros(len(ids)))[subset]).sum()))
            edits.append(item)
            write_json(folder/f'refine_{step:03d}'/'decision.json',item)
            if ok:
                changed=base_rows.copy();changed[subset]=patch
                best.update(rows=changed,metrics=metrics,accepted=True)
            return dict(accepted=ok,relative_gain=gain,reasons=reasons)
        auxiliary=free_objective(scene,model) if config.get('free_background_support')=='projected_surface' else None
        optimization_indices=None;observed_loss=None
        if config.get('focused_refinement',False):
            optimization_indices=risk_views(scene,evaluator,ids[local],edits)
            observed_loss=quality_focus(scene,lambda:best['metrics'],initial)
            write_json(folder/'optimization_views.json',dict(indices=optimization_indices,
                views=[scene.views[i]['name'] for i in optimization_indices],
                selection='Automatically observed rejection cameras within native group footprints + diverse affected cameras'))
        history=refine(scene,model,local,scene.evidence_indices(),config,checkpoint,auxiliary_loss=auxiliary,
            optimization_indices=optimization_indices,observed_view_loss=observed_loss)
        write_json(folder/'refinement_history.json',history)
        if best['accepted']:
            rows,current=best['rows'],best['metrics'];accepted_any=True
            changed=np.logical_or.reduce([rows[k][local]!=base_rows[k][local]
                for k in ['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]])
            changed_ids.update(ids[local][changed].tolist())
            # Original footprint visibility no longer certifies unchanged views
            # after covariance changes. Subsequent deletions use all views.
            evaluator.force_full=True
            np.savez(folder/'accepted_parameter_changes.npz',original_ids=ids[local],old=base_rows[local],new=rows[local])
            index=torch.as_tensor(local,device='cuda')
            with torch.no_grad():
                model._scaling=model._scaling.detach().index_copy(0,index,torch.as_tensor(np.column_stack([rows[f'scale_{i}'][local] for i in range(3)]),device='cuda'))
                model._rotation=model._rotation.detach().index_copy(0,index,torch.as_tensor(np.column_stack([rows[f'rot_{i}'][local] for i in range(4)]),device='cuda'))
                model._opacity=model._opacity.detach().index_copy(0,index,torch.as_tensor(rows['opacity'][local,None].copy(),device='cuda'))
    return model,rows,current,accepted_any,changed_ids


def risk_views(scene,evaluator,original_ids,edits):
    from collections import Counter
    affected=evaluator.affected_views(original_ids)
    by_name={v['name']:i for i,v in enumerate(scene.views)}
    counts=Counter();recent=[]
    for item in reversed(edits):
        if item['accepted']:continue
        for reason in item['reasons']:
            name=reason.split(':',1)[0]
            if name in by_name and by_name[name] in affected:
                index=by_name[name]
                counts[index]+=len(reason.split(':',1)[1].split(',')) if ':' in reason else 1
                if index not in recent:recent.append(index)
    selected=sorted(recent,key=lambda i:-counts[i])[:8]
    diverse=[i for i in scene.evidence_indices() if i in affected and i not in selected]
    if not diverse:diverse=[i for i in affected if i not in selected]
    if diverse:
        count=min(16-len(selected),len(diverse))
        selected += [diverse[i] for i in np.linspace(0,len(diverse)-1,count,dtype=int)]
    return selected or scene.evidence_indices()[:16]


def quality_focus(scene,current_metrics,initial):
    """Match optimization to actual sparse dark-foreground/depth constraints."""
    import torch
    cache={};q=scene.config['quality'];baseline={v['name']:v for v in initial['views']}
    def loss(vi,out,target,mask,r):
        if vi not in cache:
            dark=mask & (target @ np.array([.2126,.7152,.0722])<.12)
            cache[vi]=(torch.as_tensor(dark,device='cuda'),torch.as_tensor(target,device='cuda'))
        dark,target_t=cache[vi]
        value=out['render'].sum()*0
        name=scene.views[vi]['name'];current=next(v for v in current_metrics()['views'] if v['name']==name)
        old=baseline[name]
        if int(dark.sum())>20:
            threshold=min(current['dark_foreground_mse'],old['dark_foreground_mse'])*(1+q['view_foreground_mse_relative_increase'])+q['foreground_mse_absolute_increase']
            mse=((out['render'].permute(1,2,0)[dark]-target_t[dark])**2).mean()
            value=value+60*torch.relu(mse-threshold)
        # A changed depth tail can fail a per-view fraction check while the
        # old mean L1 depth objective barely changes. Penalize this tail.
        ref=scene.get_view(vi)[3]
        if ref is not None:
            confident=scene.get_view(vi)[4]
            valid=mask & np.isfinite(ref) & (ref>0) & np.isfinite(confident)
            if valid.any():
                valid &= confident>=np.quantile(confident[valid],.2)
                vg=torch.as_tensor(valid,device='cuda');depth=torch.as_tensor(ref,device='cuda')
                unit=depth[vg].median().clamp_min(1e-8)
                error=(out['median_depth'].squeeze()[vg]-depth[vg]).abs()/unit
                value=value+2*torch.relu(error-q['depth_relative_p95']).clamp(max=.1).mean()
        return value
    return loss


def free_objective(scene,model):
    """Local covariance loss on generated cameras; effective alpha stays fixed."""
    import torch
    from ..runtime import camera
    from ..geometry_consistency import external_surface
    surface=external_surface(scene,scene.evidence_indices())
    supports=free_support(scene,surface);targets=[]
    for i,s in enumerate(supports):
        v=s['view'];c=camera(v['K'],v['E'],v['width'],v['height'],v['name'],i)
        with torch.no_grad():reference=scene.render(c,model)['mask'].detach().squeeze()
        # Eroded projected support avoids penalizing the background margin
        # introduced by dilation. This is a soft optimization term only.
        from scipy import ndimage
        inside=torch.as_tensor(ndimage.binary_erosion(s['surface_mask'],iterations=3),device='cuda')
        outside=torch.as_tensor(s['outside'],device='cuda')
        targets.append((c,reference,inside,outside,s['normalizer']))
    def loss(current_scene,current_model,step):
        if not targets:return current_model._scaling.sum()*0
        c,reference,inside,outside,normalizer=targets[step%len(targets)]
        alpha=current_scene.render(c,current_model)['mask'].squeeze()
        background=alpha[outside].sum()/normalizer
        coverage=torch.relu(reference[inside]-alpha[inside]).square().mean() if inside.any() else background*0
        return background+30*coverage
    return loss
