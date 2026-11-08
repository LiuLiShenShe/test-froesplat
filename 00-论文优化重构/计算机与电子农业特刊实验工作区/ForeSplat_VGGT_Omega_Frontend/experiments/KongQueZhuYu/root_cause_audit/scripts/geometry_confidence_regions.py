"""Saved-depth-only confidence-tail diagnostic; no model execution or source edits."""
import io
from PIL import Image
from geometry_common import *
from geometry_multiview_audit import world_from_depth,prepare_queries,pair_metrics,summarize

def main():
    sources=camera_sources();protocol=json.loads((AUDIT/'geometry_comparison/input_audit_protocol.json').read_text())
    ids=protocol['actual_audited_ids'];offsets=[1,5,15]
    with np.load('/tmp/foresplat_root_cause_geometry_input/standard_model_inputs.npz') as cache:
        standard_masks=cache['masks'];standard_names=list(cache['names'])
    all_rows=[];tail_rows=[];frames={};byid={m:{source_id(n):i for i,n in enumerate(s['names'])} for m,s in sources.items() if m in ['Omega_depth','Standard_depth']}
    def load(method,sid):
        key=(method,sid)
        if key in frames:return frames[key]
        s=sources[method];i=byid[method][sid];n=s['names'][i];stem=Path(n).stem
        if method=='Omega_depth':
            depth=np.load(OMEGA_DATA/'per_frame'/f'depth_{stem}.npy').squeeze();conf=np.load(OMEGA_DATA/'per_frame'/f'confidence_{stem}.npy').squeeze()
            mask=np.load(OMEGA_DATA/'per_frame'/f'model_mask_{stem}.npy').squeeze()>0
        else:
            depth=np.load(STANDARD/'Standard_VGGT_Frontend/depth'/f'{stem}.npy').squeeze();conf=np.load(STANDARD/'Standard_VGGT_Frontend/confidence'/f'{stem}.npy').squeeze()
            assert standard_names[i]==n;mask=standard_masks[i]
        frame=dict(name=n,source_id=sid,index=i,mask=mask,depth=depth,conf=conf,xyz=world_from_depth(depth,s['c2w'][i],s['k_model'][i]),w2c=s['w2c'][i],k=s['k_model'][i],a=s['a_actual'][i])
        frames[key]=frame;return frame
    for sid in ids:
        omega=load('Omega_depth',sid);y,x=np.nonzero(omega['mask']);original=np.column_stack((x,y,np.ones(len(x))))@np.linalg.inv(omega['a']).T
        if len(original)>6000:original=original[np.random.default_rng(101+sid).choice(len(original),6000,replace=False)]
        for method in ['Omega_depth','Standard_depth']:
            source=load(method,sid);query=prepare_queries(source,original[:,:2]);selection=source['mask']&np.isfinite(source['conf']);c=source['conf'][selection];lo=q(c,.25);hi=q(c,.75)
            tail_rows.append(dict(method=method,source=source['name'],source_id=sid,confidence_q25=lo,confidence_q75=hi,
                low_tail_foreground_fraction=float(np.mean(c<=lo)),high_tail_foreground_fraction=float(np.mean(c>=hi)),q25_tie_fraction=float(np.mean(c==lo)),q75_tie_fraction=float(np.mean(c==hi)),
                finite_positive_foreground_fraction=float(np.mean(np.isfinite(source['depth'][selection])&(source['depth'][selection]>0)))))
            for label,selected in [('foreground_common_queries',np.ones(len(query['conf']),bool)),('source_confidence_low_q25',query['conf']<=lo),('source_confidence_high_q75',query['conf']>=hi)]:
                subset={k:v[selected] for k,v in query.items()}
                for off in offsets:
                    for sign in [-1,1]:
                        target_id=sid+sign*off
                        if target_id not in byid[method]:continue
                        target=load(method,target_id)
                        all_rows.append(dict(method=method,source=source['name'],target=target['name'],source_offset=off*sign,region=label,control='mask_finite_target_no_quantile_filter',**pair_metrics(subset,source,target,'mask_finite')))
        print('Confidence-region saved-depth audit',sid,flush=True)
    write_csv(AUDIT/'geometry_comparison/confidence_region_pairs.csv',all_rows)
    write_csv(AUDIT/'geometry_comparison/confidence_region_summary.csv',summarize(all_rows))
    write_csv(AUDIT/'geometry_comparison/confidence_region_thresholds.csv',tail_rows)
    write_json(AUDIT/'geometry_comparison/confidence_region_protocol.json',dict(source_ids=ids,original_source_offsets=offsets,
        query_reference='At most6000 seeded actual Omega foreground pixels per source, mapped to original coordinates; same original queries before model/tail eligibility',
        tail_selection='Each model/source native foreground confidence q25 and q75; ties retained and fractions explicitly reported. Selection differs by model; use low/high comparison within a model, never infer cross-model probability calibration.',
        target_filter='Mask,finite positive depth only; no target confidence-tail threshold, isolating source confidence selection',
        purpose='Do low-confidence source pixels predict weaker internal multiview support in fixed18views? Not absolute GT geometry accuracy.',
        scope='Supplementary18 source views with real saved target views at source-ID offsets±1/5/15; not a full208 confidence-tail evaluation',
        no_model_inference=True,no_gpu=True,no_new_fusion=True))
    print('Confidence-region audit complete',len(all_rows),flush=True)

if __name__=='__main__':main()
