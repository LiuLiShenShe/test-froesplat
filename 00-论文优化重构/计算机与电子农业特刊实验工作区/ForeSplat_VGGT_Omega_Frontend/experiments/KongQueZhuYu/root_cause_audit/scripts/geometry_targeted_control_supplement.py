"""Actual input-invariant checks and same-pair effects for targeted controls."""
import argparse
from scipy.ndimage import binary_erosion
from geometry_targeted_control_eval import condition_new,load_frame,depth_rgb,sampson,fundamental
from geometry_multiview_audit import ROIS,prepare_queries,pair_metrics
from geometry_common import *

def paired_effects(root):
    output=root/'evaluation';results=[json.loads(f.read_text()) for f in sorted(output.glob('condition_*_results.json'))];rows=[]
    models=['Omega','Standard']
    for model in models:
        names=[r['condition'] for r in results if r['geometry'][0]['model']==model and not r['geometry'][0]['historical']]
        if len(names)<2:continue
        if 'background' in root.name:baseline=model+'_RAP_baseline_24';target=model+'_background_restored_24'
        else:baseline=model+'_70';target=model+'_110'
        a=next(r for r in results if r['condition']==baseline);b=next(r for r in results if r['condition']==target)
        def image_lookup(r):return {(x['source_id'],x['target_id']):x for x in r['image_matches'] if x['scope']=='all_matched' and x['accepted']}
        aa=image_lookup(a);bb=image_lookup(b)
        for gap in sorted({x['common_rank_offset'] for x in a['relative_pose']}):
            shared=[key for key in aa if key in bb and aa[key]['common_rank_offset']==gap]
            for field in ['sampson_p50_px','source_depth_to_RGB_error_p50_px','valid_fraction']:
                eligible=[k for k in shared if aa[k].get(field) is not None and bb[k].get(field) is not None];d=np.asarray([bb[k][field]-aa[k][field] for k in eligible]);better=d>0 if field=='valid_fraction' else d<0
                rows.append(dict(model=model,baseline=baseline,target=target,common_rank_offset=gap,metric=field,same_valid_image_pairs=len(eligible),target_minus_baseline_p10=q(d,.1),target_minus_baseline_p50=q(d,.5),target_minus_baseline_p90=q(d,.9),improved_pairs=int(np.sum(better)),unchanged_pairs=int(np.sum(d==0))))
            def geo_lookup(r):
                z={}
                for x in r['geometry']:
                    if x['control']!='mask_finite' or x['common_rank_offset']!=gap:continue
                    key=tuple(sorted([x['source_id'],x['target_id']]));z.setdefault(key,[]).append(x)
                return {k:sum(v['supported_5pct'] for v in xx)/sum(v['queries'] for v in xx) for k,xx in z.items()}
            ga=geo_lookup(a);gb=geo_lookup(b);keys=sorted(set(ga)&set(gb));d=np.array([gb[k]-ga[k] for k in keys])
            rows.append(dict(model=model,baseline=baseline,target=target,common_rank_offset=gap,metric='bidirectional_all_query_depth_support5pct',same_valid_image_pairs=len(keys),target_minus_baseline_p10=q(d,.1),target_minus_baseline_p50=q(d,.5),target_minus_baseline_p90=q(d,.9),improved_pairs=int(np.sum(d>0)),unchanged_pairs=int(np.sum(d==0))))
    write_csv(output/'same_pair_condition_effects.csv',rows);return rows

def invariant_check(root,background):
    checks={}
    for model in ['Omega','Standard']:
        names=[model+'_RAP_baseline_24',model+'_background_restored_24'] if background else [model+'_70',model+'_110']
        a=condition_new(root/names[0]);b=condition_new(root/names[1]);ids=sorted(set(a['frames'])&set(b['frames']));ma=a['metadata'];mb=b['metadata']
        fields=['checkpoint_sha256','seed','torch_version','python','preprocessing','source_file_sha256'];metadata_equal={k:ma.get(k)==mb.get(k) for k in fields}
        frame_equal={key:all(a['frames'][sid]['camera_record'].get(key)==b['frames'][sid]['camera_record'].get(key) for sid in ids) for key in ['mask_sha256_model','pixel_transform','pixel_transform_pixel_center','model_width','model_height','original_width','original_height']}
        input_equal=sum(a['frames'][sid]['camera_record'].get('input_float32_sha256')==b['frames'][sid]['camera_record'].get('input_float32_sha256') for sid in ids)
        checks[model]=dict(conditions=names,same_source_ids=ids,common_frames=len(ids),metadata_invariants_equal=metadata_equal,frame_mask_geometry_invariants_equal=frame_equal,
            identical_model_RGB_float32_sha_frames=input_equal,expected_RGB_identical=not background,
            all_required_invariants=all(metadata_equal.values()) and all(frame_equal.values()) and (background or input_equal==len(ids)),
            historical208_note='No old complete environment freeze; excluded from strict70vs110 input/env invariant claim',
            background_note='Only original Mask-exterior pixels restored; prepared_sources.json is producer exact pixel assertion. Model RGB can differ at interpolated foreground boundaries and is the intended condition change.' if background else None)
    write_json(root/'evaluation/actual_input_invariant_crosscheck.json',checks);assert all(c['all_required_invariants'] for c in checks.values());return checks

def background_roi(root):
    output=root/'evaluation';sources=camera_sources();ref=sources['Omega_depth'];refs={source_id(n):ref['a_actual'][i] for i,n in enumerate(ref['names'])};conditions=[condition_new(root/n) for n in ['Omega_RAP_baseline_24','Omega_background_restored_24','Standard_RAP_baseline_24','Standard_background_restored_24']]
    with (output/'fixed_pair_manifest.csv').open() as f:pairs=list(csv.DictReader(f))
    queries={};rows=[];image_rows=[];features={}
    def feature(sid):
        if sid not in features:
            with np.load(AUDIT/'camera_projection_comparison/sift_features'/f'crop_{sid:04d}.npz') as f:features[sid]=(f['points'],f['edge'])
        return features[sid]
    for pair in pairs:
        sid=int(pair['source_id']);tid=int(pair['target_id'])
        for source_id_ in [sid,tid]:
            if source_id_ in queries:continue
            mask=np.load(OMEGA_DATA/'per_frame'/f'model_mask_crop_{source_id_:04d}.npy').squeeze()>0;y,x=np.indices(mask.shape);orig=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(refs[source_id_]).T
            selections={'foreground':mask,'foreground_boundary_common_grid':mask&~binary_erosion(mask)}
            for region,(left,top,right,bottom) in ROIS.items():selections[region]=mask&(orig[...,0]>=left)&(orig[...,0]<right)&(orig[...,1]>=top)&(orig[...,1]<bottom)
            queries[source_id_]={}
            for enum,(region,selection) in enumerate(selections.items()):
                point=orig[selection,:2]
                if len(point)>1500:point=point[np.random.default_rng(source_id_*101+enum+31).choice(len(point),1500,replace=False)]
                queries[source_id_][region]=point
        with np.load(output/'fixed_foreground_matches'/f'{sid:04d}_{tid:04d}.npz') as f:p1=f['source_points'];p2=f['target_points'];accepted=bool(f['accepted'])
        original=np.c_[p1,np.ones(len(p1))]@np.linalg.inv(refs[sid]).T
        regions={'all_foreground_inliers':np.ones(len(p1),bool)}
        if accepted:
            points1,edge1=feature(sid);points2,edge2=feature(tid);i1=np.argmin(np.sum((p1[:,None]-points1[None])**2,axis=2),axis=1);i2=np.argmin(np.sum((p2[:,None]-points2[None])**2,axis=2),axis=1);boundary=edge1[i1]|edge2[i2]
            regions['boundary_le4_shared_px']=boundary;regions['interior_gt4_shared_px']=~boundary
        for region,(left,top,right,bottom) in ROIS.items():regions[region]=(original[:,0]>=left)&(original[:,0]<right)&(original[:,1]>=top)&(original[:,1]<bottom)
        for condition in conditions:
            for a,b in [(sid,tid),(tid,sid)]:
                source=load_frame(condition,a);target=load_frame(condition,b)
                for region,points in queries[a].items():
                    query=prepare_queries(source,points)
                    with np.errstate(invalid='ignore'):m=pair_metrics(query,source,target,'mask_finite')
                    rows.append(dict(condition=condition['name'],model=condition['model'],source_id=a,target_id=b,region=region,**m))
            source=load_frame(condition,sid);target=load_frame(condition,tid);k1=refs[sid]@np.linalg.inv(source['a'])@source['k'];k2=refs[tid]@np.linalg.inv(target['a'])@target['k'];f=fundamental(k1,k2,source['w2c'],target['w2c'])
            for region,keep in regions.items():
                if keep.sum()<3:continue
                err=sampson(f,p1[keep],p2[keep]);record=dict(condition=condition['name'],model=condition['model'],source_id=sid,target_id=tid,region=region,inlier_matches=int(keep.sum()),sampson_p50_px=q(err,.5),sampson_p90_px=q(err,.9))
                d=depth_rgb(source,target,p1[keep],p2[keep],refs[sid],refs[tid],condition['model'])[0];image_rows.append(dict(record,**d))
    write_csv(output/'ROI_cross_depth_pairs.csv',rows);write_csv(output/'ROI_independent_RGB_pairs.csv',image_rows);summary=[]
    for condition in conditions:
        for region in queries[int(pairs[0]['source_id'])]:
            rr=[r for r in rows if r['condition']==condition['name'] and r['region']==region];den=sum(r['queries'] for r in rr);ir=[r for r in image_rows if r['condition']==condition['name'] and r['region']==region]
            summary.append(dict(condition=condition['name'],region=region,directed_depth_pairs=len(rr),pairs_with_nonempty_reference=sum(r['queries']>0 for r in rr),reference_queries=den,depth_support5pct_all_queries=sum(r['supported_5pct'] for r in rr)/den if den else None,
                image_pairs_with_ge3_matches=len(ir),matched_image_sampson_pair_p50_px=q([r['sampson_p50_px'] for r in ir],.5),matched_image_depth_error_pair_p50_px=q([r.get('source_depth_to_RGB_error_p50_px') for r in ir],.5)))
        for region in ['all_foreground_inliers','boundary_le4_shared_px','interior_gt4_shared_px']:
            ir=[r for r in image_rows if r['condition']==condition['name'] and r['region']==region]
            summary.append(dict(condition=condition['name'],region=region,image_pairs_with_ge3_matches=len(ir),matched_image_sampson_pair_p50_px=q([r['sampson_p50_px'] for r in ir],.5),matched_image_depth_error_pair_p50_px=q([r.get('source_depth_to_RGB_error_p50_px') for r in ir],.5)))
    write_csv(output/'ROI_summary.csv',summary);write_json(output/'ROI_protocol.json',dict(rois_original=ROIS,source_ids=sorted(queries),foreground_reference='Common old Omega foreground Mask mapped original coordinates, same seeded1500query/ROI for allconditions',
        leaf_windows='Fixed source0 image windows, not tracked physical leaf instances. Empty foreground ROI pairs explicit; same ROI-selected original SIFT inliers reused, min3 per pair. Limited to24source/12pairs.',
        boundary='Depth boundary1pixel on common384688 innerMask. RGB boundary<=4shared pixels uses saved SIFT cache edge flag on either endpoint; includespot/blue, not pure leafGT',
        matched_images='Exactly prior fixed foreground-only RANSAC inliers, subset only by common coordinates or cache edge flag; no feature extraction percondition',no_gpu=True,no_new_predictions=True))
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['count','background'],required=True);args=parser.parse_args();background=args.scope=='background';root=AUDIT/'controlled_fixes'/('background_control_v1' if background else 'view_count_control_v1')
    invariant_check(root,background);effects=paired_effects(root);roi=background_roi(root) if background else None
    write_json(root/'evaluation/supplement_status.json',dict(status='complete',same_pair_effect_rows=len(effects),roi_summary_rows=len(roi) if roi is not None else 0,no_gpu=True,no_model_inference=True))
    print(json.dumps(effects,indent=2),flush=True)

if __name__=='__main__':main()
