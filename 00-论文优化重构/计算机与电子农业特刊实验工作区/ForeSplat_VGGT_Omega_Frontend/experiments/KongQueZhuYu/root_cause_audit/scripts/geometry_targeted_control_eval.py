"""CPU evaluation: fixed source IDs, foreground feature matches, and pixel queries.

View-count controls compare70/110 within the same current environment.208 is
a historical reference. Background controls share24sources and12near pairs.
No inference, GPU, retraining, changed source images, or historical writes.
"""
import argparse,io
import cv2
from PIL import Image
from geometry_common import *
from geometry_multiview_audit import world_from_depth,prepare_queries,pair_metrics
from camera_epipolar_audit import fundamental,sampson

def actual_matrix(raw):
    a=np.asarray(raw['pixel_transform_pixel_center'],float) if 'pixel_transform_pixel_center' in raw else np.asarray(raw['pixel_transform'],float)
    return a

def condition_new(path):
    meta=json.loads((path/'metadata.json').read_text());raw=json.loads((path/'cameras.json').read_text())['frames'];frames={}
    for r in raw:
        e=np.eye(4);t=np.array(r['extrinsic_model_w2c']);e[:t.shape[0]]=t
        frames[int(r['source_index'])]=dict(name=r['image_name'],w2c=e,c2w=np.linalg.inv(e),k=np.asarray(r['intrinsic_model']),a=actual_matrix(r),
            depth_file=path/'per_frame'/f'depth_{Path(r["image_name"]).stem}.npy',conf_file=path/'per_frame'/f'confidence_{Path(r["image_name"]).stem}.npy',mask_file=path/'per_frame'/f'model_mask_{Path(r["image_name"]).stem}.npy',camera_record=r)
    return dict(name=path.name,model=meta['model'],input_views=len(raw),frames=frames,metadata=meta,path=path,historical=False)

def historical_standard_mask(path,expected_sha256):
    """Replay only the persistent original mask, never loadRGB or temporary input cache."""
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected_sha256,'Historical Standard mask source changed'
    mask=Image.open(io.BytesIO(raw)).convert('L');w,h=mask.size;dimension=max(w,h)
    square=Image.new('L',(dimension,dimension),0);square.paste(mask,((dimension-w)//2,(dimension-h)//2))
    return np.asarray(square.resize((518,518),Image.Resampling.NEAREST))>=127

def condition_historical(model,sources):
    method='Omega_depth' if model=='Omega' else 'Standard_depth';s=sources[method];frames={}
    if model=='Standard':
        manifest={r['image_name']:r for r in json.loads((STANDARD/'Standard_VGGT_Frontend/metadata/input_manifest.json').read_text())['frames']}
    for i,n in enumerate(s['names']):
        stem=Path(n).stem;frame=dict(name=n,w2c=s['w2c'][i],c2w=s['c2w'][i],k=s['k_model'][i],a=s['a_actual'][i])
        if model=='Omega':frame.update(depth_file=OMEGA_DATA/'per_frame'/f'depth_{stem}.npy',conf_file=OMEGA_DATA/'per_frame'/f'confidence_{stem}.npy',mask_file=OMEGA_DATA/'per_frame'/f'model_mask_{stem}.npy')
        else:frame.update(depth_file=STANDARD/'Standard_VGGT_Frontend/depth'/f'{stem}.npy',conf_file=STANDARD/'Standard_VGGT_Frontend/confidence'/f'{stem}.npy',historical_mask_source=STANDARD/'Standard_VGGT_Dataset/masks'/n,historical_mask_sha256=manifest[n]['mask_sha256'])
        frames[source_id(n)]=frame
    return dict(name=model+'_208_historical',model=model,input_views=208,frames=frames,metadata={},path=Path(s['path']),historical=True)

def load_frame(condition,sid):
    frame=condition['frames'][sid]
    if 'depth' not in frame:
        frame['depth']=np.load(frame['depth_file']).squeeze();frame['conf']=np.load(frame['conf_file']).squeeze()
        if 'mask' not in frame:
            frame['mask']=historical_standard_mask(frame['historical_mask_source'],frame['historical_mask_sha256']) if 'historical_mask_source' in frame else np.load(frame['mask_file']).squeeze()>0
        frame['xyz']=world_from_depth(frame['depth'],frame['c2w'],frame['k'])
    return frame

def cached_matches(pairs,output):
    cv2.setNumThreads(1);directory=output/'fixed_foreground_matches';directory.mkdir(exist_ok=True);features={};bf=cv2.BFMatcher(cv2.NORM_L2);result=[]
    def feature(sid):
        if sid not in features:
            path=AUDIT/'camera_projection_comparison/sift_features'/f'crop_{sid:04d}.npz'
            assert path.is_file(),'Required old RAP foreground SIFT cache missing: '+str(path)
            with np.load(path) as f:features[sid]={k:f[k] for k in ['points','descriptors','edge']}
        return features[sid]
    for index,(sid,tid,gap) in enumerate(pairs):
        path=directory/f'{sid:04d}_{tid:04d}.npz'
        if path.exists():
            with np.load(path) as f:q1=f['source_points'];q2=f['target_points'];fimg=f['image_f'];mutual=int(f['mutual']);accepted=bool(f['accepted'])
        else:
            a=feature(sid);b=feature(tid);d1,d2=a['descriptors'],b['descriptors'];good=[]
            if min(len(d1),len(d2))>=2:
                forward=bf.knnMatch(d1,d2,k=2);reverse=bf.knnMatch(d2,d1,k=2)
                back={m.queryIdx:m.trainIdx for pair in reverse if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance}
                good=[m for pair in forward if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance and back.get(m.trainIdx)==m.queryIdx]
            p1=np.asarray([a['points'][m.queryIdx] for m in good],np.float32).reshape(-1,2);p2=np.asarray([b['points'][m.trainIdx] for m in good],np.float32).reshape(-1,2);mutual=len(good)
            cv2.setRNGSeed(1107+sid*211+tid)
            fimg,inliers=cv2.findFundamentalMat(p1,p2,cv2.FM_RANSAC,1.5,.999) if mutual>=12 else (None,None)
            accepted=fimg is not None and fimg.shape==(3,3) and inliers is not None and int(inliers.sum())>=12
            if accepted:keep=inliers.ravel().astype(bool);q1=p1[keep];q2=p2[keep]
            else:q1=np.empty((0,2),np.float32);q2=q1.copy();fimg=np.full((3,3),np.nan)
            np.savez(path,source_points=q1,target_points=q2,image_f=fimg,mutual=mutual,accepted=accepted)
        result.append(dict(source_id=sid,target_id=tid,common_rank_offset=gap,original_source_offset=tid-sid,mutual_matches=mutual,image_F_inliers=len(q1),accepted=accepted,p1=q1,p2=q2,image_f=fimg))
    write_csv(output/'fixed_pair_manifest.csv',[{k:v for k,v in r.items() if k not in ['p1','p2','image_f']} for r in result]);return result

def depth_rgb(frame,target,p1,p2,aref1,aref2,model):
    original=np.c_[p1,np.ones(len(p1))]@np.linalg.inv(aref1).T;pixel=(original@frame['a'].T)[:,:2]
    z=cv2.remap(frame['depth'],pixel[:,0].astype(np.float32)[None],pixel[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[0]
    conf=cv2.remap(frame['conf'],pixel[:,0].astype(np.float32)[None],pixel[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[0]
    mask=cv2.remap(frame['mask'].astype(np.uint8),pixel[:,0].astype(np.float32)[None],pixel[:,1].astype(np.float32)[None],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[0]>0
    rays=np.c_[pixel,np.ones(len(pixel))]@np.linalg.inv(frame['k']).T;local=rays*z[:,None];world=local@frame['c2w'][:3,:3].T+frame['c2w'][:3,3];dest=world@target['w2c'][:3,:3].T+target['w2c'][:3,3]
    kd=aref2@np.linalg.inv(target['a'])@target['k'];hom=dest@kd.T;uv=hom[:,:2]/hom[:,2,None];valid=mask&np.isfinite(uv).all(-1)&np.isfinite(z)&(z>0)&(dest[:,2]>0);error=np.linalg.norm(uv-p2,axis=1)
    threshold=1.252699375152588 if model=='Omega' else 1.0264525413513184
    rows=[]
    for label,keep in [('all_matched',valid),('fixed_historical_conf_p20',valid&(conf>=threshold))]:
        rows.append(dict(scope=label,matched_points=len(p1),valid_points=int(keep.sum()),valid_fraction=float(np.mean(keep)),source_depth_to_RGB_error_p50_px=q(error[keep],.5),source_depth_to_RGB_error_p90_px=q(error[keep],.9),within3px_fraction_all_matches=float(np.sum(keep&(error<=3))/max(len(p1),1)),confidence_threshold=threshold if label!='all_matched' else None))
    return rows

def pose_comparison(condition,reference,ids):
    a=np.array([condition['frames'][sid]['c2w'] for sid in ids]);b=np.array([reference['frames'][sid]['c2w'] for sid in ids]);x=a[:,:3,3];y=b[:,:3,3];extent=float(np.ptp(np.quantile(y,[.025,.975],axis=0),axis=0).max());held=np.arange(len(ids))%4==0;fit=~held
    scale,r,t=fit_similarity(x[fit],y[fit]);res=np.linalg.norm(scale*x@r.T+t-y,axis=1)/max(extent,1e-15);orient=np.einsum('ij,njk->nik',r,a[:,:3,:3]);angle=camera_angles(np.einsum('nij,njk->nik',b[:,:3,:3].transpose(0,2,1),orient))
    singular=np.linalg.svd(x[fit]-x[fit].mean(0),compute_uv=False);ratio=float(singular[1]/max(singular[0],1e-15));reliable=scale>0 and ratio>.05 and q(res[held],.5)<.1 and q(res[held],.9)<.25 and q(angle[held],.5)<15 and q(angle[held],.9)<30
    return dict(condition=condition['name'],reference=reference['name'],same_source_ids=len(ids),held_out=int(held.sum()),fit_frames=int(fit.sum()),scale=scale,rotation=r,translation=t,positive_scale=scale>0,SO3_determinant=float(np.linalg.det(r)),camera_path_singular_ratio=ratio,held_out_center_residual_p50_normalized=q(res[held],.5),held_out_center_residual_p90_normalized=q(res[held],.9),held_out_orientation_error_p50_deg=q(angle[held],.5),held_out_orientation_error_p90_deg=q(angle[held],.9),reliable_global_similarity=bool(reliable),scope='Post-fit camera-center diagnostic only; no subtraction of unaligned centers, no prediction transformation, no GT pose claim')

def evaluate(condition,pairs,refs,queries,output):
    save=output/f'condition_{condition["name"]}_results.json'
    if save.exists():return json.loads(save.read_text())
    geometry=[];image=[];rotation=[]
    for pair in pairs:
        sid=pair['source_id'];tid=pair['target_id'];base=dict(condition=condition['name'],model=condition['model'],input_views=condition['input_views'],historical=condition['historical'],source_id=sid,target_id=tid,common_rank_offset=pair['common_rank_offset'],original_source_offset=tid-sid)
        source=load_frame(condition,sid);target=load_frame(condition,tid);relative=target['w2c']@source['c2w'];angle=float(camera_angles(relative[:3,:3]));rotation.append(dict(base,relative_rotation_deg=angle,relative_rotation=relative[:3,:3],translation_direction=relative[:3,3]/max(np.linalg.norm(relative[:3,3]),1e-15)))
        for a,b in [(sid,tid),(tid,sid)]:
            frame=load_frame(condition,a);dest=load_frame(condition,b);query=prepare_queries(frame,queries[a])
            for control in ['mask_finite','confidence_gt1']:
                with np.errstate(invalid='ignore',over='ignore'):
                    measurements=pair_metrics(query,frame,dest,control)
                geometry.append(dict(base,source_id=a,target_id=b,original_source_offset=b-a,control=control,**measurements))
        row=dict(base,accepted=pair['accepted'],image_F_inliers=pair['image_F_inliers'])
        if pair['accepted']:
            k1=refs[sid]@np.linalg.inv(source['a'])@source['k'];k2=refs[tid]@np.linalg.inv(target['a'])@target['k'];f=fundamental(k1,k2,source['w2c'],target['w2c']);err=sampson(f,pair['p1'],pair['p2'])
            row.update(sampson_p50_px=q(err,.5),sampson_p90_px=q(err,.9),sampson_within1_5px_fraction=float(np.mean(err<=1.5)),image_only_F_sampson_p50_px=q(sampson(pair['image_f'],pair['p1'],pair['p2']),.5))
            for scope in depth_rgb(source,target,pair['p1'],pair['p2'],refs[sid],refs[tid],condition['model']):image.append(dict(row,**scope))
        else:image.append(dict(row,scope='all_matched',matched_points=0,valid_points=0))
    result=dict(condition=condition['name'],geometry=geometry,image_matches=image,relative_pose=rotation);write_json(save,result)
    print('Condition complete',condition['name'],'accepted imagepairs',sum(r['accepted'] for r in pairs),'/',len(pairs),flush=True);return result

def summarize(results):
    summary=[]
    for result in results:
        for gap in sorted({r['common_rank_offset'] for r in result['relative_pose']}):
            for control in ['mask_finite','confidence_gt1']:
                geo=[r for r in result['geometry'] if r['common_rank_offset']==gap and r['control']==control];images=[r for r in result['image_matches'] if r['common_rank_offset']==gap and r['scope']=='all_matched'];accepted=[r for r in images if r['accepted']];den=sum(r['queries'] for r in geo);rotation=[r['relative_rotation_deg'] for r in result['relative_pose'] if r['common_rank_offset']==gap]
                row=dict(condition=result['condition'],model=geo[0]['model'],input_views=geo[0]['input_views'],historical=geo[0]['historical'],common_rank_offset=gap,control=control,unordered_image_pairs=len(images),accepted_shared_image_pairs=len(accepted),bidirectional_depth_pairs=len(geo),depth_query_count=den,
                    depth_support_5pct_all_queries=sum(r['supported_5pct'] for r in geo)/max(den,1),depth_overlap_all_queries=sum(r['target_overlap'] for r in geo)/max(den,1),depth_predicted_occluded_fraction=sum(r['target_predicted_occluded'] for r in geo)/max(den,1),
                    raw_depth_relative_error_pair_p50=q([r.get('raw_overlap_relative_depth_error_p50') for r in geo],.5),sampson_pair_p50_px=q([r.get('sampson_p50_px') for r in accepted],.5),sampson_pair_p90_px=q([r.get('sampson_p90_px') for r in accepted],.5),
                    depth_to_matched_RGB_pair_p50_px=q([r.get('source_depth_to_RGB_error_p50_px') for r in accepted],.5),depth_to_matched_RGB_valid_fraction=q([r.get('valid_fraction') for r in accepted],.5),
                    relative_rotation_p50_deg=q(rotation,.5),relative_rotation_p90_deg=q(rotation,.9),relative_rotation_gt90_count=sum(r>90 for r in rotation))
                summary.append(row)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['count','background'],required=True);args=parser.parse_args();sources=camera_sources();ref=sources['Omega_depth'];refs={source_id(n):ref['a_actual'][i] for i,n in enumerate(ref['names'])}
    if args.scope=='count':
        root=AUDIT/'controlled_fixes/view_count_control_v1';conditions=[condition_new(root/n) for n in ['Omega_70','Omega_110','Standard_70','Standard_110']];ids=sorted(set.intersection(*[set(c['frames']) for c in conditions]));conditions+=[condition_historical(m,sources) for m in ['Omega','Standard']];pairs=[(ids[i],ids[i+gap],gap) for gap in [1,2,4] for i in range(len(ids)-gap)]
    else:
        root=AUDIT/'controlled_fixes/background_control_v1';conditions=[]
        for n in ['Omega_RAP_baseline_24','Omega_background_restored_24','Standard_RAP_baseline_24','Standard_background_restored_24']:
            path=root/n
            if (path/'status.json').is_file() and json.loads((path/'status.json').read_text()).get('status')=='complete':conditions.append(condition_new(path))
        ids=[0,1,19,20,38,39,57,58,76,77,95,96,114,115,133,134,152,153,170,171,189,190,208,209];pairs=[(ids[i],ids[i+1],1) for i in range(0,len(ids),2)]
        assert conditions,'No completed background controls'
    output=root/'evaluation';output.mkdir(exist_ok=True);matches=cached_matches(pairs,output);queries={}
    for sid in ids:
        n=f'crop_{sid:04d}';mask=np.load(OMEGA_DATA/'per_frame'/f'model_mask_{n}.npy').squeeze()>0;y,x=np.nonzero(mask);original=np.c_[x,y,np.ones(len(x))]@np.linalg.inv(refs[sid]).T
        if len(original)>1500:original=original[np.random.default_rng(sid*101+7).choice(len(original),1500,replace=False)]
        queries[sid]=original[:,:2]
    write_json(output/'evaluation_protocol.json',dict(scope=args.scope,common_source_ids=ids,shared_grid_wh=[384,688],unordered_pairs=len(pairs),accepted_image_only_pairs=sum(p['accepted'] for p in matches),
        matches='Read-only existing RAP foregroundSIFT cache, mutual Lowe.75; deterministic image-only FM_RANSAC1.5px,min12inliers, same cached p1/p2 for every model/condition; no features from restored background',
        pair_definition='Count common-source rank offsets1/2/4 (not model subset neighbors); background12fixed original-ID-near pairs; each actual source-ID gap saved',
        geometry_queries='Same <=1500 original-coordinate queries per source from old Omega FGmask before model eligibility; fixed source-ID RNG',
        pixel_transform='Use actual pixel_transform_pixel_center for new controls; historical208 Standard same actual correction for measurement; preserve all original recorded matrices',
        count_scope='70vs110 same current environment and shared input/hash invariants.208 only historical reference; no old full environment freeze, cannot attribute70/110vs208 difference solely to count',
        depth_to_RGB='Bilinear saved source depth at same independently matched p1, target real p2 visible by RGB match, positive-depth valid fraction explicit; full matched denominator for3px support',
        no_gpu=True,no_inference=True,no_original_predictions_modified=True))
    results=[]
    for condition in conditions:
        result=evaluate(condition,matches,refs,queries,output);results.append(result);write_csv(output/'control_summary.csv',summarize(results));write_json(output/'control_summary.json',summarize(results))
    geometry=[r for result in results for r in result['geometry']];images=[r for result in results for r in result['image_matches']];rotation=[{k:v for k,v in r.items() if k not in ['relative_rotation','translation_direction']} for result in results for r in result['relative_pose']]
    write_csv(output/'cross_depth_pairs.csv',geometry);write_csv(output/'independent_RGB_pairs.csv',images);write_csv(output/'relative_camera_pairs.csv',rotation)
    changes=[];sim=[]
    for model in ['Omega','Standard']:
        selected=[c for c in conditions if c['model']==model]
        if len(selected)<2:continue
        base=selected[0];br=next(r for r in results if r['condition']==base['name']);br_lookup={(r['source_id'],r['target_id']):r for r in br['relative_pose']}
        for condition in selected[1:]:
            cr=next(r for r in results if r['condition']==condition['name'])
            for r in cr['relative_pose']:
                prior=br_lookup[(r['source_id'],r['target_id'])];delta=float(camera_angles(np.asarray(prior['relative_rotation']).T@np.asarray(r['relative_rotation'])));direction=float(np.rad2deg(np.arccos(np.clip(np.dot(prior['translation_direction'],r['translation_direction']),-1,1))))
                changes.append(dict(model=model,reference=base['name'],condition=condition['name'],source_id=r['source_id'],target_id=r['target_id'],common_rank_offset=r['common_rank_offset'],relative_rotation_disagreement_deg=delta,relative_translation_direction_disagreement_deg=direction))
            sim.append(pose_comparison(condition,base,ids))
    write_csv(output/'same_pair_relative_pose_changes.csv',changes);write_json(output/'source_matched_camera_Sim3.json',sim)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    summary=summarize(results);fig,axes=plt.subplots(1,3,figsize=(13,4))
    for result in results:
        rr=[r for r in summary if r['condition']==result['condition'] and r['control']=='mask_finite'];style='--' if rr[0]['historical'] else '-'
        for ax,key in zip(axes,['sampson_pair_p50_px','depth_to_matched_RGB_pair_p50_px','depth_support_5pct_all_queries']):ax.plot([r['common_rank_offset'] for r in rr],[r[key] for r in rr],style+'o',label=result['condition'])
    for ax,key in zip(axes,['Sampson shared-grid px','Source-depth to matched target RGB px','5% depth support / all common queries']):ax.set_ylabel(key);ax.set_xlabel('Fixed common-source rank pair offset');ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.suptitle('Fixed foreground features, identical source pairs and pixel queries\n'+('70/110 current controls;208historical only' if args.scope=='count' else '24identicalsourceIDs; restore only original Mask-exterior RGB; no new features'));fig.tight_layout();fig.savefig(output/'controlled_geometry_comparison.png',dpi=170);plt.close(fig)
    headers=['condition','same-pair images accepted','Sampson p50 px','depth→matched RGB p50 px','depth5% support','relativeR p90 deg']
    rows=[r for r in summary if r['control']=='mask_finite' and r['common_rank_offset']==1];table='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    for r in rows:table+='| '+' | '.join([r['condition'],str(r['accepted_shared_image_pairs'])+'/'+str(r['unordered_image_pairs']),str(round(r['sampson_pair_p50_px'],3)) if r['sampson_pair_p50_px'] is not None else 'unavailable',str(round(r['depth_to_matched_RGB_pair_p50_px'],3)) if r['depth_to_matched_RGB_pair_p50_px'] is not None else 'unavailable',f'{100*r["depth_support_5pct_all_queries"]:.2f}%',f'{r["relative_rotation_p90_deg"]:.2f}'])+' |\n'
    text='# '+('View-count control' if args.scope=='count' else 'Mask-exterior background control')+'\n\n'+table+'\nSame common source IDs:'+str(ids)+'. Fixed pairs are persisted in evaluation/fixed_pair_manifest.csv; same cached foreground-only RANSAC inlier coordinates are used in every condition. The source-depth-to-RGB statistic is conditional on positive saved source/target-camera depth; valid fractions and failures remain in CSV. No GT geometry/pose or measured metric-scale claim.\n\n'+('70vs110 uses the same current environment, input pixel hashes, masks, preprocessing, weights and seed.208 is reused historical data with no full old environment freeze; it is a reference, not a strict count-only control. Common38rank neighbors have varying original-ID gaps; no comparison of each subset own-neighbor sets.' if args.scope=='count' else 'Both conditions share24sourceIDs and12nearby source pairs. Only original Mask-exterior RGB is restored; original Mask-interior pixels are unchanged. Resize interpolation may mix changed background at boundaries. Foreground SIFT features stay fixed from old RAP cache; they are never re-extracted per background condition. Completed conditions listed; missing conditions are not fabricated.')+'\n\nRelative rotations use same camera pairs and are invariant to global world gauge. Camera centers are compared only after source-ID fit positiveSim3 and labelled diagnostic/reliability; no raw center subtraction. See evaluation/source_matched_camera_Sim3.json and same_pair_relative_pose_changes.csv.\n\nOutputs: control_summary.csv/json, cross_depth_pairs.csv, independent_RGB_pairs.csv, relative_camera_pairs.csv, fixed foreground matches and controlled_geometry_comparison.png. CPU only; historical03 and all original results unchanged.\n'
    (root/('COUNT_CONTROL_REPORT.md' if args.scope=='count' else 'BACKGROUND_CONTROL_REPORT.md')).write_text(text)
    print(json.dumps([{k:r[k] for k in ['condition','accepted_shared_image_pairs','sampson_pair_p50_px','depth_to_matched_RGB_pair_p50_px','depth_support_5pct_all_queries','relative_rotation_p90_deg']} for r in rows],indent=2),flush=True)

if __name__=='__main__':main()
