"""Same independent RGB matches and original-pixel queries for old/new208."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];AUDIT=ROOT.parent/'root_cause_audit'
sys.path.insert(0,str(AUDIT/'scripts'))
from geometry_common import camera_sources,write_csv,write_json,camera_angles,q
from geometry_targeted_control_eval import condition_new,condition_historical,load_frame,cached_matches,depth_rgb
from geometry_multiview_audit import references,prepare_queries,pair_metrics,summarize
from camera_epipolar_audit import fundamental,sampson
import cv2
OUT=ROOT/'geometry_comparison'

def main():
    sources=camera_sources();conditions=[condition_historical(m,sources) for m in ['Omega','Standard']]
    conditions+=[condition_new(ROOT/n) for n in ['VGGT_Omega_Original_RGB','Standard_VGGT_Original_RGB']]
    ids=list(conditions[0]['frames']);assert len(ids)==208
    assert all(list(c['frames'])==ids for c in conditions)
    refs={sid:sources['Omega_depth']['a_actual'][i] for i,sid in enumerate(ids)}
    reference_frames=[load_frame(conditions[0],sid) for sid in ids]
    queries=references([dict(f,source_id=sid) for f,sid in zip(reference_frames,ids)])
    pairs=[(ids[i],ids[i+gap],gap) for gap in [1,5,15] for i in range(0,len(ids)-gap,3)]
    matches=cached_matches(pairs,OUT)
    geometry=[];images=[];trajectories=[];coverage=[]
    pooled_epipolar={c['name']:[] for c in conditions}
    for c in conditions:
        frames=[load_frame(c,sid) for sid in ids]
        for frame,sid in zip(frames,ids):
            frame['index']=ids.index(sid)
            fg=frame['mask'];finite=np.isfinite(frame['depth'])&(frame['depth']>0)&np.isfinite(frame['conf']);threshold=q(np.concatenate([f['conf'][f['mask']] for f in frames]),.2) if sid==ids[0] else threshold
            from scipy.ndimage import binary_erosion
            boundary=fg&~binary_erosion(fg);orig=np.c_[np.nonzero(boundary)[1],np.nonzero(boundary)[0],np.ones(boundary.sum())]@np.linalg.inv(frame['a']).T
            local=(orig[:,0]>=510)&(orig[:,0]<720)&(orig[:,1]>=630)&(orig[:,1]<1110)
            v=finite&(frame['conf']>=threshold)
            shared_fg=prepare_queries(frame,queries[frame['name']]['foreground'])
            shared_edge_points=queries[frame['name']]['foreground_boundary_common_grid']
            leaf=(shared_edge_points[:,0]>=510)&(shared_edge_points[:,0]<720)&(shared_edge_points[:,1]>=630)&(shared_edge_points[:,1]<1110)
            shared_edge=prepare_queries(frame,shared_edge_points[leaf])
            coverage.append(dict(method=c['name'],source_id=sid,fg_pixels=int(fg.sum()),fg_valid=int((fg&v).sum()),boundary_pixels=int(boundary.sum()),boundary_valid=int((boundary&v).sum()),leaf_edge_pixels=int(local.sum()),leaf_edge_valid=int(v[boundary][local].sum()),confidence_p20=threshold,
                shared_fg_queries=len(shared_fg['finite']),shared_fg_valid=int((shared_fg['finite']&(shared_fg['conf']>=threshold)).sum()),shared_leaf_edge_queries=len(shared_edge['finite']),shared_leaf_edge_valid=int((shared_edge['finite']&(shared_edge['conf']>=threshold)).sum())))
        centers=np.array([f['c2w'][:3,3] for f in frames]);extent=np.ptp(np.quantile(centers,[.025,.975],axis=0),axis=0).max()
        rotations=[float(camera_angles(frames[i+1]['w2c'][:3,:3]@frames[i]['c2w'][:3,:3])) for i in range(207)]
        step=np.linalg.norm(np.diff(centers,axis=0),axis=1)
        trajectories.append(dict(method=c['name'],frames=208,rotation_step_p50_deg=q(rotations,.5),rotation_step_p90_deg=q(rotations,.9),rotation_step_gt90_count=sum(r>90 for r in rotations),
            translation_step_p90_per_extent=q(step,.9)/extent,closure_center_gap_per_extent=float(np.linalg.norm(centers[-1]-centers[0])/extent),closure_orientation_gap_deg=float(camera_angles(frames[-1]['w2c'][:3,:3]@frames[0]['c2w'][:3,:3]))))
        np.savez(OUT/f'{c["name"]}_trajectory.npz',source_ids=ids,centers=centers,w2c=np.array([f['w2c'] for f in frames]),intrinsics=np.array([f['k'] for f in frames]))
        for frame,sid in zip(frames,ids):
            for offset in [1,5,15,30,60]:
                for target_id in [sid-offset,sid+offset]:
                    if target_id not in c['frames']:continue
                    for region,points in queries[frame['name']].items():
                        if not len(points):continue
                        with np.errstate(invalid='ignore',over='ignore'):
                            measurement=pair_metrics(prepare_queries(frame,points),frame,c['frames'][target_id],'mask_finite')
                        geometry.append(dict(method=c['name'],source_id=sid,target_id=target_id,source_offset=target_id-sid,region=region,control='mask_finite',**measurement))
        for pair in matches:
            if not pair['accepted']:continue
            sid,tid=pair['source_id'],pair['target_id'];s,t=c['frames'][sid],c['frames'][tid]
            f=fundamental(refs[sid]@np.linalg.inv(s['a'])@s['k'],refs[tid]@np.linalg.inv(t['a'])@t['k'],s['w2c'],t['w2c']);err=sampson(f,pair['p1'],pair['p2'])
            if pair['common_rank_offset']==1:pooled_epipolar[c['name']].extend(err.tolist())
            measurement=depth_rgb(s,t,pair['p1'],pair['p2'],refs[sid],refs[tid],c['model'])[0]
            images.append(dict(method=c['name'],region='foreground',source_id=sid,target_id=tid,rank_offset=pair['common_rank_offset'],actual_source_offset=tid-sid,sampson_p50_px=q(err,.5),sampson_p90_px=q(err,.9),**measurement))
        print('evaluated full208',c['name'],flush=True)
        write_csv(OUT/'cross_depth_pairs.csv',geometry);write_csv(OUT/'rgb_pairs.csv',images);write_csv(OUT/'camera_stability.csv',trajectories);write_csv(OUT/'point_coverage.csv',coverage)
    summary=summarize(geometry);write_csv(OUT/'depth_summary.csv',summary)
    # Background RGB-only feature evidence on the same shared384x688 grid.
    cv2.setNumThreads(1);features={};bf=cv2.BFMatcher(cv2.NORM_L2)
    def feature(sid):
        if sid in features:return features[sid]
        raw=Image.open(ROOT/'original_images'/f'{sid:04d}.jpg').convert('RGB').resize((384,688),Image.Resampling.BICUBIC)
        mask=np.asarray(Image.open(ROOT/'masks'/f'crop_{sid:04d}.png').resize((384,688),Image.Resampling.NEAREST))>127
        key,desc=cv2.SIFT_create(nfeatures=1600).detectAndCompute(cv2.cvtColor(np.asarray(raw),cv2.COLOR_RGB2GRAY),(~mask).astype(np.uint8)*255)
        features[sid]=(np.asarray([k.pt for k in key],np.float32).reshape(-1,2),desc);return features[sid]
    for sid,tid,gap in pairs:
        if gap!=1:continue
        p1,d1=feature(sid);p2,d2=feature(tid)
        if d1 is None or d2 is None or min(len(d1),len(d2))<12:continue
        reverse={m.queryIdx:m.trainIdx for pair in bf.knnMatch(d2,d1,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance}
        good=[m for pair in bf.knnMatch(d1,d2,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
        a=np.asarray([p1[m.queryIdx] for m in good],np.float32);b=np.asarray([p2[m.trainIdx] for m in good],np.float32)
        cv2.setRNGSeed(1107+sid*211+tid)
        ff,keep=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.5,.999) if len(a)>=12 else (None,None)
        if ff is None or ff.shape!=(3,3) or keep.sum()<12:continue
        a,b=a[keep.ravel()>0],b[keep.ravel()>0]
        np.savez(OUT/f'background_matches_{sid:04d}_{tid:04d}.npz',p1=a,p2=b,image_f=ff)
        for c in conditions:
            s,t=c['frames'][sid],c['frames'][tid];f=fundamental(refs[sid]@np.linalg.inv(s['a'])@s['k'],refs[tid]@np.linalg.inv(t['a'])@t['k'],s['w2c'],t['w2c']);err=sampson(f,a,b)
            pooled_epipolar[c['name']].extend(err.tolist())
            images.append(dict(method=c['name'],region='background',source_id=sid,target_id=tid,rank_offset=1,actual_source_offset=tid-sid,sampson_p50_px=q(err,.5),sampson_p90_px=q(err,.9),matched_points=len(a)))
    write_csv(OUT/'rgb_pairs.csv',images)
    comparison=[]
    for c in conditions:
        fg=[r for r in images if r['method']==c['name'] and r['region']=='foreground' and r['rank_offset']==1];bg=[r for r in images if r['method']==c['name'] and r['region']=='background']
        geo=next(r for r in summary if r['method']==c['name'] and r['region']=='foreground' and r['source_offset']==1)
        edge=next(r for r in summary if r['method']==c['name'] and r['region']=='foreground_boundary_common_grid' and r['source_offset']==1)
        leaf=next(r for r in summary if r['method']==c['name'] and r['region']=='leaf_window' and r['source_offset']==1)
        leaf_edge=next(r for r in summary if r['method']==c['name'] and r['region']=='leaf_edge_window' and r['source_offset']==1)
        cv=[r for r in coverage if r['method']==c['name']]
        comparison.append(dict(epipolar_error_pooled_fg_background_shared_px=q(pooled_epipolar[c['name']],.5),foreground_epipolar_error_shared_px=q([r['sampson_p50_px'] for r in fg],.5),background_epipolar_error_shared_px=q([r['sampson_p50_px'] for r in bg],.5),
            foreground_rgb_depth_error_shared_px=q([r['source_depth_to_RGB_error_p50_px'] for r in fg],.5),foreground_depth_support_ratio=geo['support_5pct_fraction'],foreground_edge_depth_support_ratio=edge['support_5pct_fraction'],depth_consistency_relative_error=geo['raw_overlap_relative_depth_error_p50_across_pair_median'],
            leaf_window_depth_support_ratio=leaf['support_5pct_fraction'],leaf_edge_window_depth_support_ratio=leaf_edge['support_5pct_fraction'],
            foreground_point_completeness=sum(r['fg_valid'] for r in cv)/sum(r['fg_pixels'] for r in cv),leaf_edge_point_coverage=sum(r['leaf_edge_valid'] for r in cv)/sum(r['leaf_edge_pixels'] for r in cv),
            shared_grid_foreground_point_completeness=sum(r['shared_fg_valid'] for r in cv)/sum(r['shared_fg_queries'] for r in cv),shared_grid_leaf_edge_point_coverage=sum(r['shared_leaf_edge_valid'] for r in cv)/sum(r['shared_leaf_edge_queries'] for r in cv),**next(r for r in trajectories if r['method']==c['name'])))
    write_csv(ROOT/'frontend_comparison.csv',comparison)
    write_json(OUT/'protocol.json',dict(shared_grid_wh=[384,688],full_source_ids=ids,depth_queries='Exact original geometry_multiview_audit references/prepare_queries/pair_metrics, <=1500 per original ROI; original source-ID offsets; full-query denominator',
        foreground_matches='Existing RAP foreground SIFT; same deterministic image-only RANSAC p1/p2 for all four old/new conditions',background_matches='Same untouched02-FFT RGB background features for all conditions; independent F selection, depth unused',overall_epipolar='Median pooled foreground and background accepted point errors at rank offset1; FG/BG separately report median of pair medians',point_completeness='Native-grid coverage and independently shared original-coordinate foreground/leaf-boundary queries both saved; same original p20 confidence filter',pixel_transform='Actual half-pixel A for both old and new standard measurements; original A separately retained',absolute_camera_or_3d_gt_available=False))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,c in zip(axes,conditions):
        data=np.load(OUT/f'{c["name"]}_trajectory.npz');p=data['centers'];ax.plot(p[:,0],p[:,2]);ax.scatter(p[0,0],p[0,2],c='g');ax.scatter(p[-1,0],p[-1,2],c='r');ax.set_title(c['name']);ax.set_aspect('equal')
    fig.tight_layout();fig.savefig(OUT/'camera_trajectories.png',dpi=160)
if __name__=='__main__':main()
