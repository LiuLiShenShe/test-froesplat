"""Independent saved-camera statistics and source-ID-matched robust Sim3."""
from geometry_common import *

def independent(name,source):
    p=source['c2w'];centers=p[:,:3,3];finite=np.isfinite(p).all(axis=(1,2));extent=float(np.max(np.ptp(np.quantile(centers[finite],[.025,.975],axis=0),axis=0)))
    rel=np.einsum('nij,njk->nik',p[:-1,:3,:3].transpose(0,2,1),p[1:,:3,:3]);angles=camera_angles(rel)
    steps=np.linalg.norm(np.diff(centers,axis=0),axis=1)/max(extent,1e-15)
    k_original=np.linalg.inv(source['a_actual'])@source['k_model']
    rows=[]
    for i,n in enumerate(source['names']):
        rows.append(dict(method=name,name=n,source_id=source_id(n),center_x=float(centers[i,0]),center_y=float(centers[i,1]),center_z=float(centers[i,2]),
            fx_original=float(k_original[i,0,0]),fy_original=float(k_original[i,1,1]),cx_original=float(k_original[i,0,2]),cy_original=float(k_original[i,1,2]),
            rotation_step_deg=float(angles[i-1]) if i else None,translation_step_normalized=float(steps[i-1]) if i else None))
    write_csv(AUDIT/'geometry_comparison'/f'trajectory_{name}_per_frame.csv',rows)
    return dict(method=name,frames=len(p),finite_cameras=int(finite.sum()),duplicate_source_ids=len(source['names'])-len(set(source['names'])),
        consecutive_same_camera=int(np.sum((np.linalg.norm(np.diff(centers,axis=0),axis=1)<extent*1e-6)&(angles<.001))),
        rotation_step_p50_deg=q(angles,.5),rotation_step_p90_deg=q(angles,.9),rotation_step_max_deg=q(angles,1),
        rotation_step_gt30_count=int((angles>30).sum()),rotation_step_gt90_count=int((angles>90).sum()),
        translation_step_p50_normalized=q(steps,.5),translation_step_p90_normalized=q(steps,.9),extent_camera_centers=extent,
        fx_original_p10=q(k_original[:,0,0],.1),fx_original_p50=q(k_original[:,0,0],.5),fx_original_p90=q(k_original[:,0,0],.9),
        fy_original_p50=q(k_original[:,1,1],.5),rotation_det_max_error=float(np.max(np.abs(np.linalg.det(p[finite,:3,:3])-1))),
        source_id_order=[source_id(n) for n in source['names']])

def robust_match(name,source,reference):
    mapping={n:i for i,n in enumerate(source['names'])};names=[n for n in reference['names'] if n in mapping]
    src=np.array([source['c2w'][mapping[n]] for n in names]);target=np.array([reference['c2w'][reference['names'].index(n)] for n in names]);valid=np.isfinite(src).all(axis=(1,2))&np.isfinite(target).all(axis=(1,2))
    src=src[valid];target=target[valid];names=[n for n,v in zip(names,valid) if v]
    x=src[:,:3,3];y=target[:,:3,3];extent=float(np.ptp(np.quantile(y,[.025,.975],axis=0),axis=0).max())
    holdout=np.arange(len(x))%4==0;train=~holdout;inlier=train.copy();s,r,t=fit_similarity(x[inlier],y[inlier])
    for _ in range(12):
        residual=np.linalg.norm(s*x@r.T+t-y,axis=1);med=np.median(residual[train]);mad=np.median(np.abs(residual[train]-med))
        candidate=train&(residual<=med+3*max(1.4826*mad,extent*1e-6))
        if candidate.sum()<12 or np.array_equal(inlier,candidate):break
        inlier=candidate;s,r,t=fit_similarity(x[inlier],y[inlier])
    aligned=s*x@r.T+t;residual=np.linalg.norm(aligned-y,axis=1)/max(extent,1e-15)
    oriented=np.einsum('ij,njk->nik',r,src[:,:3,:3]);angle=camera_angles(np.einsum('nij,njk->nik',target[:,:3,:3].transpose(0,2,1),oriented))
    singular=np.linalg.svd(x[inlier]-x[inlier].mean(0),compute_uv=False)
    scales=[]
    for split in range(4):
        subset=inlier&(np.arange(len(x))%4!=split)
        if subset.sum()>=12:scales.append(fit_similarity(x[subset],y[subset])[0])
    stable_scale=(max(scales)-min(scales))/max(abs(s),1e-15) if scales else None
    reliable=(s>0 and singular[1]/max(singular[0],1e-15)>.05 and q(residual[holdout],.5)<.10 and q(residual[holdout],.9)<.25 and q(angle[holdout],.5)<15 and q(angle[holdout],.9)<30 and stable_scale is not None and stable_scale<.2)
    rows=[dict(method=name,name=n,source_id=source_id(n),fit_inlier=bool(inlier[i]),held_out=bool(holdout[i]),normalized_center_error=float(residual[i]),orientation_error_deg=float(angle[i])) for i,n in enumerate(names)]
    write_csv(AUDIT/'geometry_comparison'/f'trajectory_{name}_Sim3_residuals.csv',rows)
    result=dict(method=name,reference='Omega_depth',source_id_correspondences=len(names),training_inliers=int(inlier.sum()),held_out_frames=int(holdout.sum()),
        similarity_scale=float(s),rotation=r,translation=t,positive_scale=bool(s>0),rotation_determinant=float(np.linalg.det(r)),
        held_out_center_error_p50_normalized=q(residual[holdout],.5),held_out_center_error_p90_normalized=q(residual[holdout],.9),
        held_out_orientation_error_p50_deg=q(angle[holdout],.5),held_out_orientation_error_p90_deg=q(angle[holdout],.9),
        camera_path_second_singular_ratio=float(singular[1]/max(singular[0],1e-15)),scale_spread_across_four_splits=stable_scale,
        reliable_similarity_comparison=bool(reliable),status='relative_camera_agreement_supported' if reliable else 'Unresolved: globally reliable camera Sim3 not established',
        scope='Camera centers and orientations matched by exact208original source filenames; held-out evaluation; display/diagnostic only, no pointmap or dataset modification, not GT pose accuracy')
    return result

def main():
    sources=camera_sources();reference=sources['Omega_depth'];ordered=reference['names']
    checks={}
    for name,s in sources.items():checks[name]=dict(exact_source_name_order=s['names']==ordered,frames=len(s['names']),finite_K=bool(np.isfinite(s['k_model']).all()),inverse_error=float(np.max(np.abs(s['c2w']@s['w2c']-np.eye(4)))))
    standard=sources['Standard_depth'];checks['Standard_depth']['COLMAP_image_and_camera_ids_match_input_order']=all(standard['colmap_ids'][n]==[i+1,i+1] for i,n in enumerate(ordered))
    stats=[independent(name,s) for name,s in sources.items()];matches=[robust_match(name,s,reference) for name,s in sources.items() if name!='Omega_depth']
    write_json(AUDIT/'geometry_comparison/camera_source_checks.json',checks);write_json(AUDIT/'geometry_comparison/camera_trajectory_independent.json',stats);write_csv(AUDIT/'geometry_comparison/camera_trajectory_independent.csv',stats)
    write_json(AUDIT/'geometry_comparison/camera_trajectory_Sim3.json',matches);write_csv(AUDIT/'geometry_comparison/camera_trajectory_Sim3.csv',[{k:v for k,v in m.items() if k not in ('rotation','translation')} for m in matches])
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for name,s in sources.items():
        relative=np.einsum('nij,njk->nik',s['c2w'][:-1,:3,:3].transpose(0,2,1),s['c2w'][1:,:3,:3]);angles=camera_angles(relative)
        axes[0].plot([source_id(n) for n in s['names'][1:]],angles,label=name,linewidth=.7)
        ko=np.linalg.inv(s['a_actual'])@s['k_model'];axes[1].plot([source_id(n) for n in s['names']],ko[:,0,0],label=name,linewidth=.7)
    axes[0].set_ylabel('Rotation step / degrees');axes[1].set_ylabel('fx in original pixels');axes[1].set_xlabel('Original source frame ID');axes[0].legend(ncol=4,fontsize=8)
    fig.suptitle('Independent saved camera trajectories and focal predictions\nExact source IDs; arbitrary scene scales; no GT pose claim');fig.tight_layout();fig.savefig(AUDIT/'geometry_comparison/camera_rotation_and_focal.png',dpi=160);plt.close(fig)
    for row in stats:print(row['method'],row['frames'],'rotation p50/p90',row['rotation_step_p50_deg'],row['rotation_step_p90_deg'],'fx median',row['fx_original_p50'],flush=True)
    for row in matches:print('Sim3',row['method'],'reliable',row['reliable_similarity_comparison'],'heldout center/orientation',row['held_out_center_error_p50_normalized'],row['held_out_orientation_error_p50_deg'],flush=True)

if __name__=='__main__':main()
