from common import *
from plyfile import PlyData,PlyElement
import csv
def export(name,removed,shrink=None):
    d=ROOT/'task0_gaussian_cleaning'/name;d.mkdir(parents=True,exist_ok=True);v=PlyData.read(str(SOURCE))['vertex'].data.copy();original=v.copy()
    if shrink is not None:
        for k in range(3):v[f'scale_{k}'][shrink]+=np.log(.5)
        ids=np.flatnonzero(shrink&~removed)
        with (d/'parameter_changes.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['original_gaussian_id','original_scale0','original_scale1','original_scale2','new_scale0','new_scale1','new_scale2','factor'])
            for i in ids:w.writerow([i,*[original[f'scale_{k}'][i] for k in range(3)],*[v[f'scale_{k}'][i] for k in range(3)],.5])
    for file,sel in [('gaussians.ply',~removed),('removed_gaussians.ply',removed)]:PlyData([PlyElement.describe(v[sel].copy(),'vertex')],text=False,byte_order='<').write(str(d/file))
    np.save(d/'original_gaussian_ids.npy',np.flatnonzero(~removed));np.save(d/'removed_original_gaussian_ids.npy',np.flatnonzero(removed))
    write(d/'edit_manifest.json',dict(source=str(SOURCE),source_sha256=sha(SOURCE),gaussians=int((~removed).sum()),deleted=int(removed.sum()),shrunk=int((shrink&~removed).sum()) if shrink is not None else 0,parameter='Native log scale axes factor0.5; XYZ,rotation,opacity,SH,filter_3D retained',native_schema=list(v.dtype.names)))
def main():
    d=ROOT/'task0_gaussian_cleaning/diagnosis';ev={k:v for k,v in np.load(d/'actual_contribution_evidence.npz').items()};old={k:v for k,v in np.load(SEL/'P2_GGGS_Cleaning/cleaning_evidence.npz').items()};v=PlyData.read(str(SOURCE))['vertex'].data
    fg=ev['fg_contribution']/208;bg=ev['bg_contribution']/208+ev['free_bg_contribution']/8;ratio=bg/np.maximum(fg+bg,1e-12);black=ev['free_black_contribution']/8
    scales=np.sort(np.exp(np.column_stack([v[f'scale_{k}'] for k in range(3)])),1);needle=scales[:,2]/np.maximum(scales[:,1],1e-12)
    r1=old['removed']|((ratio>.99)&(bg>.05))
    isolation=(ratio>.8)&(bg>.05)&(fg<1)&((old['nearest_supported_distance']>.0069557324)|((needle>10)&(old['depth_consistent']<2)))
    large_background=(ratio>.95)&(bg>.05)
    r2=r1|isolation|large_background
    mixed=(ratio>.7)&(bg>.05)&(ev['max_radius_px']>16)&~r2
    r3=r2.copy()
    aggressive=r2|((ratio>.8)&(bg>.05))
    export('round_01',r1);export('round_02',r2);export('round_03',r3,mixed);export('additional_rounds/aggressive_delete',aggressive)
    save_npz_fast(d/'round_rules.npz',round1_removed=r1,round2_removed=r2,round3_removed=r3,round3_shrunk=mixed,aggressive_removed=aggressive,harm_ratio=ratio,needle_ratio=needle)
    # Full ID-level evidence, with original native fields and old protection rationale.
    names=['original_gaussian_id',*v.dtype.names,*ev.keys(),'harm_ratio','needle_ratio','p2_protected','p2_removed','round1_removed','round2_removed','round3_shrunk']
    with (d/'gaussian_artifact_evidence.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(names)
        for i in range(len(v)):w.writerow([i,*v[i].tolist(),*[ev[k][i] for k in ev.keys()],ratio[i],needle[i],bool(old['protected'][i]),bool(old['removed'][i]),bool(r1[i]),bool(r2[i]),bool(mixed[i])])
    order=np.argsort(-black);top=order[:250]
    write(d/'dominant_cloud_gaussian_ids.json',dict(ids=top.tolist(),top250_black_contribution_fraction=float(black[top].sum()/black.sum()),top_ids_source_xyz=[[float(v[k][i]) for k in ['x','y','z']] for i in top],interpretation='Measured native contribution in fixed8 free-camera black ROIs. Original E10 compositing before deletion; remaining-model contribution must be reevaluated due occlusion changes.'))
    print('round counts',[(k,int((~m).sum())) for k,m in [('r1',r1),('r2',r2),('r3',r3),('aggressive',aggressive)]],'shrunk',int(mixed.sum()),flush=True)
if __name__=='__main__':main()
