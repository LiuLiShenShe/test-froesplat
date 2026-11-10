from common import *
from plyfile import PlyData,PlyElement
import open3d as o3d
from scipy.optimize import least_squares
def save(path,v):PlyData([PlyElement.describe(v.copy(),'vertex')],text=False,byte_order='<').write(str(path))
def main():
    d=ROOT/'task0_gaussian_cleaning/best_cleaned';d.mkdir(exist_ok=True);input=ROOT/'task0_gaussian_cleaning/additional_rounds/round_05';v=PlyData.read(str(input/'gaussians.ply'))['vertex'].data.copy();ids=np.load(input/'original_gaussian_ids.npy');xyz=np.column_stack([v[k] for k in ['x','y','z']]);marker=json.loads((ROOT/'task1_p3_local_repair/pot_marker/marker_region.json').read_text());pad=.0034778662;remove=np.all((xyz>=np.array(marker['bbox_min'])-pad)&(xyz<=np.array(marker['bbox_max'])+pad),axis=1)
    save(d/'historical_scene_gaussians.ply',v);save(d/'cleaned_gaussians.ply',v[~remove]);save(d/'removed_marker_gaussians.ply',v[remove]);np.save(d/'original_gaussian_ids.npy',ids[~remove]);np.save(d/'semantic_removed_original_ids.npy',ids[remove])
    original=PlyData.read(str(SOURCE))['vertex'].data;removed_all=np.ones(len(original),bool);removed_all[ids[~remove]]=False;save(d/'removed_gaussians.ply',original[removed_all]);np.save(d/'removed_original_gaussian_ids.npy',np.flatnonzero(removed_all))
    kept=v[~remove].copy();sigma=np.exp(np.column_stack([kept[f'scale_{k}'] for k in range(3)]).astype(np.float64));fil=kept['filter_3D'].astype(np.float64)[:,None];filtered=np.hypot(sigma,fil);op=1/(1+np.exp(-kept['opacity'].astype(np.float64)));newop=np.clip(op*(sigma/filtered).prod(1),1e-12,1-1e-7)
    baked=kept.copy()
    for k in range(3):baked[f'scale_{k}']=np.log(filtered[:,k])
    baked['opacity']=np.log(newop/(1-newop));baked['filter_3D']=0;save(d/'cleaned_gaussians_supersplat.ply',baked)
    save_npz_fast(d/'supersplat_baking_changes.npz',original_ids=ids[~remove],old_log_scales=np.column_stack([kept[f'scale_{k}'] for k in range(3)]),new_log_scales=np.column_stack([baked[f'scale_{k}'] for k in range(3)]),old_opacity_logit=kept['opacity'],new_opacity_logit=baked['opacity'],old_filter=kept['filter_3D'])
    # Plant-only variant removes fitted pot volume below its rim; label uncertain overlap explicitly.
    mesh=o3d.io.read_triangle_mesh(str(SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply'));points=np.asarray(mesh.vertices);colors=np.asarray(mesh.vertex_colors);body=points[(points[:,1]>.03)&(points[:,1]<.18)&(colors[:,1]>colors[:,0])&(colors.max(1)-colors.min(1)<.15)][::5]
    def residual(t):return np.linalg.norm(body[:,[0,2]]-t[:2],axis=1)-(t[2]+t[3]*body[:,1])
    fit=least_squares(residual,[.0,.96,.13,-.1],loss='soft_l1',f_scale=.003,max_nfev=150).x
    keptxyz=xyz[~remove];radius=np.linalg.norm(keptxyz[:,[0,2]]-fit[:2],axis=1);pot=(keptxyz[:,1]>-.015)&(radius<fit[2]+fit[3]*keptxyz[:,1]+.005)
    save(d/'plant_only_geometry.ply',kept[~pot]);np.save(d/'plant_only_original_gaussian_ids.npy',ids[~remove][~pot])
    write(d/'best_manifest.json',dict(selected_round='Round5 reattributed background layers',historical_count=len(v),scene_count=len(kept),marker_removed=int(remove.sum()),removed_all=int(removed_all.sum()),source=str(SOURCE),sha256=sha(d/'cleaned_gaussians.ply'),supersplat_sha256=sha(d/'cleaned_gaussians_supersplat.ply'),supersplat_conversion='Bake exact native filtered scaling and opacity coefficient into standard fields, zero filter_3D. Native parity must be tested.',plant_only_count=int((~pot).sum()),pot_volume_fit=dict(cx=float(fit[0]),cz=float(fit[1]),radius_intercept=float(fit[2]),radius_y_slope=float(fit[3]),rim_y=-.015),plant_only_status='Exploratory spatial semantic exclusion, roots/leaf-pot overlap unverified; not a certified complete botanical segmentation.',coordinate_system='Unchanged corrected standard VGGT / E10 OpenCV world gauge, no physical-unit calibration',marker='Actual independently disconnected blue marker AABB in original P3, plus2 voxels; semantic removal separate from background removal'))
    print('best scene',len(kept),'marker removed',int(remove.sum()),'plant-only',int((~pot).sum()),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
