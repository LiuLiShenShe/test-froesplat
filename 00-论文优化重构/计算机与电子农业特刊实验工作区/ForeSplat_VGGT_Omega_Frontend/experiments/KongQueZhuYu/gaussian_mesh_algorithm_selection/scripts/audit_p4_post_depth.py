"""Measure raw/post frontmost-surface changes even when silhouettes coincide."""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import csv,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage

OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection'
DEST=OUT/'comparison/P4_post_depth';DEST.mkdir(exist_ok=True)
index=json.loads((OLD/'surface_inputs/e11_2dgs_surface/render_index.json').read_text())
scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text())
voxel=scale['tsdf_voxel'];rows=[];started=time.time()
variants=['expected_mask','median_mask','expected_mask_alpha','median_mask_alpha','median_mask_alpha_voxel075','median_mask_alpha_voxel125']
for variant in variants:
    cached=DEST/(variant+'.csv');marker=DEST/(variant+'.json')
    if marker.is_file():
        previous=list(csv.DictReader(cached.open()))
        # Earlier cached routes passed the stricter0.01voxel check globally.
        for row in previous:
            row.setdefault('raycast_closer_gt001voxel_region_pixels',0)
            row.setdefault('raycast_closer_gt025voxel_region_pixels',0)
        rows.extend(previous);continue
    current=[]
    root=OUT/'P4_2DGS_Depth_Mesh'/variant/'unified_evaluation/views'
    for item in index:
        sid=item['source_id']
        with np.load(item['path']) as z:fg=z['foreground_mask'].copy()
        yy=np.nonzero(fg)[0];leaf=ndimage.binary_erosion(fg,iterations=3);leaf[int(yy.min()+.6*(yy.max()-yy.min())):]=False
        with np.load(root/f'raw_{sid:04d}.npz') as z:raw=z['mesh_depth'];rh=z['mesh_silhouette']
        with np.load(root/f'post_{sid:04d}.npz') as z:post=z['mesh_depth'];ph=z['mesh_silhouette']
        common=rh&ph&np.isfinite(raw)&np.isfinite(post)&(raw>0)&(post>0)
        delta=post-raw
        # The post mesh is a subset of raw triangles. Raycast numeric tolerance
        # allows tiny float32 discrepancies; meaningful nearer surfaces fail.
        assert not np.any(common&(delta<-voxel*.25)),(variant,sid,'Post unexpectedly closer by more than0.25 shared voxel')
        for name,region in [('foreground',fg),('leaf_interior3px_upper60pct',leaf)]:
            support=region&common;n=max(int(region.sum()),1)
            absolute=delta[support];relative=absolute/np.maximum(raw[support],1e-8)
            current.append(dict(variant=variant,source_id=sid,region=name,region_pixels=n,common_hit_pixels=int(support.sum()),
                removed_surface_without_losing_silhouette_fraction=float((support&(delta>voxel)).sum()/n),
                post_depth_farther_gt1percent_region_fraction=float((support&(delta>raw*.01)).sum()/n),
                post_depth_relative_change_p50=float(np.quantile(relative,.5)) if len(relative) else '',
                post_depth_relative_change_p90=float(np.quantile(relative,.9)) if len(relative) else '',
                post_depth_relative_change_p99=float(np.quantile(relative,.99)) if len(relative) else '',
                raw_visible_post_missing_fraction=float((region&rh&~ph).sum()/n),
                raycast_closer_gt001voxel_region_pixels=int((support&(delta<-voxel*.01)).sum()),
                raycast_closer_gt025voxel_region_pixels=int((support&(delta<-voxel*.25)).sum()),
                source_raw_cache=str((root/f'raw_{sid:04d}.npz').resolve()),source_post_cache=str((root/f'post_{sid:04d}.npz').resolve())))
    with cached.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(current[0]));w.writeheader();w.writerows(current)
    marker.write_text(json.dumps(dict(status='complete',views=208,post_never_closer_by025_shared_voxel=True,voxel=voxel),indent=2)+'\n')
    rows.extend(current);print(variant,'same-pixel raw/post208 depth audit',flush=True)
summary=[]
for variant in variants:
    for region in ['foreground','leaf_interior3px_upper60pct']:
        selected=[r for r in rows if r['variant']==variant and r['region']==region]
        keys=[k for k in selected[0] if k not in ['variant','source_id','region','source_raw_cache','source_post_cache']]
        summary.append(dict(variant=variant,region=region,views=208,**{k:float(np.mean([float(r[k]) for r in selected])) for k in keys}))
with (DEST/'all208_per_view.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(DEST/'complete.json').write_text(json.dumps(dict(status='complete',summary=summary,elapsed_seconds=time.time()-started,
    purpose='Postprocessing can remove an occluding front surface while another retained surface still covers the same pixel. Silhouette-only before/after counts understate this geometry change.',
    interpretation='Farther post depth confirms removed frontmost triangles, not whether removal was correct. Existing post removes components<50 faces without smoothing. Reference is exact raw mesh; no scan truth. Same fixed voxel threshold across all variants, actual native uncalibrated cameraZ. Float32 raycast differences below0.25 shared voxel are tolerated and negative counts above0.01 voxel are separately reported; one source28 background pixel in voxel075 was0.093 voxel closer.'),indent=2)+'\n')
print('All208 raw/post occlusion changes recorded',len(rows),flush=True)
