"""P4 samecheckpoint pixel evidence: depth availability, raw fusion, component post.
No scan: a rendered source depth can be valid and geometrically wrong.
"""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MPLBACKEND='Agg')
import json,csv,time
from pathlib import Path
import numpy as np
from scipy import ndimage
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection';P4=OUT/'P4_2DGS_Depth_Mesh';start=time.time()
names=['expected_mask','median_mask','expected_mask_alpha','median_mask_alpha','median_mask_alpha_voxel075','median_mask_alpha_voxel125']
assert all((P4/name/'unified_evaluation/complete.json').is_file() for name in names)
index=json.loads((OLD/'surface_inputs/e11_2dgs_surface/render_index.json').read_text());rows=[]
for item in index:
    sid=item['source_id']
    with np.load(item['path']) as z:fg=z['foreground_mask'];alpha=z['alpha'];expected=z['depth_expected'];median=z['depth_median']
    yy,xx=np.nonzero(fg);leaf=ndimage.binary_erosion(fg,iterations=3);leaf[int(yy.min()+.6*(yy.max()-yy.min())):]=False
    offset=np.abs(expected-median);rel=offset/np.maximum(np.abs(median),1e-8);validpair=leaf&np.isfinite(expected)&np.isfinite(median)&(expected>0)&(median>0)
    for name in names:
        folder=P4/name/'unified_evaluation'
        with np.load(folder/'views'/f'raw_{sid:04d}.npz') as z:raw=z['mesh_silhouette'];rh=z['internal_missing_proxy']
        with np.load(folder/'views'/f'post_{sid:04d}.npz') as z:post=z['mesh_silhouette'];ph=z['internal_missing_proxy']
        dep=expected if name.startswith('expected') else median
        valid=np.isfinite(dep)&(dep>0)&fg
        if 'alpha' in name:valid &= alpha>=.5
        for region,mask in [('foreground',fg),('leaf_interior3px_upper60pct',leaf)]:
            n=max(int(mask.sum()),1);missing=mask&~raw;post_removed=mask&raw&~post
            rows.append(dict(variant=name,source_id=sid,region=region,region_pixels=n,source_depth_valid_fraction=float((mask&valid).sum()/n),alpha05_fraction=float((mask&(alpha>=.5)).sum()/n),raw_mesh_coverage=float((mask&raw).sum()/n),post_mesh_coverage=float((mask&post).sum()/n),source_valid_but_raw_missing_fraction=float((missing&valid).sum()/n),source_invalid_and_raw_missing_fraction=float((missing&~valid).sum()/n),post_removed_visible_fraction=float(post_removed.sum()/n),post_added_visible_fraction=float((mask&post&~raw).sum()/n),raw_enclosed_missing_area_px=int((mask&rh).sum()),post_enclosed_missing_area_px=int((mask&ph).sum()),post_new_enclosed_missing_area_px=int((mask&ph&~rh).sum()),expected_vs_median_abs_depth_median=float(np.median(offset[validpair])) if validpair.any() else '',expected_vs_median_relative_depth_median=float(np.median(rel[validpair])) if validpair.any() else '',expected_vs_median_relative_depth_p90=float(np.quantile(rel[validpair],.9)) if validpair.any() else ''))
    print('P4 stage correspondence',sid,flush=True)
dest=OUT/'comparison/p4_stage_decomposition.csv'
with dest.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for name in names:
    for region in ['foreground','leaf_interior3px_upper60pct']:
        r=[r for r in rows if r['variant']==name and r['region']==region]
        summary.append(dict(variant=name,region=region,views=len(r),**{k:float(np.mean([r[k] for r in r])) for k in r[0] if isinstance(r[0][k],(int,float)) and k!='source_id'}))
(OUT/'comparison/P4_DECOMPOSITION.json').write_text(json.dumps(dict(status='complete',summary=summary,elapsed_seconds=time.time()-start,source_checkpoint='Actual same E11 native30K, no retraining',interpretation='Depth availability: finite positive source native depth. Sourcevalid-but-rawmissing identifies loss at fusion/reprojection stage, but valid depth can be wrong and causality is not exclusively TSDF. Rawvisible-to-postmissing isolates effects of component postprocessing with unchanged same geometry; no smoothing. Voxel alternatives preserve all raw/post.',units='Uncalibrated VGGT cameraZ for depth; image pixels for projected missing areas; not physical3D holes or scan distance'),indent=2))
