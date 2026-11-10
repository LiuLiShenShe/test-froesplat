"""Stratify actual 208-view raycasts using an unchanged RGB-independent mask rule.

The upper 60% of the foreground bounding-box height is a leaf-region proxy,
not a new semantic segmentation. It matches the P4 decomposition convention.
Whole-plant scores are also checked against the existing measured tables.
"""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import csv,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
from queue_io import write_status

OUT=Path(__file__).resolve().parents[1]
OLD=OUT.parent/'geometry_mesh_final_selection'
DEST=OUT/'comparison/leaf_region_mesh'
DEST.mkdir(exist_ok=True)
truth=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text())
assert len(truth)==208
metadata={r['source_id']:r for r in truth}
masks={}
for r in truth:
    with np.load(r['path']) as z:fg=z['foreground_mask'].copy()
    assert hashlib.sha256(fg.tobytes()).hexdigest()==r['mask_sha256']
    yy=np.nonzero(fg)[0]
    cut=int(yy.min()+.6*(yy.max()-yy.min()))
    interior=ndimage.binary_erosion(fg,iterations=3);interior[cut:]=False
    masks[r['source_id']]=(fg,interior,cut)

def csvwrite(path,rows):
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)

def enclosed(pred,fg):
    labels,count=ndimage.label(fg&~pred)
    rim=fg&~ndimage.binary_erosion(fg,iterations=3)
    sizes=np.bincount(labels.ravel())
    ids=np.flatnonzero(sizes>=9)
    ids=ids[(ids>0)&~np.isin(ids,np.unique(labels[rim]))]
    return np.isin(labels,ids)

allrows=[];summaries=[];started=time.time()
routes=list(csv.DictReader((OUT/'comparison/ALL_MESH_EXPERIMENTS.csv').open()))
for route in routes:
    name,stage=route['mesh_id'],route['mesh_stage']
    measured=Path(route['metric_source'])
    if not measured.is_file():continue
    source=list(csv.DictReader(measured.open()))
    assert len(source)==208
    key=name+'_'+stage
    dest=DEST/(key+'.csv');marker=DEST/(key+'.json')
    if marker.is_file():
        info=json.loads(marker.read_text())
        assert info['source_metrics']==str(measured.resolve())
        rows=list(csv.DictReader(dest.open()))
        for row in rows:
            for k in row:
                if k not in ['mesh_id','mesh_stage','source_native_raycast','region_definition']:
                    row[k]=float(row[k])
    else:
        rows=[]
        for original in source:
            sid=int(original['source_id']);fg,interior,cut=masks[sid]
            cache=Path(original['source_native_raycast']) if original.get('source_native_raycast') else measured.parent/'views'/f'{stage}_{sid:04d}.npz'
            assert cache.is_file(),cache
            with np.load(cache) as z:
                pred=z['mesh_silhouette'].copy()
                holes=z['internal_missing_proxy'].copy() if 'internal_missing_proxy' in z else enclosed(pred,fg)
                if 'foreground_mask' in z:assert np.array_equal(z['foreground_mask'],fg)
            assert pred.dtype==bool and pred.shape==fg.shape
            whole_iou=(pred&fg).sum()/max((pred|fg).sum(),1)
            assert np.isclose(whole_iou,float(original['silhouette_iou']),atol=1e-10),(key,sid,'Whole-score mismatch')
            upper_fg=fg[:cut];upper_pred=pred[:cut]
            lower_fg=fg[cut:];lower_pred=pred[cut:]
            rows.append(dict(mesh_id=name,mesh_stage=stage,source_id=sid,cut_y_native=cut,
                region_definition='RAP bounding-box upper60pct height, same frozen P4 rule; approximate leaf region, not semantic segmentation',
                upper_fg_pixels=int(upper_fg.sum()),lower_fg_pixels=int(lower_fg.sum()),
                upper_silhouette_iou=float((upper_fg&upper_pred).sum()/max((upper_fg|upper_pred).sum(),1)),
                upper_foreground_coverage=float((upper_fg&upper_pred).sum()/max(upper_fg.sum(),1)),
                upper_background_geometry_ratio=float((upper_pred&~upper_fg).sum()/max(upper_pred.sum(),1)),
                upper_enclosed_missing_area_px=int(holes[:cut].sum()),
                upper_enclosed_missing_fg_ratio=float(holes[:cut].sum()/max(upper_fg.sum(),1)),
                leaf_interior3px_coverage=float((pred&interior).sum()/max(interior.sum(),1)),
                leaf_interior3px_enclosed_missing_area_px=int((holes&interior).sum()),
                lower_silhouette_iou=float((lower_fg&lower_pred).sum()/max((lower_fg|lower_pred).sum(),1)),
                lower_foreground_coverage=float((lower_fg&lower_pred).sum()/max(lower_fg.sum(),1)),
                whole_silhouette_iou=float(whole_iou),source_native_raycast=str(cache.resolve())))
        csvwrite(dest,rows)
        write_status(marker,dict(status='complete',source_metrics=str(measured.resolve()),views=208,whole208_iou_matches=True,mask_source=str((OLD/'surface_inputs/e10_gggs_surface/render_index.json').resolve())))
    allrows+=rows
    columns=[k for k in rows[0] if k not in ['mesh_id','mesh_stage','source_id','source_native_raycast','region_definition']]
    summaries.append(dict(mesh_id=name,mesh_stage=stage,views=len(rows),**{k:float(np.mean([float(r[k]) for r in rows])) for k in columns}))
    print(key,'leaf-region208 complete',flush=True)
csvwrite(OUT/'comparison/leaf_region_mesh_metrics.csv',summaries)
csvwrite(OUT/'comparison/leaf_region_mesh_per_view.csv',allrows)
write_status(OUT/'comparison/LEAF_REGION_MESH_PROTOCOL.json',dict(status='actual_available_complete',routes=len(summaries),rows=len(allrows),elapsed_seconds=time.time()-started,
    definition='Per-view y < floor(ymin+0.6*(ymax-ymin)) from the unchanged RAP foreground mask. Upper region includes background above this line; interior excludes3px foreground boundary. Lower region includes pot/marker and can also contain leaves.',
    limitations='Image-height proxy rather than a verified leaf semantic mask; no new segmentation or front-end inference. Enclosed missing areas use the whole original foreground topology before partitioning, so the split does not create artificial holes. Pixel areas and consistency proxies, not physical3D accuracy.',
    validation='All208 original mask hashes and whole silhouetteIoU verified for every source raycast route. Same masks for Omega/standard source images; each mesh retains its original camera gauge.'))
