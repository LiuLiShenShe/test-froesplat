"""Quantify source53's actual dark foreground and same-pixel source/Mesh depth."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection'
DEST=OUT/'comparison/P4_local_failure';DEST.mkdir(exist_ok=True)
protocol=json.loads((OUT/'comparison/visualizations/gaussian/PROTOCOL.json').read_text())
ref=next(r for r in protocol['records'] if r['method']=='E11' and r['source_id']==53)
ref10=next(r for r in protocol['records'] if r['method']=='E10' and r['source_id']==53)
bounds=ref['ROIs']['leaf_texture'];x0,y0,x1,y1=bounds
with np.load(ref['source_npz']) as z:
    gt=z['gt_rgb'];rgb=z['render_raw'];fg=z['foreground_mask'];alpha=z['alpha']
    expected=z['depth_expected'];median=z['depth_median'];normal=z['native_normal']
with np.load(ref10['source_npz']) as z:
    assert np.array_equal(z['gt_rgb'],gt) and np.array_equal(z['foreground_mask'],fg)
    e10=z['depth_surface']
roi=np.zeros(fg.shape,bool);roi[y0:y1,x0:x1]=True
lum=np.array([.2126,.7152,.0722],np.float32)
mask=roi&fg&((gt@lum)>.15)&((rgb@lum)<.01)
assert mask.any(),'No measured dark foreground candidates under recorded thresholds'
yy,xx=np.nonzero(mask)
native={r['source_id']:r for r in json.loads((OLD/'surface_inputs/e11_2dgs_surface/render_index.json').read_text())}[53]
camera=native['camera'];W2C=np.asarray(camera['world_view_transform']).T
ray=np.stack([(xx-camera['Cx'])/camera['Fx'],(yy-camera['Cy'])/camera['Fy'],np.ones(len(xx))],axis=1)
C2W=np.linalg.inv(W2C)
points=ray*median[yy,xx,None];world=points@C2W[:3,:3].T+C2W[:3,3]
summary=dict(status='complete',source_id=53,ROI=bounds,ROI_foreground_pixels=int((roi&fg).sum()),
    dark_foreground_pixels=len(xx),dark_fraction_of_ROI_foreground=float(mask.sum()/max((roi&fg).sum(),1)),
    selection='Same GT-only historical leaf_texture ROI, actual RAP foreground, GT Rec709 luminance>.15 and E11 native luminance<.01. These are dark reconstruction errors, not automatically holes or validated specular pixels.',
    native_alpha_median=float(np.median(alpha[mask])),native_alpha_gt09_fraction=float(np.mean(alpha[mask]>.9)),
    expected_depth_positive_finite_fraction=float(np.mean(np.isfinite(expected[mask])&(expected[mask]>0))),
    median_depth_positive_finite_fraction=float(np.mean(np.isfinite(median[mask])&(median[mask]>0))),
    E11_expected_vs_median_relative_depth_median=float(np.median(np.abs(expected[mask]-median[mask])/np.maximum(np.abs(median[mask]),1e-8))),
    E11_median_vs_E10_surface_relative_depth_median=float(np.median(np.abs(median[mask]-e10[mask])/np.maximum(np.abs(e10[mask]),1e-8))),
    source_npz=ref['source_npz'],source_camera=native['camera'],Mesh_stages=[])
fields=dict(pixel_x=xx,pixel_y=yy,GT_luminance=(gt@lum)[mask],E11_luminance=(rgb@lum)[mask],E11_alpha=alpha[mask],E11_expected_depth=expected[mask],E11_median_depth=median[mask],E10_native_surface_depth=e10[mask],E11_median_world_x=world[:,0],E11_median_world_y=world[:,1],E11_median_world_z=world[:,2])
for depth in ['expected','median']:
    for stage in ['raw','post']:
        cache=OUT/f'P4_2DGS_Depth_Mesh/{depth}_mask/unified_evaluation/views/{stage}_0053.npz'
        with np.load(cache) as z:hit=z['mesh_silhouette'][mask];mesh_depth=z['mesh_depth'][mask]
        source=expected[mask] if depth=='expected' else median[mask]
        key=depth+'_'+stage
        fields[key+'_hit']=hit.astype(np.uint8);fields[key+'_depth']=mesh_depth
        mesh_world=(ray*mesh_depth[:,None])@C2W[:3,:3].T+C2W[:3,3]
        mesh_world[~hit]=np.nan
        for axis,label in enumerate(['x','y','z']):fields[key+'_world_'+label]=mesh_world[:,axis]
        summary['Mesh_stages'].append(dict(depth=depth,stage=stage,source_cache=str(cache.resolve()),
            coverage_on_native_dark_foreground=float(hit.mean()),
            native_vs_mesh_relative_depth_median=float(np.median(np.abs(mesh_depth[hit]-source[hit])/np.maximum(np.abs(source[hit]),1e-8))) if hit.any() else None))
for depth in ['expected','median']:
    raw=fields[depth+'_raw_depth'];post=fields[depth+'_post_depth']
    valid=(fields[depth+'_raw_hit']>0)&(fields[depth+'_post_hit']>0)
    summary[depth+'_post_farther_gt1percent_common_hit_fraction']=float(np.mean((post[valid]-raw[valid])>raw[valid]*.01))
with (DEST/'source53_same_pixel_spatial.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(fields)
    writer.writerows(zip(*fields.values()))
np.savez_compressed(DEST/'source53_selection.npz',ROI_mask=roi,foreground_mask=fg,dark_foreground_error=mask,normal_world_on_selection=normal[mask])
overlay=(np.clip(gt,0,1)*255).astype(np.uint8);overlay[mask]=[255,0,0]
Image.fromarray(overlay[y0:y1,x0:x1]).save(DEST/'source53_GT_dark_error_overlay.png')
summary['interpretation']='High alpha and continuous finite source depth can represent the wrong dark surface. Raw/post coverage isolates whether those same pixels are geometrically absent; finite depth and agreement with E10 are consistency evidence, not scan truth. Post removes small components only. No claim of a physical3D hole or proven cause from dark RGB alone.'
(DEST/'complete.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Actual same-pixel E11/P4 dark-region audit',len(xx),summary['native_alpha_median'],[(r['depth'],r['stage'],r['coverage_on_native_dark_foreground']) for r in summary['Mesh_stages']],flush=True)
