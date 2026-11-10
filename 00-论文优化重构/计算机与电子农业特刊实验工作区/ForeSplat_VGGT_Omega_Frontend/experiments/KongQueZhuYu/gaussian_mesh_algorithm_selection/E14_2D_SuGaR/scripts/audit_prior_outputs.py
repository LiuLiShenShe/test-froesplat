"""Audit real Metric3D files before native normalization; never alter predictions."""
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main():
    started=time.time()
    completed=json.loads((ROOT/'priors_complete.json').read_text())
    assert completed['status']=='complete' and completed['views']==208
    records=json.loads((ROOT/'prior_index.json').read_text())
    expected=json.loads((ROOT.parent.parent/'geometry_mesh_final_selection/surface_inputs/e10_gggs_surface/render_index.json').read_text())
    assert len(records)==208 and {r['source_id'] for r in records}=={r['source_id'] for r in expected}
    rows=[];problems=[]
    for rec in records:
        depth_path=Path(rec['depth']).resolve(strict=True)
        normal_path=Path(rec['normal']).resolve(strict=True)
        mask_path=(ROOT/'input_colmap/masks'/Path(rec['image']).name).resolve(strict=True)
        depth=np.load(depth_path,mmap_mode='r',allow_pickle=False)
        normal=np.load(normal_path,mmap_mode='r',allow_pickle=False)
        mask=np.asarray(Image.open(mask_path).convert('L'))>127
        width,height=rec['source_size']
        assert depth.shape==(height,width) and normal.shape==(height,width,3) and mask.shape==depth.shape
        assert depth.dtype==normal.dtype==np.dtype('float32')
        length=np.linalg.norm(normal,axis=-1)
        normal_finite=np.isfinite(normal).all(-1)
        row=dict(source_id=rec['source_id'],width=width,height=height,
            depth_path=str(depth_path),normal_path=str(normal_path),mask_path=str(mask_path),
            depth_bytes=depth_path.stat().st_size,normal_bytes=normal_path.stat().st_size,
            depth_nonfinite_pixels=int((~np.isfinite(depth)).sum()),
            normal_nonfinite_pixels=int((~normal_finite).sum()),
            normal_length_nonfinite_pixels=int((~np.isfinite(length)).sum()),
            normal_zero_length_pixels=int((np.isfinite(length)&(length<=1e-12)).sum()),
            foreground_pixels=int(mask.sum()),foreground_nonpositive_depth_pixels=int((mask&(depth<=0)).sum()),
            focal_decanonical_factor=rec['prior_depth_decanonical_factor'],
            RGB_SHA256=rec['image_sha256'])
        for name,region in [('all',np.ones(depth.shape,dtype=bool)),('foreground',mask),('background',~mask)]:
            valid_depth=depth[region&np.isfinite(depth)]
            valid_length=length[region&np.isfinite(length)]
            row[name+'_depth_median']=float(np.median(valid_depth)) if len(valid_depth) else ''
            for quantile,label in [(0,'min'),(.5,'median'),(.99,'p99')]:
                row[name+'_normal_length_'+label]=float(np.quantile(valid_length,quantile)) if len(valid_length) else ''
        if any(row[k] for k in ['depth_nonfinite_pixels','normal_nonfinite_pixels','normal_length_nonfinite_pixels','normal_zero_length_pixels','foreground_nonpositive_depth_pixels']):
            problems.append({k:row[k] for k in ['source_id','depth_nonfinite_pixels','normal_nonfinite_pixels','normal_length_nonfinite_pixels','normal_zero_length_pixels','foreground_nonpositive_depth_pixels']})
        rows.append(row)
        print('Actual E14 prior numerical audit',rec['source_id'],len(rows),'/',len(records),flush=True)
    with (ROOT/'prior_numerical_per_view.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='complete' if not problems else 'failed_numerical_prior',views=len(rows),
        all_source_pixel_shapes_and_float32_verified=True,CPU_only=True,original_predictions_unchanged=True,
        prior_depth_frame='Focal-decanonicalized monocular camera-Z BEFORE per-image VGGT track scale',
        prior_normal_frame='Raw camera normals BEFORE native world rotation and per-pixel normalization',
        full_map_validation_reason='Native normal-prior loss normalizes and uses the complete world-normal map without a RAP/alpha multiplier',
        foreground_depth_validation_reason='RAP-limited depth initialization and per-image sparse-track scaling require positive finite foreground predictions',
        normal_length_statistics_are_not_angular_accuracy=True,scan_ground_truth_available=False,
        problems=problems,elapsed_seconds=time.time()-started,
        source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        per_view_csv=str((ROOT/'prior_numerical_per_view.csv').resolve()))
    (ROOT/'prior_numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if problems:raise RuntimeError('Actual upstream predictions are not numerically usable; files are retained unchanged. Inspect prior_numerical_validation.json.')
    print('Actual208 prior outputs numerically verified',flush=True)

if __name__=='__main__':main()
