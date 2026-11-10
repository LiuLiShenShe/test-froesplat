"""Audit frozen RGB queries against actual unmasked02-FFT source images."""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import csv,json,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter,map_coordinates,distance_transform_edt
from queue_io import write_status

OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection'
DEST=OUT/'comparison/reflection_analysis'

def read(p):return list(csv.DictReader(p.open()))
def csvwrite(p,rows):
 if not rows:return
 temporary=p.with_suffix('.csv.tmp')
 with temporary.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 temporary.replace(p)
def sample(a,uv):return map_coordinates(a,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=np.nan)
def truthy(value):return str(value).lower()=='true'

def main():
 query_path=DEST/'RGB_QUERY_MANIFEST.csv';queries=read(query_path)
 sources={int(r['source_index']):r for r in read(OUT.parent/'geometry_rgb_dual_frontend/shared_manifest.csv')}
 native={int(r['source_id']):r for r in json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text())}
 source_records={}
 @lru_cache(maxsize=6)
 def frame(sid):
  path=Path(sources[sid]['raw_rgb_path']).resolve(strict=True)
  with Image.open(path) as im:
   original_size=im.size;assert original_size==(2160,3840)
   # Same actual native PIL resize call used by PILtoTorch, rather than cv2.
   rgb=np.asarray(im.convert('RGB').resize((1080,1920)),np.float32)/255
  with np.load(native[sid]['path']) as z:train=z['gt_rgb'].copy();mask=z['foreground_mask'].copy()
  assert hashlib.sha256(train.tobytes()).hexdigest()==native[sid]['gt_sha256']
  assert hashlib.sha256(mask.tobytes()).hexdigest()==native[sid]['mask_sha256']
  source_records[sid]=dict(source_id=sid,raw_rgb_path=str(path),raw_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_original_size=list(original_size),
   training_rgb_path=str(Path(sources[sid]['training_rgb_path']).resolve()),training_foreground_mean_abs_difference_from_raw=float(np.abs(rgb[mask]-train[mask]).mean()),
   training_background_all_channels_zero_fraction=float(np.all(train[~mask]==0,axis=1).mean()),raw_background_all_channels_zero_fraction=float(np.all(rgb[~mask]==0,axis=1).mean()))
  return uniform_filter(rgb.mean(-1),size=5),uniform_filter(train.mean(-1),size=5),distance_transform_edt(mask)
 pairs={}
 for r in queries:pairs.setdefault((int(r['source_id']),int(r['target_id'])),[]).append(r)
 records=[]
 for (sid,tid),group in pairs.items():
  uv=np.array([[float(r['x']),float(r['y'])] for r in group]);target=np.array([[float(r['target_x']),float(r['target_y'])] for r in group])
  raw1,masked1,edge1=frame(sid);raw2,masked2,edge2=frame(tid)
  a=sample(raw1,uv);b=sample(raw2,target);ma=sample(masked1,uv);mb=sample(masked2,target)
  assert np.isfinite(a).all() and np.isfinite(b).all()
  assert np.allclose(ma,[float(r['source_luminance']) for r in group],atol=1e-6)
  assert np.allclose(mb,[float(r['target_luminance']) for r in group],atol=1e-6)
  distance1=sample(edge1,uv);distance2=sample(edge2,target)
  raw_candidate=(np.maximum(a,b)>=.55)&(np.abs(a-b)>=.12)
  for i,r in enumerate(group):
   old=truthy(r['brightness_candidate']);new=bool(raw_candidate[i])
   records.append(dict(query_id=int(r['query_id']),source_id=sid,target_id=tid,legacy_masked_candidate=old,raw_context_candidate=new,
    candidate_relation='both' if old and new else 'masked_only' if old else 'raw_only' if new else 'neither',
    raw_source_luminance=float(a[i]),raw_target_luminance=float(b[i]),raw_luminance_change=float(abs(a[i]-b[i])),
    masked_source_luminance=float(ma[i]),masked_target_luminance=float(mb[i]),masked_luminance_change=float(abs(ma[i]-mb[i])),
    source_luminance_raw_minus_masked=float(a[i]-ma[i]),target_luminance_raw_minus_masked=float(b[i]-mb[i]),
    source_RAP_interior_distance_px=float(distance1[i]),target_RAP_interior_distance_px=float(distance2[i]),
    source_raw_rgb_path=sources[sid]['raw_rgb_path'],target_raw_rgb_path=sources[tid]['raw_rgb_path']))
  print('Original RGB query audit',sid,tid,len(records),flush=True)
 records.sort(key=lambda r:r['query_id']);assert [r['query_id'] for r in records]==list(range(len(queries)))
 csvwrite(DEST/'ORIGINAL_RGB_QUERY_VALIDATION.csv',records)
 csvwrite(DEST/'ORIGINAL_RGB_SOURCE_FILES.csv',[source_records[k] for k in sorted(source_records)])
 groups=dict(legacy_masked_candidates={r['query_id'] for r in records if r['legacy_masked_candidate']},raw_context_candidates={r['query_id'] for r in records if r['raw_context_candidate']},
  common_both={r['query_id'] for r in records if r['candidate_relation']=='both'},masked_only={r['query_id'] for r in records if r['candidate_relation']=='masked_only'},
  raw_only={r['query_id'] for r in records if r['candidate_relation']=='raw_only'},raw_context_controls={r['query_id'] for r in records if not r['raw_context_candidate']})
 metrics=[]
 for path in sorted(DEST.glob('*_spatial_correspondence.csv')):
  corrected=OUT/'comparison/reflection_normal_v2'/path.name
  used=corrected if corrected.is_file() else path
  rows=read(used);index={int(r['query_id']):r for r in rows};assert set(index)==set(range(len(queries)))
  for name,ids in groups.items():
   selected=[index[q] for q in sorted(ids)];valid=[r for r in selected if truthy(r['valid'])]
   result=dict(method=rows[0]['method'],RGB_candidate_group=name,queries=len(selected),valid_native_geometry_queries=len(valid),valid_fraction=len(valid)/len(selected) if selected else '',
    native_geometry_source=str(used.resolve()),depth_type=rows[0]['depth_type'],normal_metric_version=rows[0].get('normal_metric_version','legacy_unit_pixels_unrenormalized_bilinear_sample'))
   for key in ['RGB_match_reprojection_px','matched_relative_depth_error','normal_crossview_abs_angle_deg','local_normal_jump3px_deg','nearest_primitive_anisotropy']:
    values=[float(r[key]) for r in valid if r.get(key) not in ['',None] and np.isfinite(float(r[key]))]
    result[key+'_finite_queries']=len(values);result[key+'_median']=float(np.median(values)) if values else '';result[key+'_p90']=float(np.quantile(values,.9)) if values else ''
   metrics.append(result)
 csvwrite(DEST/'ORIGINAL_RGB_CANDIDATE_GEOMETRY.csv',metrics)
 write_status(DEST/'ORIGINAL_RGB_AUDIT.json',dict(status='complete_actual_available',fixed_queries=len(queries),candidate_group_counts={k:len(v) for k,v in groups.items()},actual_original_files=len(source_records),
  source_file_index=str((DEST/'ORIGINAL_RGB_SOURCE_FILES.csv').resolve()),query_validation=str((DEST/'ORIGINAL_RGB_QUERY_VALIDATION.csv').resolve()),native_subset_metrics=str((DEST/'ORIGINAL_RGB_CANDIDATE_GEOMETRY.csv').resolve()),
  frozen_legacy_manifest_unchanged=True,legacy_manifest_sha256=hashlib.sha256(query_path.read_bytes()).hexdigest(),
  threshold='Same maximum5px meanRGB>=.55 and absolute change>=.12; original raw02-FFT vs masked03-SAM evaluated at identical frozen RGB match coordinates.',
  interpretation='No new feature matching, masks or front-end inference. Masked training background can affect boundary windows. Raw and masked candidate groups are both retained; brightness alone is not certified specular reflection, and geometry quantities remain proxy consistency, not scan error.'))
 print('Actual raw RGB candidate groups',{k:len(v) for k,v in groups.items()},flush=True)

if __name__=='__main__':main()
