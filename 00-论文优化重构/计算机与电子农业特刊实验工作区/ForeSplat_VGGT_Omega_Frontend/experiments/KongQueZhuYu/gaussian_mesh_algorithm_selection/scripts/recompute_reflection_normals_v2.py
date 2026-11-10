"""Normalize sampled native normals and report undefined support explicitly.

Legacy correspondence CSVs are retained byte-for-byte. Only derived normal
metrics are recomputed; RGB queries, native depths and model files are unchanged.
"""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import csv,json,hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from queue_io import read_status,write_status

OUT=Path(__file__).resolve().parents[1]
SOURCE=OUT/'comparison/reflection_analysis';DEST=OUT/'comparison/reflection_normal_v2'
VERSION='unit_pixel_normal_then_unit_bilinear_sample_all_valid_support_v2'

def truthy(x):return str(x).lower()=='true'
def sample(a,uv):
 if a.ndim==2:return map_coordinates(a,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=0)
 return np.column_stack([sample(a[...,i],uv) for i in range(a.shape[-1])])
def normalized_sample(normal,valid_map,uv):
 vector=sample(normal,uv);length=np.linalg.norm(vector,axis=1)
 valid=np.isfinite(vector).all(1)&(length>1e-8)&(sample(valid_map.astype(np.float32),uv)>1-1e-6)
 unit=np.zeros_like(vector);unit[valid]=vector[valid]/length[valid,None]
 return unit,valid
def angle(a,b,valid):
 result=np.full(len(a),np.nan)
 result[valid]=np.degrees(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i',a[valid],b[valid])),0,1)))
 return result
def csvwrite(path,rows):
 temporary=path.with_suffix('.csv.tmp')
 with temporary.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 temporary.replace(path)

def main():
 DEST.mkdir(exist_ok=True)
 # Angle invariance to interpolated-vector length is the mathematical property
 # under audit; zero/undefined vectors remain invalid rather than90-degree data.
 test=np.array([[.2,.4,.1],[0.,0.,0.]])
 length=np.linalg.norm(test,axis=1);unit=test[0]/length[0]
 assert np.degrees(np.arccos(np.clip(np.abs(unit@unit),0,1)))<1e-5 and length[1]==0
 summaries=[];records=[]
 for path in sorted(SOURCE.glob('*_spatial_correspondence.csv')):
  name=path.name.removesuffix('_spatial_correspondence.csv');marker=SOURCE/f'{name}_summary.json'
  info=json.loads(marker.read_text());digest=hashlib.sha256(path.read_bytes()).hexdigest()
  prior=read_status(DEST/f'{name}_complete.json')
  if prior.get('status')=='complete' and prior.get('version')==VERSION and prior.get('legacy_source_sha256')==digest:
   summaries+=json.loads((DEST/f'{name}_summary.json').read_text())['rows'];records.append(prior);continue
  rows=list(csv.DictReader(path.open()));index={int(r['source_id']):r for r in json.loads((Path(info['input'])/'render_index.json').read_text())}
  @lru_cache(maxsize=6)
  def frame(sid):
   item=index[sid]
   with np.load(item['path']) as z:
    normal=z['normal_smallest_axis_aux' if info['auxiliary'] else 'native_normal'].copy()
   if info['normal_coordinates']=='camera':normal=normal@np.asarray(item['camera']['world_view_transform']).T[:3,:3]
   length=np.linalg.norm(normal,axis=-1)
   valid=np.isfinite(normal).all(-1)&np.isfinite(length)&(length>1e-8)
   unit=np.zeros_like(normal);unit[valid]=normal[valid]/length[valid,None]
   return unit,valid
  pairs={}
  for row in rows:pairs.setdefault((int(row['source_id']),int(row['target_id'])),[]).append(row)
  corrected=[]
  for (sid,tid),group in pairs.items():
   uv=np.array([[float(r['x']),float(r['y'])] for r in group]);target=np.array([[float(r['target_x']),float(r['target_y'])] for r in group])
   normal,source_map=frame(sid);target_normal,target_map=frame(tid)
   ns,source_valid=normalized_sample(normal,source_map,uv);nt,target_valid=normalized_sample(target_normal,target_map,target)
   geometry=np.array([truthy(r['valid']) for r in group])
   paired=geometry&source_valid&target_valid;cross=angle(ns,nt,paired)
   local=[];counts=np.zeros(len(group),np.int16)
   for shift in [(3,0),(-3,0),(0,3),(0,-3)]:
    adj,adj_valid=normalized_sample(normal,source_map,uv+shift)
    eligible=geometry&source_valid&adj_valid
    local.append(angle(ns,adj,eligible));counts+=eligible
   local=np.column_stack(local);jump=np.full(len(group),np.nan);ok=counts>0
   jump[ok]=np.nansum(local[ok],axis=1)/counts[ok]
   for i,old in enumerate(group):
    row=dict(old,legacy_unrenormalized_normal_crossview_abs_angle_deg=old.get('normal_crossview_abs_angle_deg',''),legacy_unrenormalized_local_normal_jump3px_deg=old.get('local_normal_jump3px_deg',''),
     normal_crossview_abs_angle_deg=float(cross[i]) if paired[i] else '',local_normal_jump3px_deg=float(jump[i]) if ok[i] else '',
     native_normal_source_valid=bool(source_valid[i]),native_normal_target_valid=bool(target_valid[i]),native_normal_crossview_valid=bool(paired[i]),native_normal_local_valid_directions=int(counts[i]),
     normal_metric_version=VERSION,legacy_source_csv=str(path.resolve()))
    corrected.append(row)
   if len(corrected)%10==0:print('Native normalv2',name,sid,tid,len(corrected),flush=True)
  corrected.sort(key=lambda r:int(r['query_id']));assert len(corrected)==len(rows)
  original={int(r['query_id']):r for r in rows}
  for row in corrected:
   for key in ['source_depth','target_depth','source_world_x','source_world_y','source_world_z','RGB_match_reprojection_px','matched_relative_depth_error','valid']:
    assert row[key]==original[int(row['query_id'])][key],('Non-normal native field changed',name,key)
  target=DEST/path.name;csvwrite(target,corrected)
  result=[]
  for bright in [True,False]:
   selected=[r for r in corrected if truthy(r['brightness_candidate'])==bright];valid=[r for r in selected if truthy(r['valid'])]
   record=dict(method=rows[0]['method'],brightness_candidate=bright,queries=len(selected),valid_native_geometry_queries=len(valid),valid_crossview_normal_queries=sum(r['native_normal_crossview_valid'] for r in selected),
    invalid_source_normal_queries=sum(not r['native_normal_source_valid'] for r in selected),invalid_target_normal_queries=sum(not r['native_normal_target_valid'] for r in selected),version=VERSION,depth_type=rows[0]['depth_type'])
   for key in ['normal_crossview_abs_angle_deg','local_normal_jump3px_deg']:
    values=[float(r[key]) for r in valid if r[key]!='' and np.isfinite(float(r[key]))]
    record[key+'_finite_queries']=len(values);record[key+'_median']=float(np.median(values)) if values else None;record[key+'_p90']=float(np.quantile(values,.9)) if values else None
   result.append(record)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Legacy CSV changed during normal audit'
  write_status(DEST/f'{name}_summary.json',dict(status='complete',rows=result,legacy_source_csv=str(path.resolve()),corrected_source_csv=str(target.resolve()),version=VERSION))
  record=dict(status='complete',method=name,version=VERSION,legacy_source_sha256=digest,corrected_csv=str(target.resolve()),queries=len(corrected),legacy_preserved=True,native_depth_and_RGB_queries_identical=True)
  write_status(DEST/f'{name}_complete.json',record);records.append(record);summaries+=result
  frame.cache_clear();print('Actual native normalv2 complete',name,flush=True)
 csvwrite(DEST/'ALL_NORMAL_METRICS.csv',summaries)
 write_status(DEST/'PROTOCOL.json',dict(status='complete_actual_available',version=VERSION,methods=len(records),records=records,
  rule='Rotate native normals to world; normalize valid pixel vectors; bilinearly sample; require all nonzero interpolation-weight support valid; renormalize sampled direction before abs-dot angle. Undefined/zero/nonfinite normals have blank angles and explicit validity counts.',
  local_jump='Mean over available valid3px shifted directions, with actual count0..4; never replace undefined normal by90degrees.',
  scope='Legacy normal angles and original correspondence CSVs preserved. Only normal sampling changes; exact same RGB queries, model files, native depth/alpha/geometry flags. Refined3D axis AUX is separately marked and is not native2D surface-normal truth.'))
 print('Actual normalized native reflection normals',len(records),'methods',flush=True)

if __name__=='__main__':main()
