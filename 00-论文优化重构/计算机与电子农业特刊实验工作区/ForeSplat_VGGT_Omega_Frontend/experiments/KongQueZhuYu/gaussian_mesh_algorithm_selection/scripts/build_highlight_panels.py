"""Actual same-pixel native maps around fixed independent RGB brightness queries."""
import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MPLBACKEND='Agg')
import csv,json,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw
from matplotlib import colormaps
from queue_io import write_status

OUT=Path(__file__).resolve().parents[1]
OLD=OUT.parent/'geometry_mesh_final_selection'
DEST=OUT/'comparison/visualizations/highlight_correspondence';DEST.mkdir(exist_ok=True)
QUERY=OUT/'comparison/reflection_analysis/RGB_QUERY_MANIFEST.csv'
METHODS=[
 ('E10',OLD/'surface_inputs/e10_gggs_surface','camera',False),
 ('E11',OLD/'surface_inputs/e11_2dgs_surface','world',False),
 ('E12_raw',OUT/'E12_GGGS_EO/surface_inputs','camera',False),
 ('E12_corrected',OUT/'E12_GGGS_EO/surface_inputs','camera',False),
 ('E13_QGS',OUT/'E13_QGS/surface_inputs','camera',False),
 ('E14_2D_30K',OUT/'E14_2D_SuGaR/stage_renders/final_30000','world',False),
 ('E14_refined_aux',OUT/'E14_2D_SuGaR/stage_renders/refined','world',True)]

def png(array):
 return Image.fromarray(np.rint(np.clip(np.nan_to_num(array),0,1)*255).astype('uint8'))

def marked(array,center):
 result=png(array);d=ImageDraw.Draw(result)
 x,y=center;d.ellipse((x-4,y-4,x+4,y+4),outline='#ff0000',width=1)
 return result

def main():
 assert QUERY.is_file()
 queries=list(csv.DictReader(QUERY.open()))
 chosen=sorted([r for r in queries if r['brightness_candidate'].lower()=='true'],
               key=lambda r:(-float(r['luminance_change']),int(r['query_id'])))[:16]
 assert len(chosen)==16
 raw_audit=OUT/'comparison/reflection_analysis/ORIGINAL_RGB_QUERY_VALIDATION.csv'
 raw_query_rows={}
 if raw_audit.is_file():
  raw_query_rows={int(r['query_id']):r for r in csv.DictReader(raw_audit.open())}
  raw_top=sorted([r for r in raw_query_rows.values() if r['raw_context_candidate'].lower()=='true'],key=lambda r:(-float(r['raw_luminance_change']),int(r['query_id'])))[:16]
  by_query={int(r['query_id']):r for r in queries}
  selected={int(r['query_id']):r for r in chosen}
  selected.update({int(r['query_id']):by_query[int(r['query_id'])] for r in raw_top})
  chosen=[selected[k] for k in sorted(selected)]
 source_manifest={int(r['source_index']):r for r in csv.DictReader((OUT.parent/'geometry_rgb_dual_frontend/shared_manifest.csv').open())}
 @lru_cache(maxsize=8)
 def original_rgb(sid):
  path=Path(source_manifest[sid]['raw_rgb_path']).resolve(strict=True)
  with Image.open(path) as im:
   assert im.size==(2160,3840)
   rgb=np.asarray(im.convert('RGB').resize((1080,1920)),np.float32)/255
  return rgb,str(path),hashlib.sha256(path.read_bytes()).hexdigest()
 available=[]
 for name,root,ncoord,aux in METHODS:
  if not (root/'render_complete.json').is_file():continue
  index={int(r['source_id']):r for r in json.loads((root/'render_index.json').read_text())}
  assert len(index)==208
  available.append((name,root,ncoord,aux,index))
 assert available
 scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text())
 detail_half_range=8*float(scale['tsdf_voxel'])
 @lru_cache(maxsize=8)
 def frame(path,aux):
  with np.load(path) as z:
   result={k:z[k] for k in ['gt_rgb','foreground_mask','alpha','render_raw']}
   if 'render_corrected' in z:result['render_corrected']=z['render_corrected']
   dk='depth_expected_center_aux' if aux else 'depth_surface'
   nk='normal_smallest_axis_aux' if aux else 'native_normal'
   result['depth']=z[dk];result['normal']=z[nk]
  return result
 records=[];images=[]
 for query in chosen:
  qid=int(query['query_id']);folder=DEST/f'query_{qid:06d}';folder.mkdir(exist_ok=True)
  canvas=Image.new('RGB',(150*(len(available)+2),160*10+28),'#202020')
  draw=ImageDraw.Draw(canvas)
  draw.text((4,4),f"Fixed RGB query {qid}; masked delta={float(query['luminance_change']):.4f}; red=RGB match",fill='white')
  for endpoint,sid_key,xkey,ykey in [('source','source_id','x','y'),('target','target_id','target_x','target_y')]:
   sid=int(query[sid_key]);cx=int(round(float(query[xkey])));cy=int(round(float(query[ykey])))
   x1,x2=max(0,cx-64),min(1080,cx+64)
   y1,y2=max(0,cy-64),min(1920,cy+64)
   marker=(cx-x1,cy-y1);truth=None;mask=None
   original,original_path,original_sha=original_rgb(sid)
   raw_crop=original[y1:y2,x1:x2]
   baseline=frame(available[0][4][sid]['path'],False)
   ref_depth=baseline['depth'][y1:y2,x1:x2]
   ref_valid=baseline['foreground_mask'][y1:y2,x1:x2]&(baseline['alpha'][y1:y2,x1:x2]>=.5)&np.isfinite(ref_depth)&(ref_depth>0)
   assert np.any(ref_valid)
   detail_center=float(np.median(ref_depth[ref_valid]))
   for mi,(name,root,ncoord,aux,index) in enumerate(available):
    item=index[sid];data=frame(item['path'],aux)
    gt=data['gt_rgb'][y1:y2,x1:x2];fg=data['foreground_mask'][y1:y2,x1:x2]
    if truth is None:truth=gt;mask=fg
    assert np.array_equal(gt,truth) and np.array_equal(fg,mask),(qid,sid,name,'GT mismatch')
    rgb=data['render_corrected' if name=='E12_corrected' else 'render_raw'][y1:y2,x1:x2]
    alpha=data['alpha'][y1:y2,x1:x2]
    depth=data['depth'][y1:y2,x1:x2]
    normal=data['normal'][y1:y2,x1:x2].copy()
    E=np.asarray(item['camera']['world_view_transform']).T
    if ncoord=='world':normal=normal@E[:3,:3].T
    nl=np.linalg.norm(normal,axis=-1,keepdims=True)
    undefined=(~np.isfinite(normal).all(-1))|(nl[...,0]<1e-8)
    normal/=np.maximum(nl,1e-8)
    colors=(normal+1)/2;colors[alpha<.01]=0
    colors[undefined&(alpha>=.01)]=[1,0,1]
    depths=colormaps['viridis'](np.clip((depth-.15)/(1.57-.15),0,1))[...,:3]
    depths[(alpha<.01)|~np.isfinite(depth)|(depth<=0)]=0
    detail=colormaps['viridis'](np.clip((depth-detail_center+detail_half_range)/(2*detail_half_range),0,1))[...,:3]
    detail[(alpha<.01)|~np.isfinite(depth)|(depth<=0)]=0
    arrays=[('RGB',rgb),('Alpha',np.repeat(alpha[...,None],3,-1)),
            ('Depth_aux' if aux else 'Depth',depths),('Depth_detail_aux' if aux else 'Depth_detail',detail),('Normal_axis_aux' if aux else 'Normal_camera',colors)]
    for ri,(kind,arr) in enumerate(arrays):
     path=folder/f'{endpoint}_{sid:04d}_{name}_{kind}.png';marked(arr,marker).save(path)
     row=ri+(5 if endpoint=='target' else 0);x=150*(mi+2);y=28+160*row
     draw.text((x+3,y+2),f'{name}',fill='white');draw.text((x+3,y+15),f'{endpoint} {sid} {kind}',fill='white')
     canvas.paste(marked(arr,marker),(x+3,y+30));images.append(str(path.resolve()))
    records.append(dict(query_id=qid,endpoint=endpoint,source_id=sid,method=name,
     source_npz=str(Path(item['path']).resolve()),pixel_window=[x1,y1,x2,y2],
     matched_pixel=[float(query[xkey]),float(query[ykey])],native_pixel_center=marker,
     native_map_auxiliary=aux,GT_crop_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),
     depth_detail_center=detail_center,depth_detail_half_range=detail_half_range,
     original_context_RGB_path=original_path,original_context_file_sha256=original_sha,
     native_normal_source=ncoord,camera=item['camera'],output=str(folder.resolve())))
   for ri in range(5):
    row=ri+(5 if endpoint=='target' else 0);y=28+160*row
    draw.text((3,y+2),f'02-FFT source {sid}',fill='white')
    canvas.paste(marked(raw_crop,marker),(3,y+30))
    draw.text((153,y+2),f'03-SAM train {sid}',fill='white')
    canvas.paste(marked(truth,marker),(153,y+30))
   for label,rgb in [('GT',truth),('Original_RGB_context',raw_crop)]:
    path=folder/f'{endpoint}_{sid:04d}_{label}.png';marked(rgb,marker).save(path);images.append(str(path.resolve()))
  path=DEST/f'query_{qid:06d}_all_native_maps.png';canvas.save(path);images.append(str(path.resolve()))
  print('Actual brightness-query panels',qid,'methods',len(available),flush=True)
 write_status(DEST/'PROTOCOL.json',dict(status='actual_available_highlight_panels',
  fixed_query_manifest=str(QUERY.resolve()),manifest_sha256=hashlib.sha256(QUERY.read_bytes()).hexdigest(),
  selected_query_ids=[int(q['query_id']) for q in chosen],methods=[m[0] for m in available],
  selection='Union of top16 frozen masked-training brightness candidates and, when audited, top16 actual original02-FFT brightness candidates; all use existing fixed image-only query IDs, no model-based selection',
  pixel_crop='128 original training pixels around nearest integer match; red marker; boundary clipped only at image edge',
  normal_display='Unit camera normal using exact saved W2C; undefined visible normals are magenta; refined3D smallest-axis AUX clearly separate',
  depth_display='Shared .15..1.57 uncalibrated VGGT gauge; native surface depth and refined3D AUX are distinct',
  depth_detail_display='Same per-endpoint E10 valid RAP patch median +/-8 shared TSDF voxels for all models; bounds stored in every record; no per-model rescaling',
  GT_column='Actual unmasked02-FFT source RGB and actual03-SAM masked training RGB are separate columns, repeated only as RGB context; no GT depth/normal measurement',
  raw_context_query_audit=str(raw_audit.resolve()) if raw_audit.is_file() else None,
  interpretation='Brightness is a reflection/shadow/exposure/registration candidate. Same-pixel spatial native/Mesh correspondence CSVs are authoritative; red markers are RGB matches, not scanned 3D truth.',
  records=records,images=images,all_paths_exist=all(Path(p).is_file() for p in images)))

if __name__=='__main__':main()
