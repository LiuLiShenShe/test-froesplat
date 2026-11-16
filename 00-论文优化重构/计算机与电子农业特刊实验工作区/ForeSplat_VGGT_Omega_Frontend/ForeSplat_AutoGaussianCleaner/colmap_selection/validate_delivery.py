"""Audit actual completed experiment artifacts and unchanged historical COLMAP."""
import hashlib,json,io,sys,os,csv,time
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from PIL import Image
from pathlib import Path
import numpy as np
from plyfile import PlyData
from runtime import OUT,Meter,write

def native_vertices(path,meter):
 # Native Gaussian PLY contains fixed scalar vertex fields. Avoid plyfile's
 # Python loop over ~100M scalar values; validate the header then bulk-read
 # the unchanged binary layout from the SAME rate-limited byte payload.
 payload=meter.read(path);stream=io.BytesIO(payload);assert stream.readline().strip()==b'ply'
 types={'float':'f4','float32':'f4','double':'f8','float64':'f8','uchar':'u1','uint8':'u1','int':'i4','int32':'i4'}
 fields=[];element=None;count=None;order=None
 for _ in range(300):
  line=stream.readline().decode('ascii').strip();parts=line.split()
  if not parts:continue
  if parts[0]=='format':
   assert parts[1] in ['binary_little_endian','binary_big_endian'];order='<' if parts[1]=='binary_little_endian' else '>'
  elif parts[0]=='element':
   if count is None:assert parts[1]=='vertex';count=int(parts[2])
   element=parts[1]
  elif parts[0]=='property' and element=='vertex':
   assert len(parts)==3 and parts[1] in types;fields.append((parts[2],order+types[parts[1]]))
  elif line=='end_header':break
 else:raise AssertionError('No binary PLY header terminator')
 assert count and fields and order
 dtype=np.dtype(fields);assert len(payload)>=stream.tell()+count*dtype.itemsize
 return np.frombuffer(payload,dtype=dtype,count=count,offset=stream.tell())

def main():
 meter=Meter();checks=[];start=time.perf_counter()
 source=json.loads((OUT/'common_input/colmap_source.json').read_text())
 for path,record in source['source_evidence'].items():
  p=Path(path);assert p.stat().st_size==record['size_bytes'];assert p.stat().st_mtime_ns==record['mtime_ns']
  assert hashlib.sha256(meter.read(p)).hexdigest()==record['sha256'];checks.append(dict(path=path,status='unchanged'))
 for row in json.loads((OUT/'common_input/image_manifest.json').read_text()):
  p=Path(row['source']['path']);assert p.stat().st_size==row['source']['size_bytes'] and p.stat().st_mtime_ns==row['source']['mtime_ns']
  assert hashlib.sha256(meter.read(p)).hexdigest()==row['source_sha256']
  assert hashlib.sha256(meter.read(OUT/'common_input/dataset/images'/row['name'])).hexdigest()==row['image_sha256']
 print('VALIDATED real COLMAP and210 immutable source/prepared RGB',flush=True)
 names=(OUT/'common_input/train_views.txt').read_text().splitlines()+(OUT/'common_input/test_views.txt').read_text().splitlines()
 assert len(names)==len(set(names))==210
 mask=json.loads((OUT/'common_input/rap_masks_complete.json').read_text());assert mask['count']==210 and mask['status']=='complete'
 for row in json.loads((OUT/'common_input/rap_mask_manifest.json').read_text()):
  assert hashlib.sha256(meter.read(OUT/'common_input/dataset/rap_masks'/f"{Path(row['name']).stem}.png")).hexdigest()==row['mapped_sha256']
  assert hashlib.sha256(meter.read(Path(row['source']))).hexdigest()==row['source_sha256']
 methods=['3DGS','2DGS','GGGS','gsplat','MCMC_3DGS'];models=[]
 for m in methods:
  folder=OUT/'gaussian'/m;record=json.loads((folder/'complete.json').read_text());assert record['iterations']==30000
  verified=json.loads((folder/'verified_input.json').read_text());assert verified['seed']==0 and verified['train']==183 and verified['test']==27 and not verified['mask_loss'];assert verified['initial_gaussians']==177918
  p=Path(record['native_ply']);assert p.stat().st_size==record['model_bytes']
  assert Path(record['checkpoint']).stat().st_size>0
  data=native_vertices(p,meter);assert len(data)==record['gaussian_count']
  assert all(np.isfinite(data[k]).all() for k in data.dtype.names)
  assert len([n for n in data.dtype.names if n.startswith('scale_')])==(2 if m=='2DGS' else 3)
  assert len([n for n in data.dtype.names if n.startswith('f_rest_')])==45
  if m=='GGGS':
   assert 'filter_3D' in data.dtype.names
   clean=native_vertices(OUT/'cleaning/GGGS/cleaned_gaussians.ply',meter)
   idx=np.load(io.BytesIO(meter.read(OUT/'cleaning/GGGS/retained_indices.npy')))
   assert len(idx)==len(clean)==1352835 and clean.dtype==data.dtype and np.all(np.diff(idx)>0)
   assert all(np.array_equal(clean[k],data[k][idx]) for k in data.dtype.names)
   del clean,idx
  evaluation=json.loads((folder/'evaluation/evaluation_complete.json').read_text());assert evaluation['views']==27
  assert set(evaluation['geometry_views'])=={Path(n).stem for n in (OUT/'common_input/test_views.txt').read_text().splitlines()}
  for name in evaluation['geometry_views']:
   with np.load(io.BytesIO(meter.read(folder/'evaluation'/f'{name}_geometry.npz'))) as d:
    z,valid=d['depth_camera_z'],d['valid'];assert z.shape==valid.shape==(948,530)
    assert np.isfinite(z[valid]).all() and (z[valid]>0).all() and np.isfinite(d['alpha']).all()
    assert np.allclose(d['K'],source['K'],atol=1e-5)
  assert (folder/'centres_3d.html').stat().st_size>0
  for name in ['0000','0048','0096','0144','0192','0208']:
   for channel in ['RGB','Alpha','Depth','Normal_derived']:
    with Image.open(folder/'evaluation'/f'{name}_{channel}.png') as im:assert im.size==(530,948)
  models.append(dict(method=m,vertices=len(data),native_fields=list(data.dtype.names),native_model_valid=True))
  del data
  write(OUT/'summary/validation_progress.json',dict(status='running',native_models_verified=[r['method'] for r in models],elapsed_s=time.perf_counter()-start))
  print('VALIDATED native model/checkpoint/SH3/all27 geometry',m,flush=True)
 baseline=json.loads((OUT/'gaussian/3DGS/complete.json').read_text())
 capacity=json.loads((OUT/'gaussian/MCMC_3DGS/capacity_lock.json').read_text());assert capacity['fixed_before_training'] and capacity['cap_max']==baseline['gaussian_count']
 selected=json.loads((OUT/'summary/selected_models.json').read_text())
 meshes=[];input_hashes={}
 for model in selected['mesh_model_keys']:
  surface=OUT/'mesh'/f'{model}_surface';info=json.loads((surface/'complete.json').read_text())
  assert info['input_views']==183 and info['cached_views']==210
  assert set(info['input_view_names'])=={Path(n).stem for n in (OUT/'common_input/train_views.txt').read_text().splitlines()}
  input_hashes[model]=hashlib.sha256(meter.read(surface/'nksr_input.npz')).hexdigest()
  with np.load(io.BytesIO(meter.read(surface/'nksr_input.npz'))) as d:
   assert set(d.files)=={'xyz','normal','sensor','rgb'}
   assert all(d[k].shape==(info['points'],3) and np.isfinite(d[k]).all() for k in d.files)
   assert np.allclose(np.linalg.norm(d['normal'],axis=1),1,atol=1e-4)
  for method in ['NKSR','TSDF','Poisson']+(['native'] if model=='GGGS' else []):
   folder=OUT/'mesh'/method/model;assert json.loads((folder/'complete.json').read_text())['status']=='complete'
   assert json.loads((folder/'evaluation/complete.json').read_text())['views_per_stage']==27
   assert all((folder/f'{stage}.ply').exists() and (folder/f'{stage}_3d.html').exists() and (folder/'evaluation'/f'{stage}_four_views.png').exists() for stage in ['raw','post'])
   record=json.loads((folder/'complete.json').read_text())
   if method!='native':assert record['oriented_input_sha256']==input_hashes[model]
   if method=='native':assert record['train_views']==183 and record['test_views_excluded']==27
   if method=='TSDF':assert record['parameters']['frames']==183 and record['parameters']['held_out_test_views']==27
   assert len(list(csv.DictReader((folder/'evaluation/per_view.csv').open())))==54
   for row in json.loads((folder/'evaluation/metrics.json').read_text()):
    assert all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
    counts={}
    with (folder/f"{row['stage']}.ply").open('rb') as stream:
     for _ in range(300):
      line=stream.readline().decode('ascii').strip()
      if line.startswith('element '):
       _,key,value=line.split();counts[key]=int(value)
      if line=='end_header':break
    assert counts['vertex']==row['vertices'] and counts['face']==row['faces']
    for view in ['0000','0048','0096','0144']:
     for channel in ['RGB','Normal','Silhouette']:
      with Image.open(folder/'evaluation'/f"{row['stage']}_{view}_{channel}.png") as im:assert im.size==(530,948)
   pf=folder/'mesh' if method=='NKSR' else folder
   post=json.loads((pf/'postprocess.json').read_text());assert '<50' in post['rule'] and post['raw_is_preserved']
   assert (pf/'components_and_removals.npz').stat().st_size>0
   meshes.append(dict(model=model,method=method,status='complete'))
   print('VALIDATED Raw/Post mesh/all27 projections',model,method,flush=True)
 rows=list(csv.DictReader((OUT/'summary/mesh_comparison.csv').open()));assert len(rows)==14
 assert len({(r['model'],r['method'],r['stage']) for r in rows})==14
 assert all(all(r[k]!='' for k in ['vertices','faces','silhouette_iou','foreground_coverage','depth_mae_normalized','extraction_process_wall_s']) for r in rows)
 assert len(list(csv.DictReader((OUT/'summary/gaussian_comparison.csv').open())))==5
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.items=[]
  def handle_starttag(self,tag,attrs):self.items.extend(v for k,v in attrs if k in ['href','src'] and v)
 parser=Links();parser.feed((OUT/'summary/index.html').read_text());links=0
 for link in parser.items:
  url=urlsplit(link)
  if url.scheme or not url.path:continue
  p=OUT/'summary'/unquote(url.path);assert p.exists(),str(p);links+=1
 for name,expected in json.loads((OUT/'summary/wrapper_sha256.json').read_text()).items():
  assert hashlib.sha256(meter.read(OUT/'scripts'/name)).hexdigest()==expected
 required=['gaussian_comparison.csv','gaussian_overview.png','cleaning_decision.md','mesh_comparison.csv','mesh_overview.png','final_selection.md','index.html']
 assert all((OUT/'summary'/n).is_file() and (OUT/'summary'/n).stat().st_size>0 for n in required)
 write(OUT/'summary/DELIVERY_VALIDATION.json',dict(status='passed',wall_s=time.perf_counter()-start,historical_source_checks=checks,immutable_rgb_images=210,mapped_rap_masks=210,gaussians=models,cleaned_GGGS_native_fields_bitwise_equal=True,meshes=meshes,mesh_csv_rows=14,local_index_links=links,oriented_input_sha256=input_hashes,required_reports=required,io=meter.stats(),limitation='Artifact validity and provenance checks do not certify biological or metric geometric accuracy; quality limitations and missing2DGS external device peak are explicitly reported'))
 print('DELIVERY_VALIDATION passed',flush=True)

if __name__=='__main__':main()
