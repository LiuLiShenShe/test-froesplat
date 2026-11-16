"""Local review UI. All human edits are separate from frozen inputs.

Run with the existing uv environment; binds to localhost by default. SAM3 is
loaded only on an explicit Mask request and GPU1 is locked for the request.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
from common import *
import argparse,threading,fcntl,traceback,base64
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs

MUTEX=threading.Lock()
EDIT_FILE=OUT/'annotations/user_edits.json'
MASK_ROOT=OUT/'annotations/user_masks'
AUTHOR='human via local UI'
GEOMETRY=None

def edits():
 return read_json(EDIT_FILE) if EDIT_FILE.exists() else dict(schema=1,author='human via local annotation UI',records={},events=[])

def validate(data):
 if data.get('view') not in CAMERAS:raise ValueError('Unknown view')
 if data.get('leaf_id') not in [f'Leaf_{i:03}' for i in range(1,13)]:raise ValueError('Select one of the twelve Leaf IDs')
 if data.get('box'):
  box=data['box']
  if len(box)!=4 or not(0<=box[0]<box[2]<530 and 0<=box[1]<box[3]<948):raise ValueError('Invalid box')
 for point in data.get('points',[]):
  x,y,label=point
  if not(0<=x<530 and 0<=y<948 and label in [0,1]):raise ValueError('Invalid point')
 for name,points in data.get('landmarks',{}).items():
  if name not in ['base','tip','boundary','plant_base']:raise ValueError('Unknown landmark')
  if not all(len(p)==2 and 0<=p[0]<530 and 0<=p[1]<948 for p in points):raise ValueError('Invalid landmark pixel')
 return data

def lift(view,point):
 global GEOMETRY
 cache=OUT/'plant_only/projection_cache'/f'{view}.npz'
 if not cache.exists():return dict(pixel_xy=point,xyz_units=None,status='No cached surface ray; choose a key view for 3D confirmation')
 with np.load(cache) as d:ids=d['triangle_id'];uv=d['barycentric_uv']
 x,y=np.rint(point).astype(int);fid=int(ids[y,x])
 if fid<0:return dict(pixel_xy=point,xyz_units=None,status='No source mesh intersection')
 if GEOMETRY is None:
  mesh=read_mesh(Meter());GEOMETRY=(np.asarray(mesh.triangles).copy(),np.asarray(mesh.vertices).copy())
 f,v=GEOMETRY;b=np.r_[1-uv[y,x].sum(),uv[y,x]]
 xyz=(v[f[fid]]*b[:,None]).sum(0)
 return dict(pixel_xy=point,xyz_units=xyz.tolist(),original_face_id=fid,barycentric_weights=b.tolist(),status='First visible original mesh intersection; inspect before confirming')

def sam_mask(data):
 if not data.get('points') and not data.get('box'):raise ValueError('Provide a positive point or a box')
 import torch
 from scipy.ndimage import distance_transform_edt
 sys.path.insert(0,str(ROOT/'第三方源码/SAM3-latest'))
 from sam3.model_builder import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 meter=Meter();weight=ROOT/'sam3/sam3.pt';original=torch.load
 def paced(file,*args,**kwargs):
  if hasattr(file,'name') and str(weight) in str(file.name):return original(io.BytesIO(meter.read(weight)),*args,**kwargs)
  return original(file,*args,**kwargs)
 lock=(PRE/'summary/gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 model=None
 try:
  torch.load=paced;torch.set_num_threads(8)
  model=build_sam3_image_model(checkpoint_path=str(weight),load_from_HF=False,device='cuda',enable_inst_interactivity=True)
  torch.load=original;processor=Sam3Processor(model,device='cuda',confidence_threshold=.3)
  points=data.get('points',[]);box=data.get('box')
  if not points and not box:raise ValueError('Provide a positive point or a box')
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=processor.set_image(rgb(data['view'],meter))
   masks,scores,_=model.predict_inst(state,point_coords=np.array([p[:2] for p in points]) if points else None,point_labels=np.array([p[2] for p in points]) if points else None,box=np.array(box) if box else None,multimask_output=True)
  chosen=int(np.argmax(scores));mask=(masks[chosen]>0).astype('uint8')*255
  path=MASK_ROOT/data['leaf_id']/f"{data['view']}.png";png(path,mask,meter)
  return dict(mask=str(path.relative_to(OUT)),score=float(scores[chosen]),mask_pixels=int((mask>0).sum()),physical_gpu=1,io=meter.stats())
 finally:
  torch.load=original
  # Processor/state hold references too; release them before returning GPU1.
  if 'state' in locals():del state
  if 'processor' in locals():del processor
  if model is not None:del model
  import gc
  gc.collect()
  torch.cuda.empty_cache();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def save(data,inference=None):
 data=validate(data);cfg=edits();key=data['leaf_id']+'/'+data['view'];old=cfg['records'].get(key,{})
 record={**old,**data,'author':AUTHOR,'saved_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'human_confirmed':bool(data.get('confirmed')) and AUTHOR=='human via local UI'}
 if inference:record.update(inference)
 record['landmarks_3d']={kind:[lift(data['view'],p) for p in points] for kind,points in data.get('landmarks',{}).items()}
 cfg['records'][key]=record;cfg['events'].append(dict(key=key,action='SAM3 prompt' if inference else 'save review',time=record['saved_at_utc'],positive_points=sum(p[2]==1 for p in data.get('points',[])),negative_points=sum(p[2]==0 for p in data.get('points',[])),box_count=int(bool(data.get('box')))))
 write(EDIT_FILE,cfg)
 if record['human_confirmed'] and record['landmarks_3d'].get('plant_base'):
  base=record['landmarks_3d']['plant_base'][0]
  if base.get('xyz_units'):write(OUT/'annotations/plant_base.json',dict(**base,confirmed=True,view=data['view'],author=record['author']))
 return record

class Handler(BaseHTTPRequestHandler):
 def respond(self,status,content,mime='application/json'):
  body=json.dumps(content,ensure_ascii=False).encode() if mime=='application/json' else content
  self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
 def do_GET(self):
  u=urlparse(self.path);q=parse_qs(u.query)
  try:
   if u.path=='/api/config':
    instances=read_json(OUT/'leaf_instances/leaf_instances.json')['instances']
    return self.respond(200,dict(views=sorted(CAMERAS),leaves=[dict(id=x['leaf_id'],views=x['view_names']) for x in instances],edits=edits(),width=530,height=948))
   if u.path=='/api/image':
    name=q.get('view',[''])[0]
    if name not in CAMERAS:raise ValueError('Unknown view')
    b=io.BytesIO();rgb(name).save(b,format='PNG');return self.respond(200,b.getvalue(),'image/png')
   if u.path=='/api/mask':
    data=validate(dict(view=q.get('view',[''])[0],leaf_id=q.get('leaf_id',[''])[0]));record=edits()['records'].get(data['leaf_id']+'/'+data['view'],{})
    path=OUT/record['mask'] if record.get('mask') else OUT/'leaf_instances'/data['leaf_id']/'masks'/f"{data['view']}.png"
    if not path.exists():return self.respond(404,dict(error='No mask in this view'))
    return self.respond(200,path.read_bytes(),'image/png')
   if u.path in ['/','/annotation.html']:
    return self.respond(200,Path(__file__).with_name('annotation.html').read_bytes(),'text/html; charset=utf-8')
   return self.respond(404,dict(error='Unknown route'))
  except Exception as e:self.respond(400,dict(error=str(e)))
 def do_POST(self):
  try:
   length=int(self.headers.get('Content-Length',0))
   if not 0<length<=200000:raise ValueError('Invalid request size')
   data=validate(json.loads(self.rfile.read(length)))
   if self.path not in ['/api/save','/api/sam']:return self.respond(404,dict(error='Unknown route'))
   with MUTEX:
    inference=sam_mask(data) if self.path=='/api/sam' else None
    result=save(data,inference)
   self.respond(200,result)
  except Exception as e:traceback.print_exc();self.respond(400,dict(error=str(e)))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--validation',action='store_true');args=parser.parse_args()
 if args.validation:
  EDIT_FILE=OUT/'annotations/validation/ui_smoke_edits.json';MASK_ROOT=OUT/'annotations/validation/masks';AUTHOR='agent automated API validation, not human annotation'
 print(f'Annotation UI http://{args.host}:{args.port}; no GPU used until explicit SAM3 request',flush=True)
 ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()
