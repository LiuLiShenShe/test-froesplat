"""All19 manual references and existing SAM3 point/box reviewer, dynamic Leaf IDs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
from batch_common import *
from batch_measure import csvread,csvwrite
import annotation_server as existing
import threading,traceback,argparse,mimetypes
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import three_leaf_matching as three_leaf
MUTEX=threading.Lock()

def configure(ctx):
 data,cameras=ctx.cameras();catalog=read(ctx.out/'leaf_instances/leaf_instances.json');known={i['leaf_id'] for i in catalog['instances']}
 def validate(value):
  if value.get('view') not in cameras or value.get('leaf_id') not in known:raise ValueError('Unknown registered view or stable Leaf ID')
  w,h=data['width'],data['height']
  if value.get('box'):
   x0,y0,x1,y1=value['box']
   if not 0<=x0<x1<w or not 0<=y0<y1<h:raise ValueError('Invalid box')
  for x,y,label in value.get('points',[]):
   if not 0<=x<w or not 0<=y<h or label not in [0,1]:raise ValueError('Invalid prompt')
  for kind,points in value.get('landmarks',{}).items():
   if kind not in ['base','tip','boundary','plant_base'] or not all(len(p)==2 and 0<=p[0]<w and 0<=p[1]<h for p in points):raise ValueError('Invalid landmark')
  return value
 def lift(view,point):
  cache=ctx.out/'plant_only/projection_cache'/f'{view}.npz'
  if not cache.exists():return dict(pixel_xy=point,xyz_units=None,status='No cached surface ray in this view')
  with np.load(cache) as d:ids=d['triangle_id'];uv=d['uv']
  x,y=np.rint(point).astype(int);fid=int(ids[y,x]);metric=(ctx.out/'mesh/mesh_cm.ply').exists()
  if fid<0:return dict(pixel_xy=point,xyz_units=None,status='No original mesh intersection')
  mesh=ctx.mesh();v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);weights=np.r_[1-uv[y,x].sum(),uv[y,x]];xyz=(v[f[fid]]*weights[:,None]).sum(0)
  return dict(pixel_xy=point,xyz_units=xyz.tolist(),unit='cm' if metric else 'model_units',original_face_id=fid,barycentric_weights=weights.tolist(),status='First visible original mesh surface; inspect identity before confirmation')
 existing.OUT=ctx.out;existing.EDIT_FILE=ctx.out/'annotations/user_edits.json';existing.MASK_ROOT=ctx.out/'annotations/user_masks';existing.rgb=lambda name,meter=None:ctx.image(name);existing.validate=validate;existing.lift=lift
 return data,cameras,catalog

class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(OUT),**kwargs)
 def send_json(self,value,status=200):
  payload=json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
 def content(self,payload,mime):
  self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
 def context(self,q):return Context(q.get('plant',['KongQueZhuYu'])[0])
 def do_GET(self):
  u=urlparse(self.path);q=parse_qs(u.query)
  try:
   if u.path=='/':self.send_response(302);self.send_header('Location','/summary/leaf_mesh_index.html');self.end_headers();return
   if u.path=='/match':return self.content((CODE/'three_leaf_match.html').read_bytes(),'text/html; charset=utf-8')
   if u.path=='/api/match/plants':
    rows=three_leaf.ensure_matches();return self.send_json(dict(plants=[dict(plant=r['plant'],group=r['group'],confirmed=sum(x['human_confirmed']=='True' for x in rows if x['plant_id']==r['plant'])) for r in read(MANIFEST)['plants']]))
   if u.path=='/api/match/config':
    plant=q.get('plant',[''])[0];state=three_leaf.current_state(plant);view=q.get('view',[''])[0];number=str(q.get('number',['1'])[0]);row=next(x for x in state['rows'] if x['handwritten_number']==number);return self.send_json(dict(plant=plant,views=three_leaf.views(plant),row=row,all_rows=state['all_rows']))
   if u.path=='/api/match/pixel':
    return self.send_json(three_leaf.pixel_candidates(q.get('plant',[''])[0],q.get('view',[''])[0],float(q.get('x',['0'])[0]),float(q.get('y',['0'])[0]),int(q.get('radius',['48'])[0])))
   if u.path=='/api/match/leaf':
    return self.send_json(three_leaf.leaf_detail(q.get('plant',[''])[0],q.get('leaf_id',[''])[0],q.get('view',[''])[0]))
   if u.path=='/api/match/history':
    plant=q.get('plant',[''])[0];number=int(q.get('number',['1'])[0]);return self.send_json(dict(plant=plant,number=number,candidates=three_leaf.history_candidates(plant,number)))
   if u.path=='/api/match/metric-candidates':
    plant=q.get('plant',[''])[0];number=int(q.get('number',['1'])[0]);history_row_id=q.get('history_row_id',[''])[0]
    return self.send_json(three_leaf.metric_candidates(plant,number,history_row_id))
   if u.path=='/api/match/rgb':
    ctx=self.context(q);name=q.get('view',[''])[0];data,cameras=ctx.cameras()
    if name not in cameras:raise ValueError('Unknown registered view')
    buffer=io.BytesIO();ctx.image(name).save(buffer,format='PNG');return self.content(buffer.getvalue(),'image/png')
   if u.path=='/api/match/mask':
    plant=q.get('plant',[''])[0];leaf=q.get('leaf_id',[''])[0];view=q.get('view',[''])[0];ctx=self.context(q)
    catalog=read(ctx.out/'leaf_instances'/'leaf_instances.json')
    item=next((entry for entry in catalog.get('instances',[]) if entry.get('leaf_id')==leaf),None)
    if item is None or view not in set(item.get('views',[])):
     return self.send_json(dict(error='Unknown Leaf ID or unavailable RGB view'),404)
    path=ctx.out/'leaf_instances'/leaf/'masks'/f'{view}.png'
    if not path.exists():return self.send_json(dict(error='No leaf mask in this view'),404)
    return self.content(path.read_bytes(),'image/png')
   if u.path=='/api/match/overview':
    three_leaf.export_three_leaf_tables();return self.send_json(dict(url='/summary/three_leaf_match_overview.html'))
   if u.path=='/review':return self.content((CODE/'batch_review.html').read_bytes(),'text/html; charset=utf-8')
   if u.path=='/api/plants':return self.send_json(dict(plants=[dict(plant=r['plant'],group=r['group'],ready=(OUT/r['plant']/'phenotype/manual_reference.csv').exists()) for r in read(MANIFEST)['plants']]))
   if u.path=='/api/manual':
    ctx=self.context(q);path=ctx.out/'phenotype/manual_reference.csv';return self.send_json(dict(plant=ctx.plant,rows=csvread(path) if path.exists() else [],ready=path.exists()))
   if u.path in ['/annotation','/api/config','/api/image','/api/mask']:
    ctx=self.context(q);data,cameras=ctx.cameras();catalog=read(ctx.out/'leaf_instances/leaf_instances.json');editpath=ctx.out/'annotations/user_edits.json';edits=read(editpath) if editpath.exists() else dict(schema=1,author='human via local annotation UI',records={},events=[])
    if u.path=='/annotation':
     text=(CODE/'annotation.html').read_text().replace('孔雀竹芋',ctx.plant).replace('530',str(data['width'])).replace('529',str(data['width']-1)).replace('948',str(data['height'])).replace('947',str(data['height']-1));plant=ctx.plant
     text=text.replace("'/api/image?view='",f"'/api/image?plant={plant}&view='").replace("'/api/mask?view='",f"'/api/mask?plant={plant}&view='").replace("fetch('/api/config')",f"fetch('/api/config?plant={plant}')").replace('fetch(path,{',f"fetch(path+'?plant={plant}',{{").replace('apply_corrections.py','batch_apply_corrections.py '+plant)
     text=text.replace('<button id="save">','<label><input type="checkbox" id="whole_complete">我已核对整片叶面的边界及可见完整性</label><button id="save">').replace("confirmed:$('confirmed').checked,alias:","confirmed:$('confirmed').checked,whole_leaf_surface_complete_confirmed:$('whole_complete').checked,alias:").replace("$('alias').value=state.alias||'';","$('alias').value=state.alias||'';$('whole_complete').checked=state.whole_leaf_surface_complete_confirmed||false;")
     text=text.replace('</html>',"<script>(function(){const q=new URLSearchParams(location.search);setTimeout(()=>{const leaf=q.get('leaf_id'),view=q.get('view'),ls=document.querySelector('#leaf'),vs=document.querySelector('#view');if(leaf&&Array.from(ls.options).some(o=>o.value===leaf))ls.value=leaf;if(view&&Array.from(vs.options).some(o=>o.value===view))vs.value=view;if(leaf||view)ls.dispatchEvent(new Event('change'))},800)})();</script></html>")
     return self.content(text.encode(),'text/html; charset=utf-8')
    if u.path=='/api/config':return self.send_json(dict(views=sorted(cameras),leaves=[dict(id=i['leaf_id'],views=i['views']) for i in catalog['instances']],edits=edits,width=data['width'],height=data['height']))
    if u.path=='/api/image':
     name=q.get('view',[''])[0]
     if name not in cameras:raise ValueError('Unknown view')
     buffer=io.BytesIO();ctx.image(name).save(buffer,format='PNG');return self.content(buffer.getvalue(),'image/png')
    value=dict(view=q.get('view',[''])[0],leaf_id=q.get('leaf_id',[''])[0])
    if value['view'] not in cameras or value['leaf_id'] not in {item['leaf_id'] for item in catalog['instances']}:raise ValueError('Unknown registered view or stable Leaf ID')
    entry=edits['records'].get(value['leaf_id']+'/'+value['view'],{});path=ctx.out/entry['mask'] if entry.get('mask') else ctx.out/'leaf_instances'/value['leaf_id']/'masks'/f"{value['view']}.png"
    if not path.exists():return self.send_json(dict(error='No instance mask in this view'),404)
    return self.content(path.read_bytes(),'image/png')
   return super().do_GET()
  except Exception as e:self.send_json(dict(error=str(e)),400)
 def do_POST(self):
  u=urlparse(self.path);q=parse_qs(u.query)
  try:
   size=int(self.headers.get('Content-Length',0))
   if not 0<size<=200000:raise ValueError('Invalid request size')
   value=json.loads(self.rfile.read(size));ctx=self.context(q)
   with MUTEX:
    if u.path=='/api/manual':
     path=ctx.out/'phenotype/manual_reference.csv'
     if not path.exists():raise ValueError('Plant traits have not been generated yet')
     table=csvread(path);match=[i for i,r in enumerate(table) if r['leaf_id']==value.get('leaf_id')]
     if len(match)!=1:raise ValueError('Unknown or ambiguous stable Leaf ID')
     row=table[match[0]].copy()
     for key in ['leaf_length_cm','leaf_width_cm','leaf_area_cm2','plant_height_cm','canopy_span_a_cm','canopy_span_b_cm','canopy_width_cm']:
      v=value.get(key,'')
      if str(v).strip():
       number=float(v)
       if not np.isfinite(number) or number<=0:raise ValueError('Manual measurements must be positive finite numbers')
       row[key]=number
      else:row[key]=''
     for key in ['capture_session','measured_at','measurement_tool','operator','notes']:row[key]=str(value.get(key,''))[:2000]
     row['pair_confirmed']=value.get('pair_confirmed') is True;table[match[0]]=row
     backup=ctx.out/'annotations/manual_history'/f'{time.time_ns()}.csv';backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(path.read_bytes());csvwrite(path,table)
     from batch_manual import main
     from batch_report import summarize
     main(ctx);summarize();return self.send_json(dict(saved=True,plant=ctx.plant,leaf_id=row['leaf_id'],audit=read(ctx.out/'phenotype/manual_pairing_audit.json')))
    if u.path=='/api/match/confirm':
     result=three_leaf.confirm(value);return self.send_json(result)
    if u.path not in ['/api/save','/api/sam']:return self.send_json(dict(error='Unknown route'),404)
    configure(ctx);existing.validate(value);inference=existing.sam_mask(value) if u.path=='/api/sam' else None;result=existing.save(value,inference)
    if result.get('human_confirmed') and result.get('landmarks_3d',{}).get('plant_base'):
     base=result['landmarks_3d']['plant_base'][0]
     if base.get('xyz_units'):write(ctx.out/'annotations/plant_base.json',dict(**base,xyz=base['xyz_units'],confirmed=True,author=result['author']))
    return self.send_json(result)
  except Exception as e:traceback.print_exc();self.send_json(dict(error=str(e)),400)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8766);args=parser.parse_args();write(OUT/'summary/review_server.json',dict(pid=os.getpid(),host=args.host,port=args.port,url=f'http://{args.host}:{args.port}',started_unix=time.time()));print(f'All19 mesh/review server http://{args.host}:{args.port}; GPU only on explicit SAM3 requests',flush=True);ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()
