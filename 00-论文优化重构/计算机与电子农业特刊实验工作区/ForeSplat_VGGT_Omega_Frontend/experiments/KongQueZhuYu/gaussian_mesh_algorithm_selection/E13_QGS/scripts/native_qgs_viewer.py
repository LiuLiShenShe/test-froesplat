"""Local interactive viewer of actual native QGS quadratic primitives."""
import qgs_common as q
import argparse,copy,io,json,math,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readColmapCameras
from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary
from scene.cameras import Camera
from gaussian_renderer import render
HTML=Path(__file__).with_name('native_qgs_viewer.html')
def rotation(axis,angle):
 axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
 x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
 return np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)
def png(array):
 assert np.isfinite(array).all(),'Nonfinite native display output; no silent uint8 conversion'
 buf=io.BytesIO();Image.fromarray((np.clip(array,0,1)*255).round().astype(np.uint8)).save(buf,format='PNG');return buf.getvalue()
class Viewer:
 def __init__(self,a):
  self.a=a;self.lock=threading.Lock();self.cache_key=None;self.cache=None
  self.cfg=OmegaConf.load(a.config or q.ROOT/'actual_config.yaml')
  self.native=a.native_ply.resolve() if a.native_ply else q.ROOT/f'checkpoints/point_cloud/iteration_{a.iteration}/point_cloud.ply'
  assert self.native.is_file(),f'Actual native QGS PLY missing: {self.native}'
  self.model=GaussianModel(self.cfg.gs_model);self.model.load_ply(str(self.native))
  assert self.model._scaling.shape[1]==6,'QGS native six-scale representation required'
  exported=self.native.parent/'NATIVE_EXPORT_PROVENANCE.json'
  metadata=json.loads(exported.read_text()) if exported.is_file() else {}
  self.model.active_sh_degree=int(metadata.get('active_sh_degree',self.model.max_sh_degree))
  invalid=torch.zeros(len(self.model.get_xyz),device='cuda',dtype=torch.bool)
  with torch.no_grad():
   for field in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']:
    invalid|=~torch.isfinite(getattr(self.model,field)).reshape(len(invalid),-1).all(dim=1)
  self.invalid_native_primitives=int(invalid.sum().item());del invalid
  self.last_render=None
  ds=Path(self.cfg.root_dir)
  ex=read_extrinsics_binary(str(ds/'sparse/0/images.bin'));intr=read_intrinsics_binary(str(ds/'sparse/0/cameras.bin'))
  infos=readColmapCameras(ex,intr,str(ds/'images'),self.cfg.dataset)
  self.infos={int(c.image_name.split('_')[-1]):c for c in infos}
  scale=json.loads((q.HIST/'configs/standard_spatial_scale.json').read_text())
  self.center=np.asarray(scale['origin'],dtype=float)
  self.bg=torch.full((3,),a.background,device='cuda');self.gray=torch.full((len(self.model.get_xyz),3),.65,device='cuda')
  self.anchors={};torch.set_num_threads(8)
 def anchor(self,sid):
  if sid not in self.anchors:
   info=self.infos[sid];resolution=(self.a.width,round(info.height*self.a.width/info.width))
   self.anchors[sid]=Camera(info,resolution,uid=sid)
  return self.anchors[sid]
 def state(self):
   return dict(method='Native QGS quadratic CUDA renderer',iteration=self.a.iteration,stage=self.a.model_stage or ('external_native_file_not_certified_terminal' if self.a.native_ply else 'terminal' if self.a.iteration==30000 else 'partial'),
    native_model=str(self.native.resolve()),quadratic_primitives=len(self.model.get_xyz),scale_channels=6,
    native_nonfinite_primitives=self.invalid_native_primitives,native_model_unfiltered=True,last_render=self.last_render,
    source_ids=sorted(self.infos),initial_source_id=self.a.source_id,width=self.a.width,
    coordinate_system='Original standard VGGT world gauge; no metric scale calibration',
    normal_coordinates='camera; RGB encodes (unit-normalized native normal+1)/2; any undefined visible normal is magenta and counted; original weighted channel retained in experiment NPZ',depth_coordinates='camera Z in original predicted gauge',
    view_flip='Optional180° world-X rotation about plant center is implemented as the inverse camera transform; native geometry file is unchanged',
    background_RGB=[self.a.background]*3,exposure='QGS native appearance/exposure disabled',server='local stdlib HTTP server',interactive_native_primitive_rendering=True)
 def image(self,params):
  sid=int(params.get('source',self.a.source_id));assert sid in self.infos
  yaw=float(params.get('yaw',0));pitch=float(params.get('pitch',0));zoom=float(params.get('zoom',1))
  assert np.isfinite([yaw,pitch,zoom]).all()
  yaw=max(-180,min(180,yaw));pitch=max(-85,min(85,pitch));zoom=max(.25,min(4,zoom))
  flip=params.get('flip180','0')=='1';mode=params.get('mode','rgb')
  assert mode in ['rgb','gray','alpha','normal','depth']
  key=(sid,round(yaw,3),round(pitch,3),round(zoom,4),flip,mode=='gray')
  with self.lock:
   if key!=self.cache_key:
    c=self.anchor(sid)
    base=np.linalg.inv(np.asarray(c.world_view_transform.cpu()).T.astype(np.float64))
    ry=rotation(-base[:3,1],math.radians(yaw))
    rp=rotation(ry@base[:3,0],math.radians(pitch));rot=rp@ry
    c2w=base.copy();c2w[:3,:3]=rot@base[:3,:3]
    c2w[:3,3]=self.center+rot@(base[:3,3]-self.center)/zoom
    if flip:
     view_rotation=np.diag([1.,-1.,-1.])
     c2w[:3,:3]=view_rotation@c2w[:3,:3]
     c2w[:3,3]=self.center+view_rotation@(c2w[:3,3]-self.center)
    w2c=np.linalg.inv(c2w)
    view=copy.copy(c)
    view.R=w2c[:3,:3].T;view.T=w2c[:3,3]
    view.world_view_transform=torch.tensor(w2c.T,dtype=torch.float32,device='cuda')
    view.full_proj_transform=view.world_view_transform@view.projection_matrix
    view.camera_center=torch.tensor(c2w[:3,3],dtype=torch.float32,device='cuda')
    with torch.no_grad():
     pkg=render(view,self.model,self.cfg.pipeline,self.cfg.optimizer,self.bg,kernel_size=self.cfg.dataset.kernel_size,
       override_color=self.gray if mode=='gray' else None,return_depth=True,return_normal=True)
     self.cache=dict(rgb=pkg['render'].cpu().numpy().transpose(1,2,0).copy(),
       alpha=pkg['render_alpha'].cpu().numpy().squeeze().copy(),
       normal=pkg['render_normal'].cpu().numpy().transpose(1,2,0).copy(),
       depth=pkg['surf_depth'].cpu().numpy().squeeze().copy())
    self.cache_key=key
   d=self.cache
   nonfinite={k:int((~np.isfinite(v)).sum()) for k,v in d.items()}
   self.last_render=dict(source_id=sid,mode=mode,nonfinite_values=nonfinite,native_invalid_primitives_retained=True)
   assert all(nonfinite[k]==0 for k in ['rgb','alpha','depth']),'Actual native RGB/alpha/depth contains nonfinite values'
   if mode in ['rgb','gray']:arr=d['rgb']
   elif mode=='alpha':arr=d['alpha']
   elif mode=='normal':
    nl=np.linalg.norm(d['normal'],axis=-1,keepdims=True)
    defined=np.isfinite(d['normal']).all(axis=-1)&np.isfinite(nl[...,0])&(nl[...,0]>1e-12)
    normal=np.divide(d['normal'],nl,out=np.zeros_like(d['normal']),where=defined[...,None])
    arr=normal*.5+.5;arr[d['alpha']<.01]=0
    invalid_visible=(~np.isfinite(d['normal']).all(axis=-1))&(d['alpha']>=.01)
    arr[invalid_visible]=[1,0,1]
    self.last_render['undefined_visible_normal_pixels']=int(invalid_visible.sum())
   else:
    depth=d['depth'];valid=np.isfinite(depth)&(depth>0)&(d['alpha']>.01)
    lo,hi=np.quantile(depth[valid],[.02,.98]) if valid.any() else (0,1)
    t=np.clip((depth-lo)/max(hi-lo,1e-6),0,1)
    arr=np.stack([np.clip(2*t,0,1),1-np.abs(2*t-1),np.clip(2*(1-t),0,1)],axis=-1)
    arr[~valid]=0
   return png(arr)
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--iteration',type=int,default=30000);p.add_argument('--source-id',type=int,default=0)
 p.add_argument('--native-ply',type=Path);p.add_argument('--config',type=Path)
 p.add_argument('--model-stage',choices=['archived_base_partial','failed_native_state','retained_Mip_partial'])
 p.add_argument('--background',type=float,default=.25);p.add_argument('--width',type=int,default=540);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8767)
 a=p.parse_args();assert 128<=a.width<=1080 and 0<=a.background<=1
 if a.native_ply:assert a.config and a.config.is_file(),'Explicit native file requires its own actual config; no cross-preset display parameters'
 viewer=Viewer(a);assert a.source_id in viewer.infos
 class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
   u=urlparse(self.path)
   try:
    if u.path in ['/','/index.html']:data=HTML.read_bytes();kind='text/html; charset=utf-8'
    elif u.path=='/api/state':data=json.dumps(viewer.state(),ensure_ascii=False).encode();kind='application/json'
    elif u.path=='/api/render':data=viewer.image({k:v[-1] for k,v in parse_qs(u.query).items()});kind='image/png'
    else:self.send_error(404);return
    self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
   except (BrokenPipeError,ConnectionResetError):pass
   except Exception as e:
    print(type(e).__name__,str(e),flush=True)
    data=json.dumps(dict(error=type(e).__name__,message=str(e))).encode()
    self.send_response(500);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 server=ThreadingHTTPServer((a.host,a.port),Handler);server.daemon_threads=True
 port=server.server_address[1]
 q.write(q.ROOT/'native_viewer/server_launch.json',dict(status='running',pid=q.os.getpid(),URL=f'http://{a.host}:{port}',**viewer.state()))
 print(f'Actual native QGS interactive viewer http://{a.host}:{port}',flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
if __name__=='__main__':main()

