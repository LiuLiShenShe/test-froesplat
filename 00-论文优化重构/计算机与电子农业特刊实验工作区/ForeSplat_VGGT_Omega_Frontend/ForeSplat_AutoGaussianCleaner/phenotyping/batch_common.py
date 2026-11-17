"""Manifest-wide context. Immutable historical sources; centimetres applied once."""
import os,sys,io,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
ROOT=Path('/data/fj/F2DMAS')
CODE=Path(__file__).resolve().parent
CLEANER=CODE.parent
FRONT=CLEANER.parent
PRE=ROOT/'experiments/COLMAP_Gaussian_Mesh_Selection'
OLD=ROOT/'experiments/COLMAP_Phenotyping/KongQueZhuYu'
OUT=ROOT/'experiments/ForeSplat_MultiSpecies_Phenotyping'
MANIFEST=CLEANER/'configs/plant_manifest.json'
sys.path.insert(0,str(CLEANER/'colmap_selection'))
from runtime import Meter
GGGS=FRONT/'gggs_community'
PYTHON=PRE/'.venv-rap/bin/python'
GPU_PYTHON=FRONT/'.venv/bin/python'
UV=Path('/home/test/.local/bin/uv')

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 def default(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,np.generic):return v.item()
  if isinstance(v,Path):return str(v)
  raise TypeError(type(v).__name__)
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp');temp.write_text(json.dumps(value,default=default,ensure_ascii=False,indent=2));temp.replace(path)
def signature(path):
 path=Path(path);s=path.stat();return dict(path=str(path.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def digest(path,meter):return hashlib.sha256(meter.read(path)).hexdigest()
def scalar_ply(payload):
 """Vectorized binary scalar-only PLY; preserve every original field byte."""
 offset=payload.index(b'end_header\n')+len(b'end_header\n');header=payload[:offset];lines=header.decode('ascii').splitlines();formats={'char':'i1','uchar':'u1','short':'i2','ushort':'u2','int':'i4','uint':'u4','float':'f4','double':'f8'}
 assert lines[1]=='format binary_little_endian 1.0'
 elements=[line.split() for line in lines if line.startswith('element ')]
 assert len(elements)==1 and elements[0][1]=='vertex'
 fields=[]
 for line in lines:
  if line.startswith('property '):
   _,kind,name=line.split();fields.append((name,'<'+formats[kind]))
 dtype=np.dtype(fields);count=int(elements[0][2]);assert len(payload)==offset+count*dtype.itemsize
 return header,np.frombuffer(payload,dtype=dtype,count=count,offset=offset)

def install_scalar_ply_io(meter):
 """Avoid Python-per-float BytesIO fallback without changing PLY schema."""
 from plyfile import PlyData,PlyElement
 original_read,original_write=PlyData.read,PlyData.write
 def load(file,*args,**kwargs):
  if isinstance(file,(str,Path)):
   payload=meter.read(file)
   try:
    header,array=scalar_ply(payload);return PlyData([PlyElement.describe(array,'vertex')],text=False,byte_order='<')
   except (AssertionError,ValueError,KeyError):return original_read(io.BytesIO(payload),mmap=False)
  return original_read(file,*args,**kwargs)
 def save(self,file,*args,**kwargs):
  if isinstance(file,(str,Path)) and not self.text and len(self.elements)==1 and self.elements[0].name=='vertex' and not self.elements[0].data.dtype.hasobject:
   payload=self.header.encode('ascii')+b'\n'+self.elements[0].data.astype(self.elements[0].dtype(self.byte_order),copy=False).tobytes();meter.save(file,payload);return
  return original_write(self,file,*args,**kwargs)
 PlyData.read=staticmethod(load);PlyData.write=save
def png(path,array,meter):
 payload=io.BytesIO();Image.fromarray(array).save(payload,format='PNG');artifact(path,payload.getbuffer(),meter)
def npz(path,meter,**arrays):
 payload=io.BytesIO();np.savez_compressed(payload,**arrays);artifact(path,payload.getbuffer(),meter)
def artifact(path,payload,meter):
 path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.partial');meter.save(temporary,payload);temporary.replace(path)
def colmap_io():
 spec=importlib.util.spec_from_file_location('frozen_colmap_io',ROOT/'2d-gaussian-splatting-main/scene/colmap_loader.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def link(path,target):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.is_symlink():
  if path.resolve()!=Path(target).resolve():raise ValueError('Existing source link differs: '+str(path))
 elif path.exists():raise FileExistsError(path)
 else:path.symlink_to(Path(target).resolve(),target_is_directory=Path(target).is_dir())

class Context:
 def __init__(self,plant):
  self.record=next(r for r in read(MANIFEST)['plants'] if r['plant']==plant)
  self.plant=plant;self.out=OUT/plant;self.common=self.out/'common_input';self.raw=ROOT/'02-FFT'/plant;self.annotation=ROOT/'03-GT-区分'/plant;self.meter=Meter()
  self.out.mkdir(parents=True,exist_ok=True)
 def cameras(self,metric=False):
  path=self.out/'gaussian/metric_cameras.json' if metric else self.common/'cameras.json'
  data=read(path);return data,{r['name']:r for r in data['views']}
 def image(self,name):return Image.open(io.BytesIO(self.meter.read(self.common/'images'/f'{name}.jpg'))).convert('RGB')
 def mask(self,name):return np.array(Image.open(self.common/'rap_masks'/f'{name}.png'))>0
 def xyz_project(self,xyz,name,metric=False):
  data,cameras=self.cameras(metric);c=cameras[name];cam=np.asarray(xyz)@np.array(c['R']).T+np.array(c['tvec']);K=np.array(data['K']);return cam[:,:2]/cam[:,2,None]*np.diag(K)[:2]+K[:2,2],cam[:,2]
 def mesh(self):
  sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import read_mesh
  metric=self.out/'mesh/mesh_cm.ply';return read_mesh(metric if metric.exists() else self.out/'mesh/mesh_units.ply',self.meter)
 def save_mesh(self,path,mesh):
  sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import write_mesh
  path=Path(path);assert path.resolve().is_relative_to(self.out.resolve());path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.stem+'.new.ply')
  if temp.exists():temp.unlink()
  write_mesh(temp,mesh,self.meter);temp.replace(path)
 def status(self,stage,status,**detail):
  path=self.out/'status.json';d=read(path) if path.exists() else dict(plant=self.plant,group=self.record['group'],stages={})
  d['stages'][stage]=dict(status=status,updated_unix=time.time(),**detail);d['updated_unix']=time.time();write(path,d)

def subset(mesh,face_ids):
 import open3d as o3d
 f=np.asarray(mesh.triangles)[face_ids];vi,inv=np.unique(f,return_inverse=True);out=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[vi]),o3d.utility.Vector3iVector(inv.reshape(-1,3)))
 if mesh.has_vertex_colors():out.vertex_colors=o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[vi])
 if mesh.has_vertex_normals():out.vertex_normals=o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals)[vi])
 return out,vi

def cast(ctx,mesh,name,metric=False,scene=None):
 import open3d as o3d
 if scene is None:
  scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
 data,cameras=ctx.cameras(metric);c=cameras[name];K=np.array(data['K']);h,w=data['height'],data['width'];y,x=np.mgrid[:h,:w];directions=np.stack([(x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],np.ones_like(x)],-1).astype('float32')@np.array(c['R']);origin=np.array(c['center']);rays=np.concatenate([np.broadcast_to(origin,directions.shape),directions],-1).astype('float32');hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8);ids=hit['primitive_ids'].numpy().astype('int32');z=hit['t_hit'].numpy();ids[~np.isfinite(z)]=-1;return ids,z,hit['primitive_uvs'].numpy(),scene
