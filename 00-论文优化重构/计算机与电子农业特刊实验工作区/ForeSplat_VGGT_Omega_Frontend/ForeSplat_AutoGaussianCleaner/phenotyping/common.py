"""Frozen source geometry and low-I/O helpers for one-plant phenotyping."""
import sys,io,json,os,hashlib,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
ROOT=Path('/data/fj/F2DMAS')
PRE=ROOT/'experiments/COLMAP_Gaussian_Mesh_Selection'
OUT=ROOT/'experiments/COLMAP_Phenotyping/KongQueZhuYu'
FRONT=PRE.resolve().parents[2]
CLEANER=PRE.resolve().parents[1]
RAP=ROOT/'00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本'
sys.path.insert(0,str(PRE/'scripts'))
from runtime import Meter,write as _write
def write(path,data):
 def convert(value):
  if isinstance(value,np.ndarray):return value.tolist()
  if isinstance(value,np.generic):return value.item()
  if isinstance(value,Path):return str(value)
  raise TypeError(type(value).__name__)
 return _write(path,json.loads(json.dumps(data,default=convert,ensure_ascii=False)))
SOURCE=json.loads((PRE/'common_input/colmap_source.json').read_text())
K=np.array(SOURCE['K'])
CAMERAS={}
for row in json.loads((PRE/'common_input/camera_trajectory.json').read_text()):
 q=row['qvec'];r=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix();t=np.array(row['tvec'])
 c2w=np.eye(4);c2w[:3,:3]=r.T;c2w[:3,3]=-r.T@t
 CAMERAS[Path(row['name']).stem]=dict(R=r,t=t,c2w=c2w)

def rgb(name,meter=None,raw=False):
 p=(ROOT/'02-FFT/KongQueZhuYu'/f'{name}.jpg') if raw else (PRE/'common_input/dataset/images'/f'{name}.jpg')
 return Image.open(io.BytesIO(meter.read(p)) if meter else p).convert('RGB')

def png(path,array,meter=None):
 im=Image.fromarray(array);b=io.BytesIO();im.save(b,format='PNG')
 if meter:meter.save(path,b.getbuffer())
 else:Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(b.getvalue())

def project(xyz,name):
 c=CAMERAS[name];cam=np.asarray(xyz)@c['R'].T+c['t'];return cam[:,:2]/cam[:,2,None]*np.diag(K)[:2]+K[:2,2],cam[:,2]

def raw_to_common(points):
 p=SOURCE['source_camera']['params'];xy=(np.array(points)-p[2:4])/p[:2];u=xy.copy()
 for _ in range(15):
  r2=(u*u).sum(1);rad=1+p[4]*r2+p[5]*r2*r2
  dx=2*p[6]*u[:,0]*u[:,1]+p[7]*(r2+2*u[:,0]**2)
  dy=p[6]*(r2+2*u[:,1]**2)+2*p[7]*u[:,0]*u[:,1]
  u=(xy-np.c_[dx,dy])/rad[:,None]
 return u*np.diag(K)[:2]+K[:2,2]

def read_json(path):return json.loads(Path(path).read_text())

def mesh_path():return Path(read_json(PRE/'summary/preferred_mesh.json')['native_ply'])

def read_mesh(meter):
 sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import read_mesh as load
 return load(mesh_path(),meter)

def save_mesh(path,mesh,meter):
 sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import write_mesh
 path=Path(path);assert path.resolve().is_relative_to(OUT.resolve()),'Cannot overwrite any frozen input mesh'
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.stem+'.new'+path.suffix)
 if tmp.exists():tmp.unlink()
 result=write_mesh(tmp,mesh,meter);tmp.replace(path);return result

def npz(path,meter,**data):
 b=io.BytesIO();np.savez_compressed(b,**data);meter.save(path,b.getbuffer())

def subset_mesh(mesh,face_ids):
 import open3d as o3d
 faces=np.asarray(mesh.triangles)[face_ids];vertices,inv=np.unique(faces,return_inverse=True)
 out=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[vertices]),o3d.utility.Vector3iVector(inv.reshape(-1,3)))
 if mesh.has_vertex_colors():out.vertex_colors=o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[vertices])
 if mesh.has_vertex_normals():out.vertex_normals=o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals)[vertices])
 return out,vertices

def cast_scene(mesh):
 import open3d as o3d
 scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));return scene

def cast(scene,name):
 import open3d as o3d
 h,w=SOURCE['height'],SOURCE['width'];y,x=np.mgrid[:h,:w]
 rays=np.stack([(x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],np.ones_like(x)],-1).astype('float32')
 c=CAMERAS[name];direction=rays@c['c2w'][:3,:3].T
 rr=np.concatenate([np.broadcast_to(c['c2w'][:3,3],direction.shape),direction],-1).astype('float32')
 hit=scene.cast_rays(o3d.core.Tensor(rr),nthreads=8)
 ids=hit['primitive_ids'].numpy().astype('int64');depth=hit['t_hit'].numpy();ids[~np.isfinite(depth)]=-1
 return ids,depth,hit['primitive_uvs'].numpy()
