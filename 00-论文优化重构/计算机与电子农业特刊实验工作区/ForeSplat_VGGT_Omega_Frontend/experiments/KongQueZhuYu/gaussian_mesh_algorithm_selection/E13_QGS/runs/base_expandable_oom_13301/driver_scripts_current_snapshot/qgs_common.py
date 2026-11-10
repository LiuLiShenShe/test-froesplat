"""QGS experiment paths and read-only frozen-input adapter."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[3]
KONG=ROOT.parent.parent
DUAL=KONG/'geometry_rgb_dual_frontend'
HIST=KONG/'geometry_mesh_final_selection'
ADAPTED=ROOT/'adapted'
sys.path.insert(0,str(ADAPTED))
import numpy as np
import torch
from omegaconf import OmegaConf
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def config(iterations=30000,output=None):
 cfg=OmegaConf.load(ADAPTED/'config/base.yaml')
 cfg.case_name='KongQueZhuYu'
 cfg.root_dir=str(DUAL/'Standard_VGGT_Original_RGB/COLMAP_Corrected')
 cfg.model_path=str(output or ROOT/'checkpoints')
 cfg.gs_model.output_dir=str(output or ROOT/'checkpoints')
 cfg.gs_model.load_iteration=False
 cfg.optimizer.iterations=iterations
 cfg.dataset.downsample=0.5
 cfg.dataset.eval=False
 # Native base loss and optimization values stay unchanged.
 cfg.dataset.use_alpha=False
 cfg.dataset.undistortion=False
 cfg.skip_train=False
 cfg.skip_test=True
 cfg.skip_mesh=False
 cfg.skip_cull=True
 cfg.skip_metrics=True
 cfg.render_path=False
 return cfg
def read_only_image_adapter():
 from scene import Scene
 # Preserve exact frozen image tensors and avoid source-tree cv2 recompression.
 def frozen_images(self,images):
  self.img_dir=str(Path(self.root_dir)/images)
  assert len(list(Path(self.img_dir).glob('*')))==208
 Scene.load_images=frozen_images
 return Scene
def camera_record(c):
 d=dict(name=c.image_name,width=c.image_width,height=c.image_height,Fx=c.Fx,Fy=c.Fy,Cx=c.Cx,Cy=c.Cy)
 for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
  d[k]=getattr(c,k).detach().cpu().tolist()
 return d
def validate_cameras(cameras,hash_rgb=True):
 import csv
 refs={r['name']:r for r in json.loads((DUAL/'render_comparison/E10/camera_runtime.json').read_text())['records']}
 expected={int(r['source_id']):r for r in csv.DictReader((DUAL/'render_comparison/E10/per_view_metrics.csv').open())}
 records=[];rng=np.random.default_rng(0);errors=[];ids=[]
 for c in sorted(cameras,key=lambda c:c.image_name):
  r=refs[c.image_name];assert (c.image_width,c.image_height)==(1080,1920)
  for k in ['Fx','Fy','Cx','Cy']:assert abs(getattr(c,k)-r[k])<1e-8,(c.image_name,k,getattr(c,k),r[k])
  for k in ['world_view_transform','projection_matrix','full_proj_transform']:
   assert np.max(np.abs(np.asarray(getattr(c,k).cpu())-r[k]))<1e-7,(c.image_name,k)
  sid=int(c.image_name.split('_')[-1]);ids.append(sid)
  rgb=c.original_image.cpu().numpy().transpose(1,2,0).copy()
  digest=hashlib.sha256(rgb.tobytes()).hexdigest()
  if hash_rgb:assert digest==expected[sid]['gt_sha256'],(sid,digest,expected[sid]['gt_sha256'])
  pc=np.column_stack([rng.uniform(-.4,.4,64),rng.uniform(-.4,.4,64),rng.uniform(.5,2,64)])
  P=np.asarray(c.projection_matrix.cpu()).T;clip=np.column_stack([pc,np.ones(64)])@P.T
  pix=((clip[:,:2]/clip[:,3,None]+1)*[c.image_width,c.image_height]-1)/2
  ideal=pc[:,:2]/pc[:,2,None]*[c.Fx,c.Fy]+[c.Cx,c.Cy]
  errors.append(float(np.max(abs(pix-ideal))))
  record=camera_record(c);record.update(source_id=sid,gt_sha256=digest);records.append(record)
 assert set(ids)==set(range(210))-{44,206}
 return dict(status='passed',views=208,projection_error_max_px=max(errors),K_W2C_P_identical_to_E10=True,all208_GT_hashes_identical_to_E10=hash_rgb,records=records)

