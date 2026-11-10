"""View native QGS primitives through the official renderer, with valid cameras."""
import qgs_common as q
import argparse
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.render_utils import generate_path
def png(path,array):
 Image.fromarray((np.clip(array,0,1)*255).round().astype(np.uint8)).save(path)
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--iteration',type=int,default=30000)
 p.add_argument('--ids',default='0,26,53,79,105,131,157,183')
 p.add_argument('--orbit-frames',type=int,default=0)
 a=p.parse_args()
 cfg=OmegaConf.load(q.ROOT/'actual_config.yaml')
 assert (q.ROOT/f'checkpoints/point_cloud/iteration_{a.iteration}/point_cloud.ply').is_file()
 Scene=q.read_only_image_adapter()
 model=GaussianModel(cfg.gs_model);scene=Scene(cfg,model,load_iteration=a.iteration,shuffle=False)
 views=sorted(scene.getTrainCameras(),key=lambda c:c.image_name)
 ids=set(int(x) for x in a.ids.split(',') if x.strip())
 chosen=[(f'source_{int(c.image_name.split("_")[-1]):04d}',c) for c in views if int(c.image_name.split('_')[-1]) in ids]
 if a.orbit_frames:
  assert a.orbit_frames>=4
  chosen += [(f'orbit_{i:04d}',c) for i,c in enumerate(generate_path(views,n_frames=a.orbit_frames))]
 out=q.ROOT/'native_viewer'/f'iteration_{a.iteration}';out.mkdir(parents=True,exist_ok=True)
 manifest=[];bg=torch.zeros(3,device='cuda')
 with torch.no_grad():
  for name,c in chosen:
   pkg=render(c,model,cfg.pipeline,cfg.optimizer,bg,kernel_size=cfg.dataset.kernel_size,return_depth=True,return_normal=True)
   rgb=pkg['render'].cpu().numpy().transpose(1,2,0)
   alpha=pkg['render_alpha'].cpu().numpy().squeeze()
   normal=pkg['render_normal'].cpu().numpy().transpose(1,2,0)
   nl=np.linalg.norm(normal,axis=-1,keepdims=True);normal=np.divide(normal,nl,out=np.zeros_like(normal),where=nl>1e-12)
   depth=pkg['surf_depth'].cpu().numpy().squeeze()
   png(out/f'{name}_rgb.png',rgb);png(out/f'{name}_alpha.png',alpha)
   png(out/f'{name}_normal_camera.png',normal*.5+.5)
   np.save(out/f'{name}_depth.npy',depth)
   manifest.append(dict(view=name,camera=q.camera_record(c),RGB=str((out/f'{name}_rgb.png').resolve()),alpha=str((out/f'{name}_alpha.png').resolve()),
      normal=str((out/f'{name}_normal_camera.png').resolve()),depth=str((out/f'{name}_depth.npy').resolve())))
   print('native QGS view',name,flush=True)
 q.write(out/'INDEX.json',dict(status='complete',native_model=str((q.ROOT/f'checkpoints/point_cloud/iteration_{a.iteration}/point_cloud.ply').resolve()),
    native_renderer='official QGS quadratic CUDA renderer',coordinate_system='original corrected standard VGGT world gauge; no geometry rotation',views=manifest))
if __name__=='__main__':main()

