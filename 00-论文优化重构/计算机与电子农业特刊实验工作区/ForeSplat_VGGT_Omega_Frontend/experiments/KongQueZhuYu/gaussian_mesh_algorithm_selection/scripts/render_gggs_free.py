"""Actual GGGS renders on shared novel poses with gray background."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',MPLBACKEND='Agg',OMP_NUM_THREADS='8')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gggs_runtime import OUT,DUAL,setup
sys.path.insert(0,str(OUT.parent/'geometry_mesh_final_selection/scripts'))
from surface_protocol import save_npz_fast,write
p=argparse.ArgumentParser();p.add_argument('--ply',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--protocol',type=Path,default=OUT/'comparison/shared_free_views.json');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
if (a.out/'complete.json').exists():raise SystemExit(0)
GaussianModel,native=setup();model=GaussianModel(3,0);model.load_ply(str(a.ply));torch.set_num_threads(8)
protocol=json.loads(a.protocol.read_text())
pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
bg=torch.full((3,),protocol['background'],device='cuda');start=time.time()
for r in protocol['views']:
    E=np.asarray(r['world_view_transform']).T
    c=argparse.Namespace(uid=0,image_name=r['name'],image_width=r['width'],image_height=r['height'],FoVx=2*np.arctan(r['width']/(2*r['Fx'])),FoVy=2*np.arctan(r['height']/(2*r['Fy'])),Fx=r['Fx'],Fy=r['Fy'],Cx=r['Cx'],Cy=r['Cy'],R=torch.tensor(E[:3,:3].T,dtype=torch.float32,device='cuda'),T=torch.tensor(E[:3,3],dtype=torch.float32,device='cuda'))
    for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:setattr(c,k,torch.tensor(r[k],dtype=torch.float32,device='cuda'))
    with torch.no_grad():pkg=native.render(c,model,pipe,bg,0.)
    rgb=pkg['render'].cpu().numpy().transpose(1,2,0);alpha=pkg['mask'].cpu().numpy().squeeze();depth=pkg['median_depth'].cpu().numpy().squeeze();normal=pkg['normal'].cpu().numpy().transpose(1,2,0)
    d=a.out/r['name'];d.mkdir(exist_ok=True);save_npz_fast(d/'native_outputs.npz',rgb_gray_background=rgb,alpha=alpha,depth_median=depth,normal_camera=normal,normal_world=normal@E[:3,:3])
    for name,image in [('RGB',rgb),('Alpha',np.repeat(alpha[...,None],3,2)),('Normal',normal*.5+.5)]:
        Image.fromarray((np.clip(image,0,1)*255).round().astype(np.uint8)).save(d/f'{name}.png')
    print('actualfreeview',r['name'],flush=True)
write(a.out/'complete.json',dict(status='complete',native_GGGS=True,ply=str(a.ply.resolve()),protocol=str(a.protocol.resolve()),views=len(protocol['views']),wall_seconds=time.time()-start,background=protocol['background'],EO_correction_applied=False))

