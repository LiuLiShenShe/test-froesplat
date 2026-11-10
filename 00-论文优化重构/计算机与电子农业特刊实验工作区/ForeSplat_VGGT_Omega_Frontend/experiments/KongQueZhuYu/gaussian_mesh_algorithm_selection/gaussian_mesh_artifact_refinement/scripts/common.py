import os
os.environ.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MPLBACKEND='Agg')
import json, hashlib, sys, argparse, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT.parent
OLD = SEL.parent/'geometry_mesh_final_selection'
DUAL = SEL.parent/'geometry_rgb_dual_frontend'
SOURCE = DUAL/'E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply'
P2 = SEL/'P2_GGGS_Cleaning/cleaned_gaussians.ply'
sys.path.insert(0, str(SEL/'scripts'))
sys.path.insert(0, str(OLD/'scripts'))
from gggs_runtime import setup, cameras, mask
from surface_protocol import save_npz_fast
def write(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def free_camera(r):
    import torch
    e=np.asarray(r['world_view_transform']).T
    c=argparse.Namespace(uid=0,image_name=r['name'],image_width=r['width'],image_height=r['height'],FoVx=2*math.atan(r['width']/(2*r['Fx'])),FoVy=2*math.atan(r['height']/(2*r['Fy'])),Fx=r['Fx'],Fy=r['Fy'],Cx=r['Cx'],Cy=r['Cy'],R=torch.tensor(e[:3,:3].T,dtype=torch.float32,device='cuda'),T=torch.tensor(e[:3,3],dtype=torch.float32,device='cuda'))
    for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:setattr(c,k,torch.tensor(r[k],dtype=torch.float32,device='cuda'))
    return c
