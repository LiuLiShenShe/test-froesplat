"""Read existing validated runtime camera matrices; never infer a new frontend."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
OUT=Path(__file__).resolve().parents[1]
DUAL=OUT.parent/'geometry_rgb_dual_frontend';OLD=OUT.parent/'geometry_mesh_final_selection'
def setup(repo=None):
    repo=repo or DUAL/'camera_interface/gggs_corrected'
    sys.path.insert(0,str(repo))
    from utils.cuda_preload import preload_cuda_ext_deps
    preload_cuda_ext_deps()
    from scene import GaussianModel
    import gaussian_renderer
    return GaussianModel,gaussian_renderer
def cameras(reference=None):
    ref=json.loads((reference or (OLD/'surface_inputs/e10_gggs_surface/camera_runtime.json')).read_text())['records']
    result=[]
    for uid,r in enumerate(ref):
        e=np.asarray(r['world_view_transform']).T
        c=argparse.Namespace(uid=uid,image_name=r['name'],image_width=r['width'],image_height=r['height'],FoVx=2*math.atan(r['width']/(2*r['Fx'])),FoVy=2*math.atan(r['height']/(2*r['Fy'])),Fx=r['Fx'],Fy=r['Fy'],Cx=r['Cx'],Cy=r['Cy'],R=torch.tensor(e[:3,:3].T,dtype=torch.float32,device='cuda'),T=torch.tensor(e[:3,3],dtype=torch.float32,device='cuda'),gt_mask=None)
        for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:setattr(c,k,torch.tensor(r[k],dtype=torch.float32,device='cuda'))
        result.append(c)
    assert len(result)==208
    return result,ref
def mask(c):
    ds=DUAL/'Standard_VGGT_Original_RGB/COLMAP_Corrected'
    return np.asarray(Image.open(ds/'masks'/f'{c.image_name}.png').convert('L').resize((c.image_width,c.image_height),Image.Resampling.NEAREST))>127

