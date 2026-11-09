"""Check the installed official extension and checkpoint before main extraction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import importlib.util,time,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
while importlib.util.find_spec('nksr') is None:time.sleep(10)
import numpy as np
import torch,torch_scatter,nksr
from nksr import ext
from nksr.configs import load_checkpoint_from_url
from surface_protocol import write

def main():
    # Native API smoke checks are only preparation; every required real Mesh follows.
    torch.set_num_threads(4)
    xyz=torch.tensor([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]],device='cuda')
    grid=nksr.SparseFeatureHierarchy if hasattr(nksr,'SparseFeatureHierarchy') else None
    from nksr.svh import SparseFeatureHierarchy
    svh=SparseFeatureHierarchy(voxel_size=.01,depth=4,device=torch.device('cuda:0'))
    svh.build_point_splatting(xyz)
    assert svh.grids[0].num_voxels>0
    # Exercise the compatibility caster with its original 3-coordinate defaults.
    coords=svh.grids[0].active_grid_coords();assert coords.shape[-1]==3
    loaded=load_checkpoint_from_url(str(ROOT/'weights/nksr_ks.pth'));assert 'state_dict' in loaded and len(loaded['state_dict'])>0
    from nksr.configs import get_hparams
    network=nksr.NKSRNetwork(get_hparams('ks'));network.load_state_dict(loaded['state_dict'],strict=True)
    import nksr._C as binary
    write(ROOT/'configs/nksr_environment_ready.json',dict(status='ready',torch=torch.__version__,cuda=torch.version.cuda,nksr=nksr.__version__,
        device=torch.cuda.get_device_name(0),physical_gpu=1,weight_sha256=hashlib.sha256((ROOT/'weights/nksr_ks.pth').read_bytes()).hexdigest(),
        extension_path=str(Path(binary.__file__).resolve()),native_cuda_grid_check=True,checkpoint_loaded=True,checkpoint_matches_official_network_strictly=True,
        commit=subprocess.check_output(['git','-C',str(ROOT/'camera_interface/nksr_official'),'rev-parse','HEAD'],text=True).strip(),compatibility_patch='camera_interface/nksr_pybind_compat.patch'))
    print('NKSR official CUDA grid, 3-element bindings and pretrained checkpoint ready',flush=True)

if __name__=='__main__':main()
