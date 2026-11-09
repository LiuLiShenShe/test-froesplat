"""Official pretrained NKSR reconstruction of recorded oriented surface inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,copy,hashlib,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import nksr
from nksr.configs import __configs__
from surface_protocol import ROOT,PROTOCOL,write
from mesh_artifacts import save_meshes,colour_from_observations
from gpu_observation import GPUMonitor

INPUTS={'M1_Omega_NKSR':'omega_direct','M2_Standard_NKSR':'standard_direct','M3_E10_GGGS_NKSR':'e10_gggs_surface',
    'E8_GGGS_NKSR':'e8_gggs_surface','E11_2DGS_NKSR':'e11_2dgs_surface'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=list(INPUTS),required=True);a=p.parse_args()
    out=ROOT/'meshes'/a.experiment if a.experiment.startswith('M') else ROOT/'meshes/supplementary'/a.experiment
    if (out/'complete.json').exists():return
    inp=ROOT/'surface_inputs'/INPUTS[a.experiment];info=json.loads((inp/'complete.json').read_text());scale=info['scale'];d=np.load(inp/'nksr_input.npz')
    xyz=d['xyz'].astype(np.float32);normal=d['normal'].astype(np.float32);sensor=d['sensor'].astype(np.float32);rgb=d['rgb']
    assert np.isfinite(xyz).all() and np.isfinite(normal).all() and np.min(np.linalg.norm(normal,axis=1))>.99
    assert np.all(np.sum(normal*(sensor-xyz),axis=1)>=-1e-6)
    config=copy.deepcopy(__configs__['ks']);weight=ROOT/'weights/nksr_ks.pth';config['url']=str(weight)
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(8);start=time.time();torch.cuda.reset_peak_memory_stats()
    with GPUMonitor(out/'gpu_memory.csv') as memory:
        reconstructor=nksr.Reconstructor(torch.device('cuda:0'),config=config)
        with torch.no_grad():
            field=reconstructor.reconstruct(torch.from_numpy(xyz).cuda(),normal=torch.from_numpy(normal).cuda(),sensor=torch.from_numpy(sensor).cuda(),
                detail_level=None,voxel_size=scale['nksr_voxel'],solver_max_iter=2000,solver_tol=1e-5,approx_kernel_grad=False,fused_mode=True)
            assert field is not None
            extracted=field.extract_dual_mesh(mise_iter=1,trim=True,max_points=1000000)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(extracted.v.cpu().numpy()),o3d.utility.Vector3iVector(extracted.f.cpu().numpy()))
        mesh=colour_from_observations(mesh,xyz,rgb);stats=save_meshes(mesh,out,scale);torch.cuda.synchronize()
    write(out/'complete.json',dict(status='complete',experiment=a.experiment,input=str(inp),input_points=len(xyz),scale=scale,protocol=PROTOCOL,config=config,
        weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),nksr_commit=subprocess.check_output(['git','-C',str(ROOT/'camera_interface/nksr_official'),'rev-parse','HEAD'],text=True).strip(),
        elapsed_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        nvidia_smi_peak_process_mib=memory.peak_mib,nvidia_smi_samples=len(memory.samples),physical_gpu=1,allocator_configuration=os.environ.get("PYTORCH_ALLOC_CONF","native default"),**stats))
    print(a.experiment,stats,flush=True)

if __name__=='__main__':
    main()
    # All CUDA work is synchronized and artifacts/monitor files are closed.
    # Avoid cold-library interpreter teardown after publishing completion.
    import sys
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
