"""P1 native GGGS Marching Tetrahedra + 10 binary steps; corrected camera/mask adapter."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MPLBACKEND='Agg')
import argparse,importlib.util,sys,json,time,shutil,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
OUT=Path(__file__).resolve().parents[1];DUAL=OUT.parent/'geometry_rgb_dual_frontend';OLD=OUT.parent/'geometry_mesh_final_selection'
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('--source-ply',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
dst=a.out or OUT/'P1_GGGS_Tetrahedra';dst.mkdir(parents=True,exist_ok=True)
if (dst/'complete.json').exists():print('already verified complete');raise SystemExit(0)
repo=DUAL/'camera_interface/gggs_corrected';official=OUT/'P1_GGGS_Tetrahedra/official'
sys.path.insert(0,str(repo));sys.path.insert(1,str(OUT/'P1_GGGS_Tetrahedra/tetra_triangulation'))
from utils.cuda_preload import preload_cuda_ext_deps
preload_cuda_ext_deps()
from scene import Scene,GaussianModel
spec=importlib.util.spec_from_file_location('utils.tetmesh',official/'utils/tetmesh.py');tet=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tet;spec.loader.exec_module(tet)
src=(official/'mesh_extract_tetrahedra.py').read_text()
adapted=src.replace('align_corners=True','align_corners=False').replace('chunk_size = 10000000','chunk_size = 2000000')
adapted_path=dst/'mesh_extract_tetrahedra_adapted.py';adapted_path.write_text(adapted)
spec=importlib.util.spec_from_file_location('native_tetra',adapted_path);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(8)
source=a.source_ply or DUAL/'E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply'
runtime=dst/'runtime_checkpoint';link=runtime/'point_cloud/iteration_30000/point_cloud.ply';link.parent.mkdir(parents=True,exist_ok=True)
if not link.exists():link.symlink_to(source.resolve())
ns=argparse.Namespace(sh_degree=3,sg_degree=0,source_path=str(DUAL/'Standard_VGGT_Original_RGB/COLMAP_Corrected'),model_path=str(runtime),images='images',dataset='',resolution=2,white_background=False,data_device='cpu',eval=False,use_decoupled_appearance=0,disable_filter3D=False,kernel_size=0.,multi_view_num=8,multi_view_max_angle=30,multi_view_min_dis=.01,multi_view_max_dis=1.5)
pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
start=time.time();model=GaussianModel(3,0);model.load_ply(str(source));from gggs_runtime import cameras;cams,_=cameras();assert len(cams)==208
reference=json.loads((OLD/'surface_inputs/e10_gggs_surface/camera_runtime.json').read_text())['records'];ref={r['name']:r for r in reference}
for c in cams:
    mask=np.asarray(Image.open(Path(ns.source_path)/'masks'/f'{c.image_name}.png').convert('L').resize((c.image_width,c.image_height),Image.Resampling.NEAREST))>127
    c.gt_mask=torch.from_numpy(mask.astype(np.float32))[None]
    for k in ['Fx','Fy','Cx','Cy']:assert abs(getattr(c,k)-ref[c.image_name][k])<1e-8
    for k in ['world_view_transform','projection_matrix','full_proj_transform']:assert np.max(np.abs(getattr(c,k).cpu().numpy()-np.asarray(ref[c.image_name][k])))<1e-7
# Post rule is the common historical rule; retain all >=50 triangle components.
def post(mesh,cluster_to_keep=1):
    result=native.o3d.geometry.TriangleMesh(mesh)
    labels,counts,areas=result.cluster_connected_triangles()
    remove=np.asarray(counts)[np.asarray(labels)]<50
    result.remove_triangles_by_mask(remove);result.remove_unreferenced_vertices();result.compute_vertex_normals()
    return result
native.post_process_mesh=post
# Synthetic mask pixel centre check for noncentral intrinsics using the patched grid.
test=torch.zeros((1,1,9,7),device='cuda');test[0,0,3,2]=1
grid=torch.tensor([[[[2*(2+.5)/7-1,2*(3+.5)/9-1]]]],device='cuda')
assert abs(float(torch.nn.functional.grid_sample(test,grid,align_corners=False).item())-1)<1e-6
write(dst/'parameters.json',dict(source_checkpoint=str(source.resolve()),official_commit=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip(),official_source_sha256=hashlib.sha256(src.encode()).hexdigest(),adapted_source_sha256=hashlib.sha256(adapted.encode()).hexdigest(),patches=['align_corners=False agrees with 2*(pixel+.5)/size-1','evaluation batch size2M, same samples and max transmittance','common post preserves all components >=50 triangles'],views=208,binary_search_steps=10,camera_reference=str(OLD/'surface_inputs/e10_gggs_surface/camera_runtime.json'),gpu=1,seed=0,num_gaussians=len(model.get_xyz)))
torch.cuda.reset_peak_memory_stats()
native.marching_tetrahedra_with_binary_search(str(dst),cams,model,pipe,0.,torch.zeros(3,device='cuda'),False,True,False,1)
for name in ['recon_init.ply','recon.ply','recon_post.ply']:assert (dst/name).is_file()
write(dst/'complete.json',dict(status='native_extraction_complete_pending_evaluation',wall_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,raw_mesh=str(dst/'recon_init.ply'),binary_search_mesh=str(dst/'recon.ply'),post_mesh=str(dst/'recon_post.ply')))
print('P1 COMPLETE',flush=True)

