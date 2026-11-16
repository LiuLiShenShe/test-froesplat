"""Existing corrected GGGS native tetra extraction, with this COLMAP model only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,importlib.util,io,json,resource,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import open3d as o3d
from PIL import Image
from runtime import OUT,FRONT,CLEANER,HISTORY,SOURCES,Meter,install_io,write


def main():
 p=argparse.ArgumentParser();p.add_argument('--cleaned',action='store_true');a=p.parse_args()
 key='GGGS_Cleaned' if a.cleaned else 'GGGS';output=OUT/'mesh/native'/key;output.mkdir(parents=True,exist_ok=True)
 if (output/'complete.json').exists():return
 meter=Meter();install_io(meter);torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(8)
 sys.path.insert(0,str(SOURCES['GGGS']));sys.path.insert(1,str(HISTORY/'P1_GGGS_Tetrahedra/tetra_triangulation'))
 from evaluate import load_native
 scene,_=load_native('GGGS',OUT/'cleaning/GGGS/cleaned_gaussians.ply' if a.cleaned else None)
 model=scene.gaussians;cameras=scene.getTrainCameras();assert len(cameras)==183
 for camera in cameras:
  mask=np.asarray(Image.open(OUT/'common_input/dataset/rap_masks'/f'{Path(camera.image_name).stem}.png').convert('L'))>127
  camera.gt_mask=torch.from_numpy(mask.astype('float32'))[None].cuda()
 native_path=HISTORY/'P1_GGGS_Tetrahedra/mesh_extract_tetrahedra_adapted.py'
 spec=importlib.util.spec_from_file_location('colmap_native_tetra_existing',native_path);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 assert 'align_corners=False' in native_path.read_text() and 'chunk_size = 2000000' in native_path.read_text()
 sys.path.insert(0,str(FRONT/'experiments/KongQueZhuYu/geometry_mesh_final_selection/scripts'));import mesh_artifacts
 sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import write_mesh
 scale=json.loads((OUT/'mesh'/f'{key}_surface/spatial_scale.json').read_text())
 # The existing adapter already uses the common component rule. Capture its
 # final binary-search geometry in memory, then use the common artifact writer
 # once: no redundant uncoloured intermediate PLY or giant cells.pt.
 captured={}
 def capture(path,mesh,**kw):captured['mesh']=o3d.geometry.TriangleMesh(mesh);return True
 def omitted(mesh,path,*args,**kw):print('OMIT_REDUNDANT_NATIVE_PLY',Path(path).name,flush=True)
 wall=time.perf_counter();cpu=time.process_time();torch.cuda.reset_peak_memory_stats()
 with patch.object(native,'post_process_mesh',lambda mesh,*args,**kw:mesh),patch.object(native.trimesh.Trimesh,'export',omitted),patch.object(native.o3d.io,'write_triangle_mesh',capture):
  native.marching_tetrahedra_with_binary_search(str(output),cameras,model, __import__('argparse').Namespace(debug=False,convert_SHs_python=False,compute_cov3D_python=False),0.,torch.zeros(3,device='cuda'),False,True,False,1)
 mesh=captured['mesh'];torch.cuda.synchronize()
 with np.load(io.BytesIO(meter.read(OUT/'mesh'/f'{key}_surface/nksr_input.npz'))) as d:xyz,rgb=d['xyz'],d['rgb']
 mesh=mesh_artifacts.colour_from_observations(mesh,xyz,rgb)
 original_npz=np.savez_compressed
 def archive(path,**arrays):
  b=io.BytesIO();original_npz(b,**arrays);meter.save(path,b.getbuffer())
 def save(path,mesh,**kw):return write_mesh(path,mesh,meter)
 with patch.object(mesh_artifacts.o3d.io,'write_triangle_mesh',save),patch.object(mesh_artifacts.np,'savez_compressed',archive):stats=mesh_artifacts.save_meshes(mesh,output,scale)
 write(output/'complete.json',dict(status='complete',method='native',model=key,algorithm='Existing GGGS marching tetrahedra with10 binary-search steps, unchanged occupancy samples and native filtered Gaussian parameters',source=str(native_path),source_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),scale=scale,train_views=183,test_views_excluded=27,move_cpu=True,save_cells=False,patches=['Reuse existing half-pixel grid adapter and 2M evaluation batches','Capture native final binary-search mesh in memory; omit duplicate intermediate PLY files','Apply common postprocessing: remove only components<50 triangles','Nearest oriented-observation colours, shared with NKSR/TSDF/Poisson'],wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,io=meter.stats(),**stats))


if __name__=='__main__':
 main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
