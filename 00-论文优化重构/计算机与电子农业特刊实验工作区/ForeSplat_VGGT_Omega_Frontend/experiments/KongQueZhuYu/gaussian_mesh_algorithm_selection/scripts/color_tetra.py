"""Official GGGS vertex colour only, separate file; no retriangulation or smoothing."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MPLBACKEND='Agg')
import argparse,sys,ast,json,time,hashlib
from pathlib import Path
import numpy as np,torch,open3d as o3d
from tqdm import tqdm
from gggs_runtime import OUT,DUAL,setup,cameras,mask
sys.path.insert(0,str(OUT.parent/'geometry_mesh_final_selection/scripts'))
from surface_protocol import write
p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--ply',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
complete=a.out.with_suffix('.colour.json')
if complete.is_file() and a.out.is_file():raise SystemExit(0)
GaussianModel,native=setup();model=GaussianModel(3,0);model.load_ply(str(a.ply));cams,_=cameras();torch.set_num_threads(8)
for c in cams:c.gt_mask=torch.from_numpy(mask(c).astype(np.float32))[None]
official=OUT/'P1_GGGS_Tetrahedra/official/mesh_extract_tetrahedra.py'
# Extract verbatim native colour functions; avoid invoking its extraction entrypoint.
tree=ast.parse(official.read_text());wanted={'evaluation_validation','add_color','add_color_silhouette'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in wanted]
code=ast.unparse(ast.Module(body=nodes,type_ignores=[])).replace('align_corners=True','align_corners=False').replace('chunk_size = 10000000','chunk_size = 2000000')
namespace=dict(torch=torch,tqdm=tqdm,evaluate_transmittance=native.evaluate_transmittance,evaluate_sdf=native.evaluate_sdf,evaluate_color=native.evaluate_color)
exec(compile(code,str(official),'exec'),namespace)
mesh=o3d.io.read_triangle_mesh(str(a.mesh));v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles)
before=hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest();points=torch.tensor(v,dtype=torch.float32,device='cuda');pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
start=time.time();torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    color,surface=namespace['add_color'](points,cams,model,pipe,0.,bg,.008)
    missing=~surface
    if missing.any():
        c2,s2=namespace['add_color'](points[missing],cams,model,pipe,0.,bg,.016);color[missing]=c2;surface[missing]=s2
    if (~surface).any():color[~surface]=namespace['add_color_silhouette'](points[~surface],cams,model,pipe,0.,bg)
colors=color.nan_to_num(nan=.5,posinf=1,neginf=0).clamp(0,1).cpu().numpy();mesh.vertex_colors=o3d.utility.Vector3dVector(colors)
a.out.parent.mkdir(parents=True,exist_ok=True);assert o3d.io.write_triangle_mesh(str(a.out),mesh)
check=o3d.io.read_triangle_mesh(str(a.out));after=hashlib.sha256(np.asarray(check.vertices).tobytes()+np.asarray(check.triangles).tobytes()).hexdigest();assert before==after
write(complete,dict(status='complete',source_mesh=str(a.mesh.resolve()),source_gaussian=str(a.ply.resolve()),output=str(a.out.resolve()),vertices=len(v),faces=len(f),geometry_sha256_before=before,geometry_sha256_after=after,geometry_unchanged=True,color='Official native GGGS add_color / silhouette fallback, RAP constrained, maxdist.008/.016; same corrected pixels and 2M memory batches',views=208,surface_colour_support_fraction=float(surface.float().mean()),EO_external_correction=False,UV_texture=False,wall_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,official_source=str(official.resolve()),source_sha256=hashlib.sha256(official.read_bytes()).hexdigest()))
sys.stdout.flush();sys.stderr.flush();os._exit(0)
