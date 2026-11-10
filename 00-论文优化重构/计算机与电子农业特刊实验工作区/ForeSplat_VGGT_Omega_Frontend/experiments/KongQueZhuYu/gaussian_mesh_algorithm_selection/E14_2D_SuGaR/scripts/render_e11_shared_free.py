"""E11 actual native2DGS on exact eight shared free cameras; no model conversion or retraining."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',MPLBACKEND='Agg',OMP_NUM_THREADS='8')
import sys,json,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HIST=ROOT.parent.parent/'geometry_mesh_final_selection'
SOURCE=HIST/'camera_interface/2dgs_corrected'
p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,default=ROOT/'visual_views/E11_shared_free')
p.add_argument('--protocol',type=Path,default=ROOT.parent/'comparison/shared_free_views.json')
p.add_argument('--check-inputs-only',action='store_true')
a=p.parse_args()
PLY=HIST/'E11_2DGS_Corrected/checkpoints/point_cloud/iteration_30000/point_cloud.ply'
PROTOCOL=a.protocol
assert PLY.is_file() and SOURCE.is_dir() and PROTOCOL.is_file()
protocol=json.loads(PROTOCOL.read_text());assert len(protocol['views'])>0
properties=[]
with PLY.open('rb') as f:
    while True:
        line=f.readline().decode('ascii','replace').strip()
        if not line:raise ValueError('Incomplete Gaussian PLY header')
        if line.startswith('property '):properties.append(line.split()[-1])
        if line=='end_header':break
assert [x for x in properties if x.startswith('scale_')]==['scale_0','scale_1']
ready=dict(status='ready',experiment='E11',source_ply=str(PLY.resolve()),native_renderer=str((SOURCE/'gaussian_renderer/__init__.py').resolve()),renderer_sha256=hashlib.sha256((SOURCE/'gaussian_renderer/__init__.py').read_bytes()).hexdigest(),protocol=str(PROTOCOL.resolve()),protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),views=len(protocol['views']),output=str(a.out.resolve()),GPU_executed=False,training_depth_ratio=0.,visual_depth_ratio=1.,note='Exact original E11 weights/native2D primitive. Native median for shared visual depth; expected depth separately retained. No E14 prior/loss is applied.')
if a.check_inputs_only:
    (ROOT/('E11_shared_free_ready.json' if PROTOCOL.resolve()==(ROOT.parent/'comparison/shared_free_views.json').resolve() else 'E11_'+PROTOCOL.stem+'_ready.json')).write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(ready,ensure_ascii=False,indent=2));raise SystemExit(0)
a.out.mkdir(parents=True,exist_ok=True)
if (a.out/'complete.json').exists() and json.loads((a.out/'complete.json').read_text())['status']=='complete':
    prior=json.loads((a.out/'complete.json').read_text())
    assert prior['protocol_sha256']==ready['protocol_sha256'],'Choose a separate --out for different camera protocol; completed outputs are preserved'
    raise SystemExit(0)
sys.path.insert(0,str(SOURCE))
import numpy as np,torch
from PIL import Image
from matplotlib import colormaps
from types import SimpleNamespace
from scene import GaussianModel
from gaussian_renderer import render
sys.path.append(str(HIST/'scripts'))
from surface_protocol import save_npz_fast
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def png(path,value):Image.fromarray((np.clip(value,0,1)*255).round().astype(np.uint8)).save(path)
def camera(record):
    E=np.asarray(record['world_view_transform'],dtype=np.float32).T
    c=SimpleNamespace(uid=0,image_name=record['name'],image_width=record['width'],image_height=record['height'],Fx=record['Fx'],Fy=record['Fy'],Cx=record['Cx'],Cy=record['Cy'],FoVx=2*np.arctan(record['width']/(2*record['Fx'])),FoVy=2*np.arctan(record['height']/(2*record['Fy'])),R=E[:3,:3].T,T=E[:3,3])
    for key in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:setattr(c,key,torch.tensor(record[key],dtype=torch.float32,device='cuda'))
    return c,E
scale=json.loads((HIST/'configs/standard_spatial_scale.json').read_text())
model=GaussianModel(3);model.load_ply(str(PLY));model.active_sh_degree=3;torch.set_num_threads(8)
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,depth_ratio=1.)
bg=torch.full((3,),float(protocol['background']),device='cuda')
started=time.time();records=[]
for record in protocol['views']:
    c,E=camera(record);folder=a.out/record['name'];folder.mkdir(exist_ok=True)
    with torch.no_grad():pkg=render(c,model,pipe,bg)
    rgb=pkg['render'].cpu().numpy().transpose(1,2,0)
    alpha=pkg['rend_alpha'].cpu().numpy().squeeze()
    median=pkg['depth_median'].cpu().numpy().squeeze()
    expected=pkg['depth_expected'].cpu().numpy().squeeze()
    world=pkg['rend_normal'].cpu().numpy().transpose(1,2,0)
    normal_camera=world@E[:3,:3].T
    # Raw native normals are alpha-weighted accumulations in world coordinates, not unit surface normals.
    save_npz_fast(folder/'native_outputs.npz',rgb_gray_background=rgb,render_raw=rgb,alpha=alpha,depth_surface=median,depth_median=median,depth_expected=expected,native_normal_world=world,normal_world=world,normal_camera=normal_camera)
    png(folder/'RGB.png',rgb);png(folder/'Alpha.png',alpha)
    png(folder/'Normal.png',normal_camera*.5+.5);png(folder/'Normal_camera.png',normal_camera*.5+.5);png(folder/'Normal_world.png',world*.5+.5)
    for label,depth in [('Depth_median',median),('Depth_expected',expected)]:
        visual=colormaps['turbo'](np.clip(depth/scale['tsdf_depth_trunc'],0,1))[...,:3];visual[alpha<.01]=0
        png(folder/(label+'.png'),visual)
    write(folder/'camera_and_channels.json',dict(camera=record,source_ply=str(PLY.resolve()),source_experiment='E11',RGB='Actual native renderer on shared gray background; original float is unclipped in NPZ; PNG clips0..1',Alpha='Exact native2DGS rend_alpha',Depth='Actual native median and alpha-normalized expected surfel rasterizer depth; camera-axis Z in original VGGT gauge, not mesh raycast',Normal='Exact rend_normal native alpha-weighted world-vector accumulation. normal_camera=normal_world@W2C_rotation.T. PNG Normal is camera coordinates; both maps retained.',depth_visual_range=[0,scale['tsdf_depth_trunc']],training_depth_ratio=0.,visual_depth_ratio=1.,conversion=False))
    records.append(dict(view=record['name'],directory=str(folder.resolve()),native_npz=str((folder/'native_outputs.npz').resolve())))
    print('E11 native shared free view',record['name'],flush=True)
write(a.out/'complete.json',dict(**{k:v for k,v in ready.items() if k not in ['status','GPU_executed']},status='complete',GPU_executed=True,model_count=len(model.get_xyz),background=float(protocol['background']),actual_outputs=records,wall_seconds=time.time()-started,EO=False))

