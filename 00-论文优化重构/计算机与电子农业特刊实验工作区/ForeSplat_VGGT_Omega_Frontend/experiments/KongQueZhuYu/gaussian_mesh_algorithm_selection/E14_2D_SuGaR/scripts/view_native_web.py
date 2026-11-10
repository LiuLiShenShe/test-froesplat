"""Open actual E14 native PLY stages using their original rasterizer in a local browser; launch only after training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys,json,argparse,base64,io
from pathlib import Path
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np,torch
from PIL import Image
from matplotlib import colormaps
ROOT=Path(__file__).resolve().parents[1];OFF=ROOT/'official'
parser=argparse.ArgumentParser()
parser.add_argument('--stage',choices=['initial','before','pruned','final','refined'],default='final')
parser.add_argument('--port',type=int,default=8864)
parser.add_argument('--ply',type=Path,default=None,help='Actual external native2DGS PLY; representation renderer only, no conversion or retraining')
parser.add_argument('--label',default=None,help='Display label for the actual model, e.g. E11')
args=parser.parse_args()
sys.path.insert(0,str(OFF))
refined=args.stage=='refined'
if refined:
    sys.path.append(str(OFF/'gaussian_splatting'))
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.gaussian_renderer import render
    from utils.general_utils import build_rotation
    if args.ply is None:
        completed=json.loads((ROOT/'refinement_complete.json').read_text())
        assert completed['status']=='complete'
        ply=Path(completed['gaussian_ply'])
    else:ply=args.ply
else:
    sys.path.insert(0,str(OFF/'gaussian_splatting_2d'))
    from scene import GaussianModel
    from gaussian_renderer import render
    ply={'initial':ROOT/'training/point_cloud/iteration_1/point_cloud.ply','before':ROOT/'training/point_cloud/iteration_7000/point_cloud.ply','pruned':ROOT/'cluster_pruning/after.ply','final':ROOT/'training/point_cloud/iteration_30000/point_cloud.ply'}[args.stage]
if args.ply is not None:ply=args.ply.expanduser().resolve(strict=True)
assert ply.is_file(),ply
properties=[]
with ply.open('rb') as f:
    while True:
        line=f.readline().decode('ascii','replace').strip()
        if not line:raise ValueError('Not a native Gaussian PLY header')
        if line.startswith('property '):properties.append(line.split()[-1])
        if line=='end_header':break
scale_properties=[x for x in properties if x.startswith('scale_')]
assert len(scale_properties)==(3 if refined else 2),('Expected native3D scales' if refined else 'Expected native2D surfel two scales',scale_properties)
assert all(x in properties for x in ['opacity','f_dc_0','f_dc_1','f_dc_2','rot_0','rot_1','rot_2','rot_3'])
viewer_title=args.label or ('External native2DGS model' if args.ply is not None and not refined else 'E14 '+args.stage)
viewer_note='External model camera/representation renderer only: training pipeline, priors and loss weights are those of the source experiment, not E14.' if args.ply is not None else 'Actual native E14 model stage; no conversion.'
refs=json.loads((ROOT.parent.parent/'geometry_rgb_dual_frontend/render_comparison/E10/camera_runtime.json').read_text())['records']
base=next(r for r in refs if r['name']=='crop_0000')
scale=json.loads((ROOT.parent.parent/'geometry_mesh_final_selection/configs/standard_spatial_scale.json').read_text())
origin=np.asarray(scale['origin'],dtype=np.float64)
model=GaussianModel(3);model.load_ply(str(ply));model.active_sh_degree=3
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,depth_ratio=1.)
gray=torch.full((3,),.25,device='cuda');black=torch.zeros(3,device='cuda')
if refined:
    rotation=build_rotation(model.get_rotation);axis=model.get_scaling.argmin(1)
    axisnormal=rotation[torch.arange(len(rotation),device='cuda'),:,axis]
def pngurl(a):
    img=Image.fromarray((np.clip(a,0,1)*255).round().astype(np.uint8))
    b=io.BytesIO();img.save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
def frame(yaw,pitch):
    y,x=np.deg2rad([yaw,pitch])
    Y=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    X=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    Q=Y@X
    C=np.linalg.inv(np.asarray(base['world_view_transform']).T)
    C[:3,:3]=Q@C[:3,:3];C[:3,3]=origin+Q@(C[:3,3]-origin);E=np.linalg.inv(C)
    W=torch.tensor(E.T,dtype=torch.float32,device='cuda')
    P=torch.tensor(base['projection_matrix'],dtype=torch.float32,device='cuda')
    c=SimpleNamespace(image_width=base['width'],image_height=base['height'],Fx=base['Fx'],Fy=base['Fy'],Cx=base['Cx'],Cy=base['Cy'],FoVx=2*np.arctan(base['width']/(2*base['Fx'])),FoVy=2*np.arctan(base['height']/(2*base['Fy'])),R=E[:3,:3].T,T=E[:3,3],world_view_transform=W,projection_matrix=P,full_proj_transform=W@P,camera_center=W.inverse()[3,:3])
    with torch.no_grad():
        package=render(c,model,pipe,gray);rgb=package['render'].cpu().numpy().transpose(1,2,0)
        if not refined:
            alpha=package['rend_alpha'].cpu().numpy()[0];depth=package['surf_depth'].cpu().numpy()[0]
            normal=package['rend_normal'].cpu().numpy().transpose(1,2,0)
        else:
            z=(model.get_xyz@torch.tensor(c.R,dtype=torch.float32,device='cuda')+torch.tensor(c.T,dtype=torch.float32,device='cuda'))[:,2]
            moments=render(c,model,pipe,black,override_color=torch.stack([torch.ones_like(z),z,z*z],1))['render']
            alpha_t=moments[0];depth_t=moments[1]/alpha_t.clamp_min(1e-8)
            n=axisnormal.clone();n[((c.camera_center[None]-model.get_xyz)*n).sum(1)<0]*=-1
            normal_t=render(c,model,pipe,black,override_color=(n+1)/2)['render']/alpha_t.clamp_min(1e-8)[None]*2-1
            normal_t=torch.nn.functional.normalize(normal_t,dim=0);normal_t[:,alpha_t<1e-8]=0
            alpha=alpha_t.cpu().numpy();depth=depth_t.cpu().numpy();normal=normal_t.cpu().numpy().transpose(1,2,0)
    dv=colormaps['turbo'](np.clip(depth/scale['tsdf_depth_trunc'],0,1))[...,:3];nv=(normal+1)/2
    dv[alpha<.01]=0;nv[alpha<.01]=0
    return dict(viewer_note=viewer_note,viewer_title=viewer_title,rgb=pngurl(rgb),alpha=pngurl(alpha),depth=pngurl(dv),normal=pngurl(nv),model=str(ply.resolve()),depth_label='Center-Z AUX' if refined else 'Native median depth',normal_label='Axis normal AUX' if refined else 'Native world normal')
HTML="""<!doctype html><meta charset="utf-8"><title>E14 native model viewer</title>
<style>body{background:#202020;color:#ddd;font:16px sans-serif;margin:24px}label{margin:16px}button{font:inherit}main{display:flex;gap:12px}figure{margin:0;width:24%}img{width:100%;height:auto}small{display:block;margin:16px 0;color:#aaa}</style>
<h1>E14 native model viewer</h1><label>Yaw <input id="yaw" type="range" min="-180" max="180" value="0"><output id="yv">0</output>°</label>
<label>Pitch <input id="pitch" type="range" min="-80" max="80" value="0"><output id="pv">0</output>°</label><button id="go">Render</button>
<small id="status">Loading native model view…</small><main><figure><figcaption>RGB</figcaption><img id="rgb"></figure><figure><figcaption>Alpha</figcaption><img id="alpha"></figure><figure><figcaption id="dl">Depth</figcaption><img id="depth"></figure><figure><figcaption id="nl">Normal</figcaption><img id="normal"></figure></main>
<script>
for(const [id,out] of [['yaw','yv'],['pitch','pv']])document.getElementById(id).oninput=()=>document.getElementById(out).textContent=document.getElementById(id).value;
let busy=false;async function show(){if(busy)return;busy=true;document.getElementById('go').disabled=true;document.getElementById('status').textContent='Rendering…';
try{const r=await fetch('/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({yaw:+document.getElementById('yaw').value,pitch:+document.getElementById('pitch').value})});const d=await r.json();if(!r.ok)throw Error(d.error);for(const k of ['rgb','alpha','depth','normal'])document.getElementById(k).src=d[k];document.getElementById('dl').textContent=d.depth_label;document.getElementById('nl').textContent=d.normal_label;document.getElementById('status').textContent=d.model+' — '+d.viewer_note;}catch(e){document.getElementById('status').textContent=e.message;}finally{busy=false;document.getElementById('go').disabled=false;}}
document.getElementById('go').onclick=show;show();
</script>"""
import html
HTML=HTML.replace('E14 native model viewer',html.escape(viewer_title))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path!='/':self.send_error(404);return
        data=HTML.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_POST(self):
        if self.path!='/render':self.send_error(404);return
        try:
            request=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length','0')),4096)))
            yaw=float(request.get('yaw',0));pitch=float(request.get('pitch',0))
            assert np.isfinite(yaw) and np.isfinite(pitch)
            result=frame(np.clip(yaw,-180,180),np.clip(pitch,-80,80));code=200
        except Exception as exc:result={'error':str(exc)};code=500
        data=json.dumps(result).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
print(f'Native model: {ply.resolve()}',flush=True)
print(f'Open http://127.0.0.1:{args.port}; remote access via SSH port forwarding. Ctrl-C stops the viewer.',flush=True)
HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

