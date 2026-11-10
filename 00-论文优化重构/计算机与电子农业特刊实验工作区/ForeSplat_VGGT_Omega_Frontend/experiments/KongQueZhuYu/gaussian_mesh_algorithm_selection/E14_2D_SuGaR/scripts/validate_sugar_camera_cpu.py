"""CPU-only real source conversion/load of all corrected E14 SuGaR cameras against E10."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,importlib.util
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];OFF=ROOT/'official'
sys.path.insert(0,str(OFF));sys.path.append(str(OFF/'gaussian_splatting'))
from sugar_scene.cameras import load_gs_cameras
spec=importlib.util.spec_from_file_location('colmap',OFF/'gaussian_splatting_2d/scene/colmap_loader.py')
cmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(cmod)
cs=cmod.read_intrinsics_binary(ROOT/'input_colmap/sparse/0/cameras.bin')
ims=cmod.read_extrinsics_binary(ROOT/'input_colmap/sparse/0/images.bin')
folder=ROOT/'camera_cpu_audit';folder.mkdir(exist_ok=True);records=[]
for i,im in enumerate(sorted(ims.values(),key=lambda x:x.name)):
    c=cs[im.camera_id];fx,fy,cx,cy=map(float,c.params)
    E=np.eye(4);E[:3,:3]=cmod.qvec2rotmat(im.qvec);E[:3,3]=im.tvec
    C=np.linalg.inv(E)
    records.append(dict(id=i,img_name=Path(im.name).stem,width=c.width,height=c.height,position=C[:3,3].tolist(),rotation=C[:3,:3].tolist(),cx=cx,cy=cy,fx=fx,fy=fy))
(folder/'cameras.json').write_text(json.dumps(records))
# GSCamera's only GPU methods for no-GT camera creation are Tensor.cuda; this explicit audit routes them to CPU.
original_cuda=torch.Tensor.cuda;torch.Tensor.cuda=lambda self,*a,**kw:self
try:cams=load_gs_cameras(str(ROOT/'input_colmap'),str(folder),load_gt_images=False)
finally:torch.Tensor.cuda=original_cuda
dual=ROOT.parent.parent/'geometry_rgb_dual_frontend'
ref={r['name']:r for r in json.loads((dual/'render_comparison/E10/camera_runtime.json').read_text())['records']}
errors={k:0. for k in ['Fx','Fy','Cx','Cy','world_view_transform','projection_matrix','full_proj_transform','camera_center']}
for c in cams:
    r=ref[c.image_name]
    assert c.image_width==r['width'] and c.image_height==r['height']
    for k in errors:
        actual=getattr(c,k)
        if torch.is_tensor(actual):actual=actual.numpy()
        errors[k]=max(errors[k],float(np.max(np.abs(np.asarray(actual)-r[k]))))
assert len(cams)==208 and max(errors.values())<3e-7,errors
(ROOT/'sugar_camera_all208_cpu_validation.json').write_text(json.dumps(dict(status='passed',views=208,max_abs_errors=errors,gt_loaded=False,gpu_called=False,method='Actual official load_gs_cameras/GSCamera with Tensor.cuda routed to CPU for no-image camera creation; full original COLMAP K, no independently modified geometry',reference=str((dual/'render_comparison/E10/camera_runtime.json').resolve())),indent=2)+'\n')
print(errors)

