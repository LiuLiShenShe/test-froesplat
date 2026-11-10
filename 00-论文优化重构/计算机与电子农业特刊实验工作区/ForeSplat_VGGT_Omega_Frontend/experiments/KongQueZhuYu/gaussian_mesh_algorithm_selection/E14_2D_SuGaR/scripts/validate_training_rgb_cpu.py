"""Exercise both native RGB/camera loaders on CPU before the expensive E14 stages.
Only Tensor.cuda and camera data-device routing are redirected for this audit.
No model is instantiated, and no training or prior inference is run.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
import sys,json,csv,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];OFF=ROOT/'official'
sys.path.insert(0,str(OFF/'gaussian_splatting_2d'));sys.path.insert(0,str(OFF))
from scene.dataset_readers import readColmapCameras
from scene.colmap_loader import read_intrinsics_binary,read_extrinsics_binary
from utils.camera_utils import loadCam
import sugar_scene.cameras as sugar
torch.set_num_threads(8)
DUAL=ROOT.parent.parent/'geometry_rgb_dual_frontend'
refs={r['name']:r for r in json.loads((DUAL/'render_comparison/E10/camera_runtime.json').read_text())['records']}
hashes={int(r['source_id']):r['gt_sha256'] for r in csv.DictReader((DUAL/'render_comparison/E10/per_view_metrics.csv').open())}
cs=read_intrinsics_binary(ROOT/'input_colmap/sparse/0/cameras.bin')
ims=read_extrinsics_binary(ROOT/'input_colmap/sparse/0/images.bin')
assert not (ROOT/'input_colmap/estimated_dense_normal').exists(),'Run this preflight before E14 dense-prior generation'
records=[];started=time.time();torch_cuda=torch.Tensor.cuda
errors={k:0. for k in ['Fx','Fy','Cx','Cy','world_view_transform','projection_matrix','full_proj_transform']}

def check(c,stage):
    ref=refs[c.image_name];sid=int(c.image_name.split('_')[-1])
    assert (c.image_width,c.image_height)==(1080,1920)
    digest=hashlib.sha256(c.original_image.numpy().transpose(1,2,0).tobytes()).hexdigest()
    assert digest==hashes[sid],(stage,sid,'RGB differs from E10/E11')
    for key in errors:
        actual=getattr(c,key)
        if torch.is_tensor(actual):actual=actual.numpy()
        delta=float(np.max(np.abs(np.asarray(actual)-ref[key])))
        errors[key]=max(errors[key],delta);assert delta<3e-7,(stage,sid,key,delta)
    records.append(dict(stage=stage,source_id=sid,RGB_sha256=digest))

try:
    torch.Tensor.cuda=lambda self,*a,**kw:self
    infos=readColmapCameras(ims,cs,str(ROOT/'input_colmap/images'))
    args=SimpleNamespace(resolution=2,data_device='cpu')
    for i,info in enumerate(sorted(infos,key=lambda x:x.image_name)):
        camera=loadCam(args,i,info,1.);check(camera,'native2DGS');del camera
    folder=ROOT/'camera_cpu_audit';assert (folder/'cameras.json').is_file()
    original_init=sugar.GSCamera.__init__
    def observed_init(self,*a,**kw):
        kw['data_device']='cpu';original_init(self,*a,**kw);check(self,'nativeSuGaR_refinement');del self.original_image
    sugar.GSCamera.__init__=observed_init
    try:cameras=sugar.load_gs_cameras(str(ROOT/'input_colmap'),str(folder),load_gt_images=True)
    finally:sugar.GSCamera.__init__=original_init
    assert len(cameras)==208
finally:torch.Tensor.cuda=torch_cuda
assert len(records)==416
result=dict(status='passed_CPU_native_input_loaders',views_per_loader=208,all208_RGB_K_W2C_P_equal_to_E10_E11=True,max_abs_camera_errors=errors,records=records,elapsed_seconds=time.time()-started,
    GPU_executed=False,prior_inference_executed=False,training_executed=False,source_hashes={str(p.relative_to(OFF)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OFF/'gaussian_splatting_2d/utils/camera_utils.py',OFF/'gaussian_splatting_2d/scene/cameras.py',OFF/'sugar_scene/cameras.py']})
(ROOT/'training_and_refinement_RGB_cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('Both actual E14 native loaders preserve all208 frozen RGB/K/W2C/P',flush=True)
