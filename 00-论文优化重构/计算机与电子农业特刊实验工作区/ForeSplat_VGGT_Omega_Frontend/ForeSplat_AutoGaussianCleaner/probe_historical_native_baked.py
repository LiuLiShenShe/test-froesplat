import os;os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='8'
from pathlib import Path
import json,sys,numpy as np,torch
from PIL import Image
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
from ForeSplat_AutoGaussianCleaner.cleaner.runtime import Scene
from ForeSplat_AutoGaussianCleaner.cleaner.io import read_ply
from ForeSplat_AutoGaussianCleaner.evaluation.free_view_evaluation import camera
from ForeSplat_AutoGaussianCleaner.evaluation.metrics import arrays
root=BASE/'ForeSplat_AutoGaussianCleaner/experiments/Historical_TwoStage_Cleaner/v4/KongQueZhuYu'
r=json.loads((root/'execution.json').read_text())['reference']; config=r['config'];scene=Scene(r['scene'],config);torch.set_num_threads(8)
model,_=scene.load_model(r['scene']['baseline_models']['R5'])
protocol=json.loads((root/'stage2_residual/visualizations/automatic_free_views.json').read_text())
out=root/'comparison/native_kernel_probe';out.mkdir(parents=True,exist_ok=True)
for v in protocol['definitions']:
 if v['name'] not in ('top','orbit_03'):continue
 c=camera(np.asarray(v['K']),np.asarray(v['E']),v['width'],v['height'],v['name'],0)
 for k in [0.,.1,.3,.5,1.0]:
  with torch.no_grad(): a=arrays(scene.renderer.render(c,model,scene.pipe,scene.background,k,require_depth=True))
  rgb=a['rgb']+(1-a['alpha'][...,None]);Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(out/f'{v["name"]}_k{k:g}.png')
baked,_=scene.load_model(r['scene']['baseline_models']['R5'].replace('cleaned_gaussians.ply','cleaned_gaussians_supersplat.ply'))
for v in protocol['definitions']:
 if v['name']!='top':continue
 c=camera(np.asarray(v['K']),np.asarray(v['E']),v['width'],v['height'],v['name'],0)
 with torch.no_grad(): a=arrays(scene.renderer.render(c,baked,scene.pipe,scene.background,0.,require_depth=True))
 Image.fromarray(np.uint8(np.clip(a['rgb']+(1-a['alpha'][...,None]),0,1)*255)).save(out/'top_baked_native.png')
print(out)
