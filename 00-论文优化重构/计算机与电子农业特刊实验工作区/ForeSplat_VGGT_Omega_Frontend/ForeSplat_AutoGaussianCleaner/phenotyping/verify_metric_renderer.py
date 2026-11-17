"""Verify native/metric depth equivalence and refresh the early pilot outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
from batch_common import *
from batch_mesh import render_frames
import copy,fcntl,subprocess

def main(ctx):
 lock=(PRE/'summary/gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 factor=read(ctx.out/'calibration/scale_calibration.json')['scale_cm_per_unit'];baseline=ctx.out/'attempts/pre_metric_automatic_cube_rejection/mesh/surface/frames';selected=[p.stem for p in sorted(baseline.glob('*.npz'))[::20]];test=copy.copy(ctx);test.out=OUT/'summary/validation'/ctx.plant/'metric_renderer_equivariance';test.out.mkdir(parents=True,exist_ok=True)
 for name in ['original','original_gaussians.ply','metric_gaussians.ply','metric_cameras.json']:
  if not(test.out/'gaussian'/name).exists():link(test.out/'gaussian'/name,ctx.out/'gaussian'/name)
 if not(test.out/'calibration').exists():link(test.out/'calibration',ctx.out/'calibration')
 render_frames(test,True,set(selected));records=[]
 for name in selected:
  with np.load(io.BytesIO(ctx.meter.read(baseline/f'{name}.npz'))) as d:raw=d['depth'];raw_rgb=d['rgb'];raw_c2w=d['c2w']
  with np.load(io.BytesIO(ctx.meter.read(test.out/'mesh/surface/frames'/f'{name}.npz'))) as d:cm=d['depth'];cm_rgb=d['rgb'];cm_c2w=d['c2w']
  shared=(raw>0)&(cm>0);union=(raw>0)|(cm>0);relative=np.abs(cm[shared]-raw[shared]*factor)/np.maximum(cm[shared],1e-6);records.append(dict(view=name,positive_depth_mask_iou=float(shared.sum()/max(1,union.sum())),relative_depth_p50=float(np.median(relative)),relative_depth_p99=float(np.quantile(relative,.99)),rgb_mean_absolute_error=float(abs(raw_rgb.astype(float)-cm_rgb).mean()),camera_center_scaled=np.allclose(cm_c2w[:3,3],raw_c2w[:3,3]*factor),camera_rotation_identical=np.array_equal(cm_c2w[:3,:3],raw_c2w[:3,:3])))
 passed=all(r['positive_depth_mask_iou']>.99 and r['relative_depth_p99']<.002 and r['camera_center_scaled'] and r['camera_rotation_identical'] for r in records);write(OUT/'summary/validation/native_renderer_uniform_scale_corrected.json',dict(passed=passed,plant=ctx.plant,factor=factor,views=records,unit_correction='Same nativeGGGS median-depth search range0.4 and nearplane0.2 expressed in metric units; no new reconstruction algorithm',io=ctx.meter.stats()));assert passed,records
 archive=ctx.out/'attempts/before_native_depth_unit_fix';archive.mkdir(parents=True,exist_ok=True)
 for name in ['mesh','plant_only','leaf_instances','phenotype','summary']:
  source=ctx.out/name
  if source.exists():source.rename(archive/name)
 (ctx.out/'leaf_instances').mkdir();source=archive/'leaf_instances'
 for name in ['candidates','candidates.json']:link(ctx.out/'leaf_instances'/name,source/name)
 render_frames(ctx,True);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 import torch,gc
 gc.collect();torch.cuda.empty_cache()
 command=[str(UV),'run','--no-project','--python',str(ROOT/'2d-gaussian-splatting-main/venv/bin/python'),str(CODE/'batch_mesh.py'),ctx.plant,'--fuse-only'];subprocess.run(command,check=True,cwd=ROOT)
 for script in ['batch_instances.py','batch_measure.py','batch_manual.py','batch_report.py','batch_validate.py']:
  subprocess.run([str(UV),'run','--no-project','--python',str(PYTHON),str(CODE/script),ctx.plant],check=True,cwd=ROOT)
 from batch_report import summarize
 summarize();print('NATIVE_METRIC_RENDER_VERIFIED_AND_PILOT_REFRESHED',ctx.plant,flush=True)

if __name__=='__main__':main(Context(sys.argv[1]));sys.stdout.flush();sys.stderr.flush();os._exit(0)
