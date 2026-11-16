"""Read-only provenance and isolated-environment inventory, without CUDA contexts."""
import hashlib,json,subprocess,os
from pathlib import Path
from runtime import OUT,SOURCES,PYTHONS,Meter,write
meter=Meter()
rows={}
for method,source in SOURCES.items():
 def git(*a):
  p=subprocess.run(['git','-C',str(source),*a],capture_output=True,text=True)
  return p.stdout.strip() if not p.returncode else {'error':p.stderr.strip()}
 p=subprocess.run([str(PYTHONS[method]),'-c',"import sys,importlib.metadata as m,json; names=['torch','torchvision','numpy','scipy','gsplat','pycolmap','open3d','lpips']; print(json.dumps({'python':sys.version,'packages':{n:m.version(n) for n in names if m.packages_distributions().get(n) or n in ['torch','numpy','scipy','lpips']}}))"],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 repo=git('rev-parse','--show-toplevel')
 native_repo=isinstance(repo,str) and Path(repo).resolve()==source.resolve()
 hashes={str(f.relative_to(source)):hashlib.sha256(meter.read(f)).hexdigest() for f in source.rglob('*') if f.is_file() and f.suffix in ['.py','.cu','.cpp','.h','.hpp'] and not any(t in f.parts for t in ['.git','build','venv','.venv','__pycache__'])}
 rows[method]=dict(source=str(source),version_kind='upstream checkout' if native_repo else 'vendored source pinned by parent repository and per-file hashes',repository_root=repo,commit=git('rev-parse','HEAD'),remote=git('remote','get-url','origin'),source_sha256=hashes,status=git('status','--short','--','.'),python=str(PYTHONS[method]),environment=json.loads(p.stdout) if p.returncode==0 else {'error':p.stderr[-2000:]})
write(OUT/'summary/source_environment_audit.json',rows)
write(OUT/'summary/wrapper_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'scripts').glob('*.py')})
write(OUT/'summary/comparison_protocol.json',dict(plant='KongQueZhuYu',frontend='Historical real COLMAP CPU SIFT, sequential matches, mapper, triangulation and BA',input='Identical raw undistorted RGB and sparse/0; no masked RGB or mask loss',width=530,height=948,train_views=183,test_views=27,split='Sorted filenames, every eighth starting at index zero held out',seed=0,sh_degree=3,iterations=30000,physical_gpu=1,gpu_serial=True,initial_points=177918,gsplat_strategy='Official DefaultStrategy',mcmc='Official ubc code and rasterizer; sfm, regs .01/.01, noise_lr500000; cap_max locked to baseline before launch',mask_reference='Archived full RAP-FSAM3 predictions from same raw acquisition, inverse-undistorted and resized nearest; not manual ground truth',depth='Camera-axis Z; estimators still differ and are documented per method',surface='Same native depth/mask/alpha/finite-difference observation protocol, identical frozen NPZ for NKSR and Poisson',io='Paced instrumented image/PLY/checkpoint/surface/mesh reads and writes; includes large artifacts. Python/import metadata and some native COLMAP readers are not part of the instrumented byte count.',no_ground_truth_3d=True,physical_units='COLMAP similarity gauge; no metric phenotype dimensions without independent calibration'))
print('Audit saved')
