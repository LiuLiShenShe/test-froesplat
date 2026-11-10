"""CPU-only exact source RGB and canonical resized-K verification."""
from pathlib import Path
import csv,json,hashlib,importlib.util,time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
DUAL=ROOT.parent.parent/'geometry_rgb_dual_frontend'
SRC=DUAL/'Standard_VGGT_Original_RGB/COLMAP_Corrected'
def main():
 started=time.time()
 refs={int(r['source_id']):r for r in csv.DictReader((DUAL/'render_comparison/E10/per_view_metrics.csv').open())}
 cams={r['name']:r for r in json.loads((DUAL/'render_comparison/E10/camera_runtime.json').read_text())['records']}
 spec=importlib.util.spec_from_file_location('qgs_colmap_only',ROOT/'adapted/scene/colmap_loader.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 extr=m.read_extrinsics_binary(str(SRC/'sparse/0/images.bin'));intr=m.read_intrinsics_binary(str(SRC/'sparse/0/cameras.bin'))
 def check(item):
  c=intr[item.camera_id];assert c.model=='PINHOLE'
  p=SRC/'images'/item.name;im=Image.open(p);assert im.mode=='RGB' and im.size==(2160,3840)
  rgb=np.asarray(im.resize((1080,1920))).astype(np.float32)/np.float32(255.)
  sid=int(Path(item.name).stem.split('_')[-1])
  digest=hashlib.sha256(rgb.tobytes()).hexdigest();assert digest==refs[sid]['gt_sha256'],(sid,digest,refs[sid]['gt_sha256'])
  k=np.asarray(c.params[:4]).copy();k[:2]*=.5;k[2:]=(k[2:]+.5)*.5-.5
  r=cams[Path(item.name).stem];refK=np.asarray([r['Fx'],r['Fy'],r['Cx'],r['Cy']])
  err=float(abs(k-refK).max());assert err<1e-8,(sid,k,refK)
  return dict(source_id=sid,path=str(p.resolve()),width=1080,height=1920,gt_sha256=digest,K=k.tolist(),K_error_max=err)
 with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(check,extr.values()))
 assert len(records)==208 and {r['source_id'] for r in records}==set(range(210))-{44,206}
 (ROOT/'input_cpu_validation.json').write_text(json.dumps(dict(status='passed',views=208,all208_GT_float_hashes_identical_to_E10=True,all208_canonical_K_identical_to_E10=True,runtime_GPU_K_P_W2C_and_kernel_validation='pending GPU release',elapsed_seconds=time.time()-started,records=sorted(records,key=lambda r:r['source_id'])),indent=2)+'\n')
 print('QGS all208 CPU frozen RGB/K validation passed',time.time()-started,flush=True)
if __name__=='__main__':main()

