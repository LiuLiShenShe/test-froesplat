"""Reuse the completed same-acquisition RAP-FSAM3 run; map to common COLMAP RGB."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib
import io
import json
import resource
import sys
import time
from pathlib import Path
from runtime import ROOT, OUT, DATA, Meter, write
import numpy as np
from PIL import Image

meter=Meter();wall=time.perf_counter();cpu=time.process_time()
archive=ROOT/'00-论文优化重构/数据管理/03-分割Mask/05-RAP-FSAM3掩膜/E4b_downstream/KongQueZhuYu'
config=json.loads((archive/'参数.json').read_text())
log=json.loads((archive/'运行日志.json').read_text())
assert (ROOT/config['input_dir']).resolve()==(ROOT/'02-FFT/KongQueZhuYu').resolve()
assert log['status']=='success' and log['num_images']==210
assert config['use_spnp_refinement'] and config['spnp_backend']=='sam3_if_supported'
assert config['thin_repair_mode']=='adjacent' and config['use_corrective_geometry']
assert (ROOT/config['colmap_dir']).resolve()==(ROOT/'04-COLMAP-rerun-original/KongQueZhuYu/sequential').resolve()
names=sorted((OUT/'common_input/train_views.txt').read_text().splitlines()+(OUT/'common_input/test_views.txt').read_text().splitlines())
assert len(names)==210
assert {p.name for p in (ROOT/config['input_dir']).glob('*.jpg')}==set(names)
sys.path.insert(0,str(ROOT/'00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十E0_三维全视图与输入冻结/scripts'))
from build_datasets import inverse_maps,remap_nearest
import pycolmap
source=json.loads((OUT/'common_input/colmap_source.json').read_text())
src=source['source_camera'];dst=source['undistorted_camera']
camera=lambda record:pycolmap.Camera(model=record['model'],width=record['width'],height=record['height'],params=record['params'])
mx,my,valid=inverse_maps(camera(src),camera(dst))
destination=DATA/'rap_masks';destination.mkdir(exist_ok=True)
manifest=[]
for name in names:
    stem=Path(name).stem;p=archive/'最终掩膜'/f'mask_{stem}.png'
    payload=meter.read(p);raw=np.array(Image.open(io.BytesIO(payload)).convert('L'))>127
    assert raw.shape==(src['height'],src['width'])
    mapped=remap_nearest(raw,mx,my,valid)
    small=Image.fromarray(mapped.astype(np.uint8)*255).resize((source['width'],source['height']),Image.Resampling.NEAREST)
    buffer=io.BytesIO();small.save(buffer,format='PNG');meter.save(destination/f'{stem}.png',buffer.getbuffer())
    manifest.append(dict(name=name,source=str(p),source_sha256=hashlib.sha256(payload).hexdigest(),
        mapped_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),foreground_pixels=int((np.array(small)>127).sum())))
    print('MAP_RAP',stem,flush=True)
assert all(r['foreground_pixels']>0 for r in manifest)
write(OUT/'common_input/rap_mask_manifest.json',manifest)
write(OUT/'common_input/rap_masks_complete.json',dict(status='complete',count=len(manifest),source=str(archive),
    reuse=True,input_same_acquisition=True,raw_image_root=str((ROOT/config['input_dir']).resolve()),
    archive_configuration=config,archive_success_log=log,
    archive_record_sha256={n:hashlib.sha256(meter.read(archive/n)).hexdigest() for n in ['参数.json','运行日志.json','图像索引.csv']},
    mask_mapping='Stage20E0 COLMAP inverse distortion map, nearest interpolation, nearest resize',
    role='Uniform evaluation and surface extraction only; no baseline receives mask loss',
    masks_are_model_predictions=True,not_manual_ground_truth=True,
    fresh_attempt='An unnecessary fresh run was interrupted after five candidate-score frames; it supplies no formal masks. See RAP_FSAM3.log.'))
write(OUT/'common_input/rap_resources.json',dict(wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu,
    peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,io=meter.stats(),
    scope='CPU archive validation and mapping only; historical inference time is in archive_success_log'))
