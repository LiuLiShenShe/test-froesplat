"""Official simple_trainer Runner + DefaultStrategy, same source and seed zero."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import dataclasses
import io
import json
import resource
import sys
import time
import types
from pathlib import Path
from runtime import DATA, OUT, SOURCES, Meter, install_io, write

sys.path.insert(0, str(SOURCES['gsplat']/'examples'))
import torch
import numpy as np
torch.set_num_threads(8)
sys.modules['tensorboard.compat.notf']=types.ModuleType('tensorboard.compat.notf')
import simple_trainer as official
from gsplat.strategy import DefaultStrategy
meter=Meter()
install_io(meter)
# The official parser uses imageio.v2 rather than PIL directly. Bound those
# dataset reads and native evaluation PNG writes through the same I/O meter.
import imageio.v2 as imageio_v2
original_imread,original_imwrite=imageio_v2.imread,imageio_v2.imwrite
def imread(uri,*a,**kw):
    if isinstance(uri,(str,Path)) and str(DATA) in str(Path(uri).resolve()):
        return original_imread(io.BytesIO(meter.read(uri)),*a,**kw)
    return original_imread(uri,*a,**kw)
def imwrite(uri,image,*a,**kw):
    if isinstance(uri,(str,Path)) and str(OUT) in str(Path(uri).resolve()):
        b=io.BytesIO();original_imwrite(b,image,*a,format=Path(uri).suffix.lstrip('.'),**kw)
        meter.save(uri,b.getbuffer());return
    return original_imwrite(uri,image,*a,**kw)
imageio_v2.imread,imageio_v2.imwrite=imread,imwrite
official.imageio.imwrite=imwrite
original_export=official.export_splats
def export(*a, **kw):
    target=kw.pop('save_to',None)
    data=original_export(*a,save_to=None,**kw)
    if target:meter.save(target,data)
    return data
official.export_splats=export
original_seed=official.set_random_seed
official.set_random_seed=lambda ignored: original_seed(0)
cfg=official.Config(data_dir=str(DATA), data_factor=1, result_dir=str(OUT/'gaussian/gsplat'),
    test_every=8, normalize_world_space=False, max_steps=30000, sh_degree=3,
    init_type='sfm', pose_opt=False, disable_viewer=True, disable_video=True,
    save_ply=True, save_steps=[30000], ply_steps=[30000], eval_steps=[30000],
    strategy=DefaultStrategy(verbose=True))
output=OUT/'gaussian/gsplat'
write(output/'training_configuration.json',json.loads(json.dumps(dataclasses.asdict(cfg),default=str)))
wall,cpu=time.perf_counter(),time.process_time()
torch.cuda.reset_peak_memory_stats()
success=False
try:
    runner=official.Runner(local_rank=0,world_rank=0,world_size=1,cfg=cfg)
    assert isinstance(runner.cfg.strategy,DefaultStrategy)
    assert len(runner.splats['means'])==177918, 'Initialization must retain the identical SfM point set'
    assert all(value is None for value in runner.parser.mask_dict.values()), 'No per-method training masks'
    common_K=np.asarray(json.loads((OUT/'common_input/colmap_source.json').read_text())['K'])
    assert all(np.allclose(value,common_K,atol=1e-5) for value in runner.parser.Ks_dict.values())
    for label,dataset in [('train',runner.trainset),('test',runner.valset)]:
        actual=sorted(Path(runner.parser.image_names[int(i)]).stem for i in dataset.indices)
        expected=sorted(Path(n).stem for n in (OUT/f'common_input/{label}_views.txt').read_text().splitlines())
        assert actual==expected,(label,len(actual),len(expected))
    write(output/'verified_input.json',dict(dataset=str(DATA),seed=0,train=len(runner.trainset),
          test=len(runner.valset),initial_gaussians=len(runner.splats['means']),strategy='DefaultStrategy',
          pose_opt=False,normalize_world_space=False,mask_loss=False))
    # Cache the 183 immutable raw RGB samples once. This avoids decoding and
    # rereading the entire dataset for ~164 epochs. No pixels/losses change.
    dataset=runner.trainset
    cache=[dataset[i] for i in range(len(dataset))]
    assert all('mask' not in row for row in cache)
    class CachedDataset(torch.utils.data.Dataset):
        def __len__(self):return len(cache)
        def __getitem__(self,index):return cache[index]
    runner.trainset=CachedDataset()
    runner.train()
    torch.cuda.synchronize()
    success=True
finally:
    write(output/'worker_resources.json',dict(success=success,wall_s=time.perf_counter()-wall,
          cpu_s=time.process_time()-cpu,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,physical_gpu=1,io=meter.stats()))
if success:
    ply=output/'ply/point_cloud_29999.ply'
    if not ply.exists():
        candidates=list((output/'ply').glob('*.ply'));assert len(candidates)==1;ply=candidates[0]
    with ply.open('rb') as f:
        for line in f:
            if line.startswith(b'element vertex '):native_count=int(line.split()[-1]);break
    assert native_count==len(runner.splats['means']), 'Official exporter filtered nonfinite rows; investigate before comparing or cleaning'
    write(output/'complete.json',dict(status='complete',iterations=30000,gaussian_count=len(runner.splats['means']),
          native_ply=str(ply),model_bytes=ply.stat().st_size,checkpoint=str(output/'ckpts/ckpt_29999_rank0.pt')))
