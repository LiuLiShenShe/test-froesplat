"""Run native training loops with instrumentation and the frozen common split."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import importlib.util
import resource
import sys
import time
from pathlib import Path
from runtime import DATA, OUT, SOURCES, Meter, install_io, write

p = argparse.ArgumentParser()
p.add_argument('method', choices=['3DGS', '2DGS', 'GGGS', 'MCMC_3DGS'])
p.add_argument('--cap-max', type=int)
args = p.parse_args()
source, output = SOURCES[args.method], OUT / 'gaussian' / args.method
sys.path.insert(0, str(source))
import torch
torch.set_num_threads(8)
if args.method == '2DGS':
    # Compatibility shim for Matplotlib>=3.10; official 2DGS calls the
    # removed FigureCanvasAgg.tostring_rgb only while writing its test images.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not hasattr(FigureCanvasAgg, 'tostring_rgb'):
        def _tostring_rgb(self):
            self.draw();rgba=np.asarray(self.buffer_rgba());return rgba[...,:3].tobytes()
        import numpy as np
        FigureCanvasAgg.tostring_rgb=_tostring_rgb
meter = Meter()
install_io(meter)
from scene import Scene
original_scene = Scene.__init__


def checked_scene(self, *a, **kw):
    original_scene(self, *a, **kw)
    assert len(self.gaussians.get_xyz)==177918, 'Initialization must retain the identical SfM point set'
    train = sorted(Path(c.image_name).stem for c in self.getTrainCameras())
    test = sorted(Path(c.image_name).stem for c in self.getTestCameras())
    for label, actual in [('train', train), ('test', test)]:
        expected = sorted(Path(n).stem for n in (OUT/f'common_input/{label}_views.txt').read_text().splitlines())
        assert actual == expected, (args.method, label, len(actual), len(expected))
    write(output/'verified_input.json', dict(dataset=str(DATA), train=len(train), test=len(test),
          initial_gaussians=len(self.gaussians.get_xyz), seed=0, sh_degree=3, mask_loss=False))


Scene.__init__ = checked_scene
spec = importlib.util.spec_from_file_location('official_train', source/'train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from arguments import ModelParams, OptimizationParams, PipelineParams
parser = argparse.ArgumentParser()
lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
cli = ['--source_path', str(DATA), '--model_path', str(output), '--resolution', '1', '--sh_degree', '3', '--eval', '--iterations', '30000']
if args.method == 'GGGS': cli += ['--sg_degree', '0']
if args.method == 'MCMC_3DGS':
    assert args.cap_max and args.cap_max >= 177918
    cli += ['--init_type','sfm','--cap_max',str(args.cap_max),'--scale_reg','.01','--opacity_reg','.01','--noise_lr','500000']
params = parser.parse_args(cli)
module.args = params
module.safe_state(False)
write(output/'training_configuration.json', dict(cli=cli, parameters=vars(params), source=str(source),
      entrypoint='official train.py::training', instrumentation='Scene split/initial-count assertions and paced serialization; 2DGS only: Matplotlib tostring_rgb compatibility alias for native report images. No optimization changes.'))
torch.cuda.reset_peak_memory_stats()
wall, cpu = time.perf_counter(), time.process_time()
success = False
try:
    # Same native training function and 30K native optimization steps. The viewer
    # socket is unnecessary; try_connect returns immediately without a listener.
    train_args = [lp.extract(params), op.extract(params), pp.extract(params), [30000], [30000], [30000], None]
    if args.method != '2DGS': train_args.append(-1)
    module.training(*train_args)
    torch.cuda.synchronize()
    success = True
finally:
    write(output/'worker_resources.json', dict(success=success, wall_s=time.perf_counter()-wall,
          cpu_s=time.process_time()-cpu, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, physical_gpu=1, io=meter.stats()))
if success:
    ply = output/'point_cloud/iteration_30000/point_cloud.ply'
    with ply.open('rb') as f:
        for line in f:
            if line.startswith(b'element vertex '):
                count = int(line.split()[-1]); break
    write(output/'complete.json', dict(status='complete', iterations=30000, gaussian_count=count,
          native_ply=str(ply), model_bytes=ply.stat().st_size, checkpoint=str(output/'chkpnt30000.pth')))
