"""Existing native GGGS training, fixed30K/SH3/SG0/seed0; no optimizer changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
from batch_common import *
import argparse,fcntl,resource

def main(plant):
 ctx=Context(plant);folder=ctx.out/'gaussian/original';complete=folder/'complete.json';folder.mkdir(parents=True,exist_ok=True)
 if complete.exists():return
 if plant=='KongQueZhuYu':
  original=read(PRE/'gaussian/GGGS/complete.json');link(ctx.out/'gaussian/original_gaussians.ply',original['native_ply']);write(complete,dict(**original,reused_true_colmap_30k=True,source=str(PRE/'gaussian/GGGS')));ctx.status('gggs','complete',reuse=True,iterations=30000);return
 lock=(PRE/'summary/gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 if complete.exists():return
 import torch
 torch.set_num_threads(8);sys.path.insert(0,str(GGGS));meter=ctx.meter
 from runtime import install_io
 install_io(meter);install_scalar_ply_io(meter)
 from scene import Scene
 original_scene=Scene.__init__
 def checked(self,*args,**kwargs):
  original_scene(self,*args,**kwargs);data,_=ctx.cameras();assert len(self.gaussians.get_xyz)==data['sparse_points']
  for split,cams in [('train',self.getTrainCameras()),('test',self.getTestCameras())]:assert sorted(c.image_name for c in cams)==sorted(Path(n).stem for n in (ctx.common/f'{split}_views.txt').read_text().splitlines())
 Scene.__init__=checked
 spec=importlib.util.spec_from_file_location('native_gggs_train',GGGS/'train.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 from arguments import ModelParams,OptimizationParams,PipelineParams
 parser=argparse.ArgumentParser();lp,op,pp=ModelParams(parser),OptimizationParams(parser),PipelineParams(parser)
 cli=['--source_path',str(ctx.common),'--model_path',str(folder),'--resolution','1','--sh_degree','3','--sg_degree','0','--eval','--iterations','30000'];params=parser.parse_args(cli);module.args=params;module.safe_state(False)
 write(folder/'training_configuration.json',dict(cli=cli,parameters=vars(params),source=str(GGGS),true_colmap=read(ctx.common/'complete.json'),no_masked_rgb=True,seed=0));ctx.status('gggs','running',iterations_target=30000,pid=os.getpid());wall=time.time();cpu=time.process_time();torch.cuda.reset_peak_memory_stats()
 try:
  module.training(lp.extract(params),op.extract(params),pp.extract(params),[30000],[30000],[],None,-1);torch.cuda.synchronize()
 except BaseException as e:ctx.status('gggs','failed',error=str(e));raise
 finally:write(folder/'resources.json',dict(wall_s=time.time()-wall,cpu_s=time.process_time()-cpu,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,physical_gpu=1,io=meter.stats()))
 ply=folder/'point_cloud/iteration_30000/point_cloud.ply';assert ply.exists();link(ctx.out/'gaussian/original_gaussians.ply',ply);write(complete,dict(status='complete',iterations=30000,native_ply=str(ply),source_units='COLMAP arbitrary units',source_scene=str(ctx.common),config=dict(sh_degree=3,sg_degree=0,seed=0)));ctx.status('gggs','complete',iterations=30000)

if __name__=='__main__':main(sys.argv[1])
