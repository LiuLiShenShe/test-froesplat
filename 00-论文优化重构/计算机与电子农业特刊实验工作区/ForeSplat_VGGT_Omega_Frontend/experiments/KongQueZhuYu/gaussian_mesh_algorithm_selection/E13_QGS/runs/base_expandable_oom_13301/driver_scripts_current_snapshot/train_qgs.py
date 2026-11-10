"""Call the official QGS numerical training function with passive observers."""
import qgs_common as q
import argparse,time,csv,inspect,importlib.util,random,hashlib
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
def execution_record(record):
 p=q.ROOT/'training_execution_records.json'
 records=q.json.loads(p.read_text()) if p.exists() else []
 records=[r for r in records if r.get('run_id')!=record['run_id']]+[record]
 q.write(p,records)
 return records
def failed_native_export(error,scene,run_id):
 failed_iteration=None;operation=None;trace=error.__traceback__
 while trace is not None:
  frame=trace.tb_frame
  if frame.f_code.co_name=='training' and Path(frame.f_code.co_filename).resolve()==q.ADAPTED/'train.py':
   failed_iteration=int(frame.f_locals.get('iteration',0))
   lines=(q.ADAPTED/'train.py').read_text().splitlines();operation=lines[trace.tb_lineno-1].strip()
  trace=trace.tb_next
 result=dict(failed_iteration=failed_iteration,failed_native_operation=operation,
   actual_latest_transient_model_exported=False,resume_compatible=False)
 if scene is None:return result
 try:
  model=scene.gaussians;n=len(model.get_xyz)
  assert all(len(getattr(model,k))==n for k in ['_features_dc','_features_rest','_scaling','_rotation','_opacity'])
  assert model._scaling.shape[1]==6
  model.optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
  dest=q.ROOT/'failure_exports'/run_id;dest.mkdir(parents=True,exist_ok=True)
  checkpoint=dest/'native_failed_state.pth';ply=dest/'native_failed_state.ply'
  iteration_metadata=max(0,(failed_iteration or 1)-1)
  torch.save((model.capture(),iteration_metadata),checkpoint)
  model.save_ply(str(ply))
  result.update(actual_latest_transient_model_exported=True,native_primitives=n,
   checkpoint=str(checkpoint.resolve()),native_ply=str(ply.resolve()),
   checkpoint_iteration_metadata=iteration_metadata,
   stage='failed native model state; not a completed formal iteration',
   resume_compatible=operation=='loss.backward()',
   iteration_note='For backward failure the parameters/optimizer are from the preceding completed step; partial gradients are cleared before export. Other failure sites are not advertised as resume compatible.')
  q.write(dest/'PROVENANCE.json',result)
 except Exception as export_error:
  result['export_failure']=type(export_error).__name__+': '+str(export_error)
 return result
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--iterations',type=int,default=30000)
 parser.add_argument('--smoke',action='store_true')
 parser.add_argument('--resume',default=None)
 parser.add_argument('--checkpoint-every',type=int,default=0)
 a=parser.parse_args()
 assert a.checkpoint_every>=0
 out=q.ROOT/('smoke' if a.smoke else 'checkpoints')
 complete=q.ROOT/('smoke_complete.json' if a.smoke else 'training_complete.json')
 if complete.exists():print(complete.read_text());return
 if not a.smoke:assert q.json.loads((q.ROOT/'kernel_validation.json').read_text())['status']=='passed'
 cfg=q.config(a.iterations,out)
 if not a.smoke:assert a.iterations==30000
 out.mkdir(parents=True,exist_ok=True)
 OmegaConf.save(cfg,q.ROOT/('smoke_config.yaml' if a.smoke else 'actual_config.yaml'))
 q.read_only_image_adapter()
 spec=importlib.util.spec_from_file_location('official_qgs_training',q.ADAPTED/'train.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 # Match the official seed-0 initialization, without running disabled __main__.
 module.safe_state(False)
 random.seed(0);np.random.seed(0);torch.manual_seed(0)
 torch.set_num_threads(8);torch.cuda.reset_peak_memory_stats();started=time.time()
 run_id=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+f'_{q.os.getpid()}'
 run_record=dict(run_id=run_id,status='running',start_time=started,resume_checkpoint=str(Path(a.resume).resolve()) if a.resume else None,
  allocator=q.os.environ.get('PYTORCH_CUDA_ALLOC_CONF','native default'),seed=0,physical_gpu=1,checkpoint_every=a.checkpoint_every)
 execution_record(run_record)
 original_scene=module.Scene;state={}
 def observed_scene(*args,**kwargs):
  scene=original_scene(*args,**kwargs)
  validation=q.validate_cameras(scene.getTrainCameras())
  q.write(q.ROOT/('smoke_runtime_cameras.json' if a.smoke else 'runtime_cameras.json'),validation)
  assert scene.gaussians.get_xyz.shape[0]==1252383
  state['scene']=scene
  save=scene.save
  def observed_save(iteration):
   save(iteration)
   torch.save((scene.gaussians.capture(),iteration),out/f'chkpnt{iteration}.pth')
  scene.save=observed_save
  return scene
 module.Scene=observed_scene
 report=module.training_report
 metrics=q.ROOT/('smoke_train_metrics.csv' if a.smoke else 'train_metrics.csv')
 existing_metrics=metrics.exists() and metrics.stat().st_size>0
 f=metrics.open('a' if a.resume and existing_metrics else 'w',newline='')
 fields=['iteration','l1','image_loss','dist_loss','normal_loss','geo_loss','ncc_loss','total_loss','quadratic_primitives','wall_seconds']
 writer=csv.DictWriter(f,fieldnames=fields)
 if not (a.resume and existing_metrics):writer.writeheader()
 def observed_report(tb,iteration,Ll1,loss,l1_loss,elapsed,testing,scene,renderFunc,renderArgs,dataset):
  state['last_iteration']=iteration
  if iteration%100==0 or iteration==a.iterations:
   loc=inspect.currentframe().f_back.f_locals
   row=dict(iteration=iteration,l1=float(Ll1),image_loss=float(loc['image_loss']),total_loss=float(loss),quadratic_primitives=len(scene.gaussians.get_xyz),wall_seconds=time.time()-started)
   for k in ['dist_loss','normal_loss','geo_loss','ncc_loss']:row[k]=float(loc[k]) if loc[k] is not None else 0.0
   writer.writerow(row);f.flush()
   q.write(q.ROOT/'STATUS.json',dict(status='running',stage='smoke' if a.smoke else 'native_training',**row))
  if a.checkpoint_every and iteration%a.checkpoint_every==0 and iteration not in save:
   # Retain one rolling recovery state; the original 7K/15K/30K native
   # PLY/checkpoints remain separately retained by observed_save.
   extra=q.ROOT/'recovery/latest_native_state.pth'
   extra.parent.mkdir(parents=True,exist_ok=True)
   temporary=extra.with_suffix('.pth.tmp')
   torch.save((scene.gaussians.capture(),iteration),temporary)
   q.os.replace(temporary,extra)
   q.write(extra.with_suffix('.provenance.json'),dict(stage='native capture before optimizer step, matching official checkpoint semantics',
     iteration=iteration,checkpoint=str(extra.resolve()),run_id=run_id,seed=0,native_primitive='QGS signed six-scale quadratic',
     resume_rng='Upstream capture has no RNG/viewpoint-stack state; native resumed seed0 trajectory is recorded separately'))
  return report(tb,iteration,Ll1,loss,l1_loss,elapsed,testing,scene,renderFunc,renderArgs,dataset)
 module.training_report=observed_report
 save=[a.iterations] if a.smoke else [7000,15000,30000]
 test=[] if a.smoke else [7000,30000]
 provenance=dict(config=OmegaConf.to_container(cfg,resolve=True),physical_gpu=1,seed=0,torch=torch.__version__,official_commit='74d05c945e99fcaef7afe5a8831903be71ad9b55',
  native_train_function=True,cli_training_call_disabled_upstream=True,training_RGB_adapter='Original source images -> same PIL resize as E10; bypass official cv2 resize/recompression',
  native_representation='Quadratic signed 6-channel scale_0..5; native PLY only; not ordinary 3DGS Gaussian',checkpoint_export='Passive Scene.save wrapper exports native capture() + iteration',
  source_sha256={r:hashlib.sha256((q.ADAPTED/r).read_bytes()).hexdigest() for r in ['train.py','scene/cameras.py','scene/gaussian_model.py','gaussian_renderer/__init__.py']})
 provenance['resume_checkpoint']=str(Path(a.resume).resolve()) if a.resume else None
 provenance['allocator']=q.os.environ.get('PYTORCH_CUDA_ALLOC_CONF','native default')
 provenance['run_id']=run_id
 provenance['driver_source_sha256']={name:hashlib.sha256((q.ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ['train_qgs.py','qgs_common.py','run_e13.py','validate_allocator.py']}
 provenance['passive_checkpoint_only_interval']=a.checkpoint_every
 provenance['camera_image_storage']='native iterable CPU cache, per-view get_image().cuda(), identical float32 pixels' if cfg.dataset.type=='iterable' else 'native '+str(cfg.dataset.type)+' image mode'
 provenance['resume_seed_behavior']='Native checkpoint restore with seed0 reset; optimizer/Gaussian state restored, upstream capture does not save RNG trajectory' if a.resume else 'Fresh native seed0 trajectory'
 provenance['checkpoint_phase']='Native Scene.save phase precedes this iteration densification/optimizer step; official restore begins at saved_iteration+1. Exact uninterrupted trajectory is not claimed.'
 import diff_quadratic_rasterization as raster
 provenance['compiled_quadratic_extension']=raster._C.__file__
 provenance['compiled_quadratic_extension_sha256']=hashlib.sha256(Path(raster._C.__file__).read_bytes()).hexdigest()
 provenance['adapter_provenance_sha256']=hashlib.sha256((q.ROOT/'adapter_provenance.json').read_bytes()).hexdigest()
 q.write(q.ROOT/('smoke_training_parameters.json' if a.smoke else 'actual_training_parameters.json'),provenance)
 try:module.training(cfg,test,save,save,a.resume,-1)
 except torch.OutOfMemoryError as error:
  failed_time=time.time();f.close()
  export=failed_native_export(error,state.get('scene'),run_id)
  run_record.update(status='failed_cuda_oom',end_time=failed_time,duration_seconds=failed_time-started,
   failure_artifact_export_seconds=time.time()-failed_time,error=str(error),**export)
  execution_record(run_record);q.write(q.ROOT/'FAILURE.json',run_record)
  q.write(q.ROOT/'STATUS.json',dict(run_record,status='failed',stage='native_training_cuda_oom'))
  raise
 torch.cuda.synchronize();f.close()
 scene=state['scene'];final=out/f'point_cloud/iteration_{a.iterations}/point_cloud.ply'
 assert final.is_file() and (out/f'chkpnt{a.iterations}.pth').is_file()
 segment_duration=time.time()-started
 run_record.update(status='complete',end_time=time.time(),duration_seconds=segment_duration)
 records=execution_record(run_record)
 total_training_duration=sum(float(r.get('duration_seconds',0)) for r in records if r.get('status') in ['complete','failed_cuda_oom'])
 result=dict(status='complete',iterations=a.iterations,quadratic_primitives=len(scene.gaussians.get_xyz),training_wall_seconds=total_training_duration,
  successful_segment_wall_seconds=segment_duration,execution_run_id=run_id,resume_checkpoint=str(Path(a.resume).resolve()) if a.resume else None,
  training_wall_definition='All recorded native training attempts, including the retained failed initial attempt; successful segment reported separately',
  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,checkpoint=str((out/f'chkpnt{a.iterations}.pth').resolve()),native_quadratic_ply=str(final.resolve()),physical_gpu=1,seed=0)
 recovery_request=q.ROOT/'RECOVERY_REQUEST.json'
 if a.resume and recovery_request.is_file():
  request=q.json.loads(recovery_request.read_text())
  if Path(a.resume).resolve()==Path(request['resume_checkpoint']).resolve():
   prefix=float(request['successful_prefix_wall_seconds'])
   result.update(successful_prefix_wall_seconds=prefix,successful_model_training_wall_seconds=prefix+segment_duration,
    discarded_initial_attempt_wall_seconds=max(0.0,request['failed_attempt_wall_seconds']-prefix),
    resume_iteration=request['resume_iteration'],successful_model_time_scope='Saved initial prefix plus successful resumed segment; actual all-attempt cost remains training_wall_seconds')
 q.write(complete,result);q.write(q.ROOT/'STATUS.json',dict(result,stage='smoke_complete' if a.smoke else 'native_training_complete'))
 print(q.json.dumps(result),flush=True)
if __name__=='__main__':main()

