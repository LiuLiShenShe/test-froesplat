"""Durable serial manifest runner; stages are resumed, never guessed complete."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
from batch_common import *
import subprocess,argparse,fcntl,traceback

STAGES=[('colmap','batch_prepare.py',PYTHON),('sam3','batch_sam.py',PYTHON),('calibration','batch_calibrate.py',PYTHON),('gggs','batch_train.py',GPU_PYTHON),('metric','auto_metric_scale.py',PYTHON),('tsdf','batch_mesh.py',GPU_PYTHON),('instances','batch_instances.py',PYTHON),('phenotype','batch_measure.py',PYTHON),('manual','batch_manual.py',PYTHON),('report','batch_report.py',PYTHON),('validation','batch_validate.py',PYTHON)]

def summarize():
 path=CODE/'batch_report.py'
 if path.exists():subprocess.run([str(UV),'run','--no-project','--python',str(PYTHON),str(path),'--summary-only'],cwd=ROOT,check=True)

def main():
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--all',action='store_true');group.add_argument('--plant');parser.add_argument('--resume',action='store_true',help='Reuse plants whose complete delivery has already passed validation');parser.add_argument('--from-stage',choices=[s[0] for s in STAGES]);parser.add_argument('--until-stage',choices=[s[0] for s in STAGES]);args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary').mkdir(exist_ok=True);lock=(OUT/'summary/batch_driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 manifest=read(MANIFEST);names=[r['plant'] for r in manifest['plants']];order=['DouBanLv1','CaoMei1']+[n for n in names if n not in ['DouBanLv1','CaoMei1']];order=order if args.all else [args.plant]
 if any(p not in names for p in order):raise ValueError('Plant must be in frozen manifest')
 reused=[]
 if args.resume:
  for plant in order:
   validation=OUT/plant/'summary/validation.json'
   if validation.exists() and read(validation).get('delivery_complete'):reused.append(plant)
  order=[plant for plant in order if plant not in reused]
 configuration=OUT/'summary/run_configuration.json'
 if configuration.exists():write(OUT/'summary/run_history'/f'{time.time_ns()}.json',read(configuration))
 write(configuration,dict(manifest=signature(MANIFEST),original_manifest=manifest,order=order,reused_validated_plants=reused,resume=args.resume,training=dict(iterations=30000,sh_degree=3,sg_degree=0,seed=0,true_colmap_only=True),physical_gpu=1,gpu_serial=True,algorithms=['Native GGGS','Selected official boundedTSDF','SAM3'],forbidden_stages=['C1','C2','five_algorithm_selection'],driver_pid=os.getpid(),started_unix=time.time()))
 failures=[]
 for plant in order:
  ctx=Context(plant);(ctx.out/'logs').mkdir(exist_ok=True);start=next((i for i,s in enumerate(STAGES) if s[0]==args.from_stage),0);end=next((i+1 for i,s in enumerate(STAGES) if s[0]==args.until_stage),len(STAGES))
  for stage,script,python in STAGES[start:end]:
   logpath=ctx.out/'logs'/f'{stage}.log';command=[str(UV),'run','--no-project','--python',str(python),str(CODE/script),plant];record=dict(plant=plant,stage=stage,status='running',command=command,driver_pid=os.getpid(),started_unix=time.time());write(OUT/'summary/progress.json',record);summarize();print('BATCH_START',plant,stage,flush=True)
   try:
    with logpath.open('a') as log:
     child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',TORCH_CUDA_ARCH_LIST='8.6',CUDA_HOME='/usr/local/cuda-12.6',MAX_JOBS='4'),stdout=log,stderr=subprocess.STDOUT);record['child_pid']=child.pid;write(OUT/'summary/progress.json',record);rc=child.wait()
    record.update(status='complete' if rc==0 else 'failed',returncode=rc,ended_unix=time.time());write(ctx.out/'logs'/f'{stage}_process.json',record)
    if rc:raise RuntimeError(f'{plant}/{stage} failed, see {logpath}')
    summarize()
   except BaseException as error:
    if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
    failures.append(dict(plant=plant,stage=stage,error=str(error)));ctx.status(stage,'failed',error=str(error),log=str(logpath));write(OUT/'summary/failures.json',failures);print('BATCH_FAILED',plant,stage,str(error),flush=True);summarize();break
  else:print('BATCH_PLANT_FINISHED',plant,flush=True)
 write(OUT/'summary/progress.json',dict(status='finished_with_failures' if failures else 'finished',driver_pid=os.getpid(),plants_attempted=len(order),failures=failures,ended_unix=time.time()));summarize()

if __name__=='__main__':main()
