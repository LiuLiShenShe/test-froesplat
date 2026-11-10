"""One sequential queue stage: official QGS train, render, evaluate and native mesh."""
from pathlib import Path
import os,subprocess,json,time,argparse,fcntl
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT.parent/'scripts'))
from queue_io import write_status
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--resume',type=Path)
 parser.add_argument('--checkpoint-every',type=int,default=1000)
 parser.add_argument('--allocator',default='expandable_segments:True')
 parser.add_argument('--native-preset',choices=['base','Mip'],default='base')
 args=parser.parse_args()
 assert args.checkpoint_every>=0
 if args.resume is not None:
  args.resume=args.resume.resolve();assert args.resume.is_file(),'Actual resume checkpoint missing'
 elif not (ROOT/'training_complete.json').is_file() and list((ROOT/'checkpoints').glob('chkpnt*.pth')):
  raise SystemExit('Existing partial native checkpoints retained. Explicit --resume <real checkpoint> is required; a fresh run must not overwrite them.')
 lock=(ROOT/'native_driver.lock').open('a+')
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Another E13 native driver holds the lock; no second GPU job launched.')
 os.environ['PYTORCH_CUDA_ALLOC_CONF']=args.allocator
 write_status(ROOT/'active_driver.json',dict(status='running',pid=os.getpid(),resume_checkpoint=str(args.resume) if args.resume else None,allocator=args.allocator,native_preset=args.native_preset))
 stages=[('validate_qgs_kernel.py','kernel_validation.json'),('validate_allocator.py','allocator_validation.json'),('train_qgs.py','training_complete.json'),('audit_native_loss_log.py','native_loss_log_audit.json'),('audit_checkpoints.py','checkpoint_provenance_validation.json'),('native_primitive_stats.py','native_geometry/statistics.json'),('render_qgs.py','surface_inputs/render_complete.json'),('render_qgs_free.py','free_views/complete.json'),
         ('evaluate_qgs.py','evaluation_complete.json'),('export_qgs_centers.py','viewer_exports/INDEX.json'),('extract_qgs_mesh.py','meshes/native_TSDF/complete.json'),('smoke_native_viewer.py','native_viewer/smoke/complete.json'),('write_execution_summary.py','execution_summary_complete.json')]
 started=time.time();logs=ROOT/'logs';logs.mkdir(exist_ok=True)
 for script,result in stages:
  if (ROOT/result).is_file():continue
  cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts'/script)]
  if script=='train_qgs.py':
   if args.resume is not None:cmd+=['--resume',str(args.resume)]
   cmd+=['--checkpoint-every',str(args.checkpoint_every),'--native-preset',args.native_preset]
  write_status(ROOT/'queue_status.json',dict(status='running',stage=script,command=cmd,pid=os.getpid()))
  with (logs/(script.removesuffix('.py')+'.log')).open('a') as f:
   p=subprocess.run(cmd,cwd=ROOT/'adapted',stdout=f,stderr=subprocess.STDOUT,env=os.environ)
  if p.returncode or not (ROOT/result).is_file():
   write_status(ROOT/'queue_status.json',dict(status='failed',stage=script,returncode=p.returncode,log=str(logs/(script.removesuffix('.py')+'.log')),pid=os.getpid()))
   (ROOT/'active_driver.json').write_text(json.dumps(dict(status='failed',pid=os.getpid(),stage=script,returncode=p.returncode),indent=2)+'\n')
   raise SystemExit(p.returncode or 1)
 write_status(ROOT/'queue_status.json',dict(status='complete',elapsed_seconds=time.time()-started,allocator=args.allocator,resume_checkpoint=str(args.resume) if args.resume else None))
 write_status(ROOT/'active_driver.json',dict(status='complete',pid=os.getpid(),resume_checkpoint=str(args.resume) if args.resume else None))
 print('E13 native QGS stages complete',flush=True)
if __name__=='__main__':main()

