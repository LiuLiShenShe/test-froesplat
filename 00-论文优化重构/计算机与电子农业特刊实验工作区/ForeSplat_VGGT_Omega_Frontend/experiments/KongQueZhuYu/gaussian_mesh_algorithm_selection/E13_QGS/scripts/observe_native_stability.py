"""Passive CPU observer of actual native checkpoints; never alters GPU training."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'scripts'))
from queue_io import read_status,write_status

def status(**values):
 write_status(ROOT/'recovery/STABILITY_OBSERVER_STATUS.json',dict(pid=os.getpid(),**values))

def main():
 lock=(ROOT/'recovery/stability_observer.lock').open('a')
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 observed=set()
 failures={}
 logs=ROOT/'logs';logs.mkdir(exist_ok=True)
 while True:
  candidates=[]
  meta=read_status(ROOT/'recovery/latest_native_state.provenance.json')
  if meta.get('iteration'):
   candidates.append((int(meta['iteration']),True,ROOT/'recovery/latest_native_state.pth'))
  train=read_status(ROOT/'STATUS.json')
  terminal=read_status(ROOT/'training_complete.json')
  for iteration in [7000,15000,30000]:
   # Official training_report runs before Scene.save. Wait for a later
   # sampled iteration (or the terminal completion marker), so a directly
   # written fixed PTH cannot be read while its save is still in progress.
   checkpoint=ROOT/f'checkpoints/chkpnt{iteration}.pth'
   saved=(int(train.get('iteration',train.get('iterations',0)))>iteration or
          terminal.get('status')=='complete' and int(terminal.get('iterations',0))>=iteration)
   if saved and checkpoint.is_file():
    candidates.append((iteration,False,checkpoint))
  changed=False
  for iteration,rolling,checkpoint in candidates:
   key=(iteration,rolling)
   target=ROOT/f'recovery/{"rolling" if rolling else "fixed"}_checkpoint_validation_{iteration}.json'
   if key in observed or target.is_file():
    observed.add(key);continue
   if not checkpoint.is_file():continue
   command=[sys.executable,str(ROOT/'scripts/audit_saved_native_state.py')]
   if not rolling:command+=['--checkpoint',str(checkpoint)]
   status(status='auditing_CPU_only',observed_iteration=iteration,source_checkpoint=str(checkpoint))
   with (logs/'native_stability_observer.log').open('a') as stream:
    result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='8',CUDA_VISIBLE_DEVICES=''))
   if result.returncode:
    # Atomic rolling-file and metadata writes are separate. Retry after the
    # producer settles; report the actual error without modifying its state.
    fingerprint=(iteration,checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
    failures[fingerprint]=failures.get(fingerprint,0)+1
    status(status='audit_retry_pending',iteration=iteration,returncode=result.returncode,consecutive_same_file_failures=failures[fingerprint],log=str(logs/'native_stability_observer.log'))
    if failures[fingerprint]>=3:
     status(status='failed_CPU_checkpoint_audit',iteration=iteration,returncode=result.returncode,log=str(logs/'native_stability_observer.log'),GPU_training_untouched=True)
     raise SystemExit(result.returncode)
    continue
   assert target.is_file(),'Structured audit returned without its actual checkpoint validation record'
   observed.add(key);changed=True
  if changed:
   with (logs/'native_stability_panels.log').open('a') as stream:
    result=subprocess.run([sys.executable,str(ROOT/'scripts/build_stability_panels.py')],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='8',MPLBACKEND='Agg',CUDA_VISIBLE_DEVICES=''))
   if result.returncode:
    status(status='failed_CPU_visualization',returncode=result.returncode);raise SystemExit(result.returncode)
  if (ROOT/'training_complete.json').is_file() and not (ROOT/'native_loss_log_audit.json').is_file():
   with (logs/'native_stability_loss_audit.log').open('a') as stream:
    result=subprocess.run([sys.executable,str(ROOT/'scripts/audit_native_loss_log.py')],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='8',CUDA_VISIBLE_DEVICES=''))
   if result.returncode:
    status(status='failed_CPU_terminal_loss_audit',returncode=result.returncode,GPU_training_untouched=True)
    raise SystemExit(result.returncode)
  driver=read_status(ROOT/'queue_status.json')
  if driver.get('status') in ['complete','failed']:
   status(status='complete_observed_native_checkpoints',native_driver_status=driver.get('status'),observed_snapshots=sorted([list(x) for x in observed]),numeric_quality_not_assumed=True)
   return
  status(status='waiting_for_next_real_checkpoint',native_iteration=train.get('iteration',train.get('iterations')),observed_snapshots=sorted([list(x) for x in observed]),CPU_only=True)
  time.sleep(5)

if __name__=='__main__':main()
