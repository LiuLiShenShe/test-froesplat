"""Finish common geometry diagnostics and review figures after all five trainings."""
import fcntl,json,os,subprocess
from runtime import OUT,ROOT,PYTHONS,write
from pipeline import stage

if __name__=='__main__':
 with (OUT/'summary/gpu1.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  try:
   methods=['3DGS','2DGS','GGGS','gsplat','MCMC_3DGS']
   assert all((OUT/'gaussian'/m/'complete.json').exists() for m in methods),'A training stage failed; inspect progress and logs before resuming'
   for method in methods:
    folder=OUT/'gaussian'/method/'evaluation'
    stage('evaluate_full_'+method,PYTHONS[method],'evaluate.py',[method],folder/'evaluation_v2_complete.json')
    stage('multiview_'+method,PYTHONS['GGGS'],'multiview.py',[method],folder/'multiview_consistency.json')
    stage('viewer_'+method,PYTHONS['GGGS'],'viewer.py',[method],OUT/'gaussian'/method/'viewer_provenance.json')
   subprocess.run([str(PYTHONS['GGGS']),str(OUT/'scripts/report.py')],cwd=ROOT,check=True)
   write(OUT/'summary/progress.json',dict(status='gaussian_review_ready',requires='Inspect all same-camera results and self-consistency; choose1–2 before conditional cleaning and mesh comparison'))
  except BaseException as e:
   write(OUT/'summary/after_training_error.json',dict(error=str(e)));raise
