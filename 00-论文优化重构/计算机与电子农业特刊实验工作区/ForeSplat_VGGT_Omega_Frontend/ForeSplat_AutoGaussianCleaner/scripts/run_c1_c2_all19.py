"""Serial all-19 frozen-rule experiment; resumes completed stages without overwriting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/C1_C2_all19'
SUMMARY=OUT/'summary'

def invoke(cmd,log):
    with log.open('x') as stream:
        result=subprocess.run([sys.executable,*cmd],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{cmd[0]} failed ({result.returncode}), see {log}')

def main():
    entries=json.loads((ROOT/'configs/plant_manifest.json').read_text())['plants']
    assert len(entries)==19
    # The anchor is always verified before any new plant is processed.
    entries=sorted(entries,key=lambda x:x['plant']!='KongQueZhuYu')
    errors=[]
    for phase in ['cleanup','evaluation']:
        for order,entry in enumerate(entries):
            plant=entry['plant'];out=OUT/plant;out.mkdir(parents=True,exist_ok=True)
            try:
                if phase=='cleanup':
                    if (out/'batch_report.json').exists():continue
                    cmd=['fast_clean_c1_c2.py','--scene',entry['scene'],'--output',str(out)]
                    reuse=SUMMARY/(plant+'_reuse.json')
                    if reuse.exists():cmd+=['--reuse',str(reuse)]
                    if plant=='KongQueZhuYu':cmd+=['--historical-ids',str(SUMMARY/'historical_ids.json')]
                    print(f'CLEANUP {order+1}/19 {plant}',flush=True)
                    invoke(cmd,out/'cleanup.log')
                    if plant=='KongQueZhuYu':
                        for method,count in [('C1_raw',443681),('C2_raw',428967)]:
                            r=json.loads((out/method/'cleanup_report.json').read_text())
                            assert r['remaining_gaussians']==count and r['historical_verification']['identical_ids']
                else:
                    if (out/'comparison/paired_quality.json').exists():continue
                    if not (out/'batch_report.json').exists():raise RuntimeError('Cleanup unavailable')
                    manifest=out/'comparison_models.json'
                    manifest.write_text(json.dumps({m:str(out/m/'cleaned_gaussians.ply') for m in ['C1_raw','C2_raw']},indent=2)+'\n')
                    cmd=['-m','fast_ply.evaluate','--scene',entry['scene'],'--output',str(out/'comparison'),
                        '--external-models',str(manifest),'--views','4','--max-side','960','--skip-baselines','--skip-ablations']
                    if plant in ['WangWenCao1','ChangShouHua1']:
                        old=json.loads((ROOT/'reports/C2_fast_validation'/plant/'paired_quality.json').read_text())
                        orders=[0,2,7,15] if plant=='WangWenCao1' else [0,7,10,15]
                        cmd+=['--view-indices',*[str(old['observed_indices'][i]) for i in orders]]
                    print(f'EVALUATION {order+1}/19 {plant}',flush=True)
                    before=time.perf_counter();invoke(cmd,out/'evaluation.log')
                    (out/'evaluation_runtime.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-before,gpu='physical GPU1',excluded_from_cleanup=True),indent=2)+'\n')
                print('DONE',phase,plant,flush=True)
                (SUMMARY/'progress.json').write_text(json.dumps(dict(phase=phase,last_completed=plant,errors=errors),indent=2)+'\n')
            except Exception:
                error=dict(phase=phase,plant=plant,error=traceback.format_exc());errors.append(error)
                print(error['error'],flush=True)
                if plant=='KongQueZhuYu' and phase=='cleanup':raise
                # Rendering/over-pruning never prevents later plants running.
                (SUMMARY/'errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    (SUMMARY/'run_complete.json').write_text(json.dumps(dict(completed=not errors,errors=errors),indent=2)+'\n')
    print('ALL 19 PIPELINE FINISHED',len(errors),'errors',flush=True)

if __name__=='__main__':main()
