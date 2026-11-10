"""Sequential native E14 stages; preserve partial outputs and honest failures without stopping independent evaluation."""
import os,sys,json,time,subprocess,traceback,fcntl,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY=ROOT/'.venv/bin/python'
sys.path.insert(0,str(ROOT.parent/'scripts'))
from queue_io import write_status
def write(p,x):write_status(p,x)
def run(stage,script):
    command=['uv','run','--no-project','--python',str(PY),'python',str(ROOT/'scripts'/script)]
    event=dict(stage=stage,command=command,cwd=str(ROOT/'official'),CUDA_VISIBLE_DEVICES='1',started_unix=time.time())
    with (ROOT/'logs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    write(ROOT/'STATUS.json',dict(status='running',stage=stage,command=command))
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='1';env['MPLBACKEND']='Agg';env['PYTHONUNBUFFERED']='1'
    with (ROOT/'logs'/f'{stage}.log').open('a') as logfile:
        p=subprocess.Popen(command,cwd=ROOT/'official',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            logfile.write(line);logfile.flush();print(line,end='',flush=True)
        code=p.wait()
    event.update(returncode=code,ended_unix=time.time())
    with (ROOT/'logs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    if code:raise RuntimeError(f'{stage} exited {code}; see {ROOT/"logs"/(stage+".log")}')
def main():
    assert (ROOT/'preparation_complete.json').exists()
    assert (ROOT/'environment_ready.json').exists()
    input_audit=json.loads((ROOT/'training_and_refinement_RGB_cpu_validation.json').read_text())
    assert input_audit['status']=='passed_CPU_native_input_loaders'
    assert input_audit['all208_RGB_K_W2C_P_equal_to_E10_E11'] and input_audit['views_per_loader']==208
    for rel,digest in input_audit['source_hashes'].items():
        assert hashlib.sha256((ROOT/'official'/rel).read_bytes()).hexdigest()==digest,('Input loader changed after actual208 preflight',rel)
    failures=[];results=[]
    for stage,script,done,essential in [
        ('01_Metric3Dv2_prior','generate_priors.py','priors_complete.json',True),
        ('01a_actual_prior_numeric_and_coordinate_audit','audit_prior_outputs.py','prior_numerical_validation.json',True),
        ('02_native_2DGS30K','train_native.py','training_complete.json',True),
        ('02a_actual_native30K_state_provenance','audit_native_terminal.py','native_terminal_provenance_validation.json',False),
        ('02b_actual_DBSCAN_stage_audit','audit_pruning_stages.py','cluster_pruning/stage_provenance_validation.json',False),
        ('03_native_render_and_initial_mesh','render_and_mesh.py','native_render_mesh_complete.json',False),
        ('03b_native_surfel_visuals','render_surfel_visuals.py','visual_views/surfel/complete.json',False),
        ('04_SuGaR15K_refinement_and_texture','refine_native.py','refinement_complete.json',False),
        ('05_refined_Gaussian_render','render_refined.py','stage_renders/refined/render_complete.json',False),
        ('05b_native_refined_visuals','render_refined_visuals.py','visual_views/refined/complete.json',False),
        ('06_all208_stage_evaluation','evaluate_stage_renders.py','evaluation_complete.json',False),
        ('07_native_mesh208_evaluation','evaluate_native_meshes.py','mesh_evaluation_complete.json',False),
        ('07b_actual_Mesh_visuals','render_mesh_geometry_visuals.py','mesh_visuals_complete.json',False),
        ('08_actual_artifact_viewing_guide','write_viewing_guide.py','artifact_guide_complete.json',False),
        ('09_actual_native_HTTP_viewer_smoke','smoke_native_viewers_http.py','native_viewer_http_smoke_complete.json',False)]:
        marker=ROOT/done
        try:
            if marker.exists() and json.loads(marker.read_text()).get('status')=='complete':
                print('Verified completed stage',stage,flush=True)
            else:run(stage,script)
            if not marker.exists():raise RuntimeError('Stage returned without a completion record')
            result=json.loads(marker.read_text())
            if result.get('status')!='complete':raise RuntimeError(f'Stage evidence is {result.get("status")}; see {marker}')
            if stage=='03_native_render_and_initial_mesh':
                bad=[r for r in result['mesh_results'] if r.get('status')!='complete' or r.get('post_status')!='complete']
                if bad:raise RuntimeError('One or more actual TSDF stages lack a nonempty raw/post surface; independent stages continue')
            results.append(dict(stage=stage,status='complete',evidence=str(marker.resolve())))
        except Exception as err:
            failure=dict(stage=stage,status='failed',reason=str(err),traceback=traceback.format_exc(),evidence=str(marker.resolve()) if marker.exists() else None,log=str((ROOT/'logs'/(stage+'.log')).resolve()))
            failures.append(failure);results.append(failure)
            write(ROOT/'logs'/f'{stage}_failure.json',failure)
            print('E14 stage unavailable:',stage,str(err),flush=True)
            if essential:raise
    final=dict(status='complete' if not failures else 'partial',stages=len(results),views=208,native2d_iterations=30000,refinement_iterations=15000 if not any(f['stage'].startswith('04_') for f in failures) else None,results=results,failures=failures)
    write(ROOT/'STATUS.json',final)
    if failures:sys.exit(2)
if __name__=='__main__':
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception:
            write(ROOT/'STATUS.json',dict(status='failed',traceback=traceback.format_exc()))
            raise

