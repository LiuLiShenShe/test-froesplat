"""CPU-only independent RGB correspondence/proxy analysis, dependent models awaited."""
import os,json,time,subprocess,fcntl
from pathlib import Path
from queue_io import read_status,write_status
OUT=Path(__file__).resolve().parents[1];OLD=OUT.parent/'geometry_mesh_final_selection';DUAL=OUT.parent/'geometry_rgb_dual_frontend';PY='/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'
lock=(OUT/'analysis_execution.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(x):write_status(OUT/'ANALYSIS_QUEUE_STATUS.json',x)
def wait_complete(dependency,stage):
    while read_status(dependency).get('status')!='complete':
        for path in [OUT/'E13_QGS/queue_status.json',OUT/'E14_QUEUE_STATUS.json',OUT/'E14_2D_SuGaR/STATUS.json']:
            upstream=read_status(path)
            if upstream.get('status') in ['failed','failed_dependency']:
                status(dict(status='failed_dependency',stage=stage,dependency=str(path),upstream=upstream,pid=os.getpid()))
                raise RuntimeError(f'Upstream execution needs recovery: {path}')
        status(dict(status='waiting',stage=stage,dependency=str(dependency),pid=os.getpid()));time.sleep(5)
def run(name,args,required,dependency=None):
    if dependency:wait_complete(dependency,name)
    if read_status(required).get('status')=='complete':return
    cmd=['uv','run','--no-project','--python',PY,'python',*map(str,args)];status(dict(status='running',stage=name,pid=os.getpid(),command=cmd));start=time.time()
    with (OUT/'logs'/f'analysis_{name}.log').open('a') as f:
        child=subprocess.Popen(cmd,cwd=OUT,stdout=f,stderr=subprocess.STDOUT);(OUT/'ANALYSIS_CHILD_PROCESS.json').write_text(json.dumps(dict(stage=name,pid=child.pid),indent=2));rc=child.wait()
    with (OUT/'actual_commands.jsonl').open('a') as f:f.write(json.dumps(dict(stage='analysis_'+name,command=cmd,returncode=rc,wall_seconds=time.time()-start))+'\n')
    if rc or not required.exists():status(dict(status='failed',stage=name,returncode=rc));raise SystemExit(rc or 1)
    print(name,'complete',flush=True)
def reflection(e,root,normal,ply,aux=False):
    args=[OUT/'scripts/reflection_correspondence.py','--method',e,'--input',root,'--normal',normal,'--ply',ply]
    if aux:args+=['--aux']
    run(e+'_reflection',args,OUT/'comparison/reflection_analysis'/f'{e}_summary.json',root/'render_complete.json')
reflection('E10',OLD/'surface_inputs/e10_gggs_surface','camera',DUAL/'E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply')
reflection('P2',OUT/'P2_GGGS_Cleaning/surface_inputs','camera',OUT/'P2_GGGS_Cleaning/cleaned_gaussians.ply')
reflection('E11',OLD/'surface_inputs/e11_2dgs_surface','world',OLD/'E11_2DGS_Corrected/checkpoints/point_cloud/iteration_30000/point_cloud.ply')
run('history_mesh',[OUT/'scripts/recompute_history_mesh.py'],OUT/'comparison/history_mesh_uniform/M5_E11_2DGS_TSDF/complete.json')
reflection('E12',OUT/'E12_GGGS_EO/surface_inputs','camera',OUT/'E12_GGGS_EO/checkpoints/point_cloud/iteration_30000/point_cloud.ply')
reflection('E13',OUT/'E13_QGS/surface_inputs','camera',OUT/'E13_QGS/checkpoints/point_cloud/iteration_30000/point_cloud.ply')
wait_complete(OUT/'E14_2D_SuGaR/refinement_complete.json','E14_refinement')
reflection('E14_30K',OUT/'E14_2D_SuGaR/stage_renders/final_30000','world',OUT/'E14_2D_SuGaR/training/point_cloud/iteration_30000/point_cloud.ply')
final=read_status(OUT/'E14_2D_SuGaR/refinement_complete.json')
reflection('E14_refined_aux',OUT/'E14_2D_SuGaR/stage_renders/refined','world',Path(final['gaussian_ply']),aux=True)
status(dict(status='complete',pid=os.getpid()))
