"""Complete historical208 evaluation after all new formal algorithms; preserve old records."""
import os,json,time,subprocess
from pathlib import Path
from queue_io import read_status, write_status
OUT=Path(__file__).resolve().parents[1];BASE=OUT.parents[2];PLANT=OUT.parent;OLD=PLANT/'geometry_mesh_final_selection';DUAL=PLANT/'geometry_rgb_dual_frontend'
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MPLBACKEND='Agg')
while True:
    marker=OUT/'E14_2D_SuGaR/STATUS.json'
    upstream=read_status(marker)
    if upstream.get('status')=='complete':break
    if upstream.get('status') in ['failed','partial']:
        write_status(OUT/'HISTORY_QUEUE_STATUS.json',dict(status='failed_dependency',upstream=upstream,pid=os.getpid()))
        raise RuntimeError('E14 requires recovery before full-history finalization')
    write_status(OUT/'HISTORY_QUEUE_STATUS.json',dict(status='waiting_for_E14_complete',pid=os.getpid()));time.sleep(5)
source_std=BASE.parent/'ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison'
tasks=[]
for eid,name in [('E2','E2_GGGS_EO'),('E3','E3_GGGS_Foreground'),('E4','E4_ForeSplat_GGGS')]:
    root=PLANT/name
    cmd=[OUT/'scripts/render_gggs.py','--ply',root/'checkpoints/point_cloud/iteration_30000/point_cloud.ply','--out',OUT/'comparison/history_uniform'/eid,'--experiment',eid,'--repo',BASE/'gggs_community','--reference',DUAL/'render_comparison/E1/camera_runtime.json']
    if eid in ['E2','E4']:cmd+=['--exposure',root/'exposure_params/exposure_matrices.npz']
    tasks.append((eid,cmd,BASE/'.venv/bin/python',OUT/'comparison/history_uniform'/eid/'render_complete.json'))
e6_reference=OUT/'comparison/history_uniform/E6/recovered_camera_runtime.json'
if not e6_reference.is_file():
    subprocess.run(['python3',str(OUT/'scripts/recover_E6_camera_runtime_fields.py')],check=True)
tasks.append(('E6',[OUT/'scripts/render_gggs.py','--method','2DGS','--repo','/data/fj/F2DMAS/2d-gaussian-splatting-main','--ply',source_std/'E6_2DGS_Standard_VGGT/checkpoints/point_cloud/iteration_30000/point_cloud.ply','--out',OUT/'comparison/history_uniform/E6','--experiment','E6','--reference',e6_reference],Path('/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'),OUT/'comparison/history_uniform/E6/render_complete.json'))
e12=OUT/'E12_GGGS_EO'
tasks.append(('E12_free',[OUT/'scripts/render_gggs_free.py','--ply',e12/'checkpoints/point_cloud/iteration_30000/point_cloud.ply','--out',OUT/'comparison/visualizations/E12_free'],BASE/'.venv/bin/python',OUT/'comparison/visualizations/E12_free/complete.json'))
for name,args,py,required in tasks:
    if required.exists():continue
    cmd=['uv','run','--no-project','--python',str(py),'python',*map(str,args)]
    (OUT/'HISTORY_QUEUE_STATUS.json').write_text(json.dumps(dict(status='running',stage=name,command=cmd,pid=os.getpid()),indent=2));start=time.time()
    with (OUT/'logs'/f'history_{name}.log').open('a') as f:
        p=subprocess.Popen(cmd,cwd=BASE,env=os.environ,stdout=f,stderr=subprocess.STDOUT)
        (OUT/'HISTORY_CHILD_PROCESS.json').write_text(json.dumps(dict(stage=name,pid=p.pid),indent=2));rc=p.wait()
    with (OUT/'actual_commands.jsonl').open('a') as f:f.write(json.dumps(dict(stage='history_'+name,command=cmd,returncode=rc,wall_seconds=time.time()-start))+'\n')
    if rc or not required.exists():
        (OUT/'HISTORY_QUEUE_STATUS.json').write_text(json.dumps(dict(status='failed',stage=name,returncode=rc),indent=2));raise SystemExit(rc or 1)
(OUT/'HISTORY_QUEUE_STATUS.json').write_text(json.dumps(dict(status='complete',stages=[t[0] for t in tasks]),indent=2))

