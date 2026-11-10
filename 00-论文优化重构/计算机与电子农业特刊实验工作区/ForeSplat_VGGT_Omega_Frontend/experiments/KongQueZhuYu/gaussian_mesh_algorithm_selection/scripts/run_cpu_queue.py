"""CPU fusion/evaluation queue; historical output never modified."""
import os,sys,time,json,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];BASE=OUT.parents[2];OLD=OUT.parent/'geometry_mesh_final_selection'
PY='/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def wait(p):
    while not p.exists():
        write(OUT/'CPU_QUEUE_STATUS.json',dict(status='waiting_for_dependency',dependency=str(p),pid=os.getpid()));time.sleep(5)
def run(name,args,required):
    if required.exists():return
    cmd=['uv','run','--no-project','--python',PY,'python',*map(str,args)];log=OUT/'logs'/f'{name}.log'
    write(OUT/'CPU_QUEUE_STATUS.json',dict(status='running',stage=name,command=cmd,pid=os.getpid()))
    start=time.time()
    with log.open('a') as f:
        p=subprocess.Popen(cmd,cwd=BASE,stdout=f,stderr=subprocess.STDOUT)
        write(OUT/'CPU_CHILD_PROCESS.json',dict(stage=name,pid=p.pid,log=str(log)));rc=p.wait()
    if rc or not required.exists():
        write(OUT/'CPU_QUEUE_STATUS.json',dict(status='failed',stage=name,returncode=rc,log=str(log)));raise SystemExit(rc or 1)
    with (OUT/'actual_commands.jsonl').open('a') as f:f.write(json.dumps(dict(stage=name,command=cmd,cwd=str(BASE),returncode=rc,wall_seconds=time.time()-start))+'\n')
    print(name,'complete',flush=True)
def evaluate(name,folder,inp,stages=['raw','post']):
    run(name,[OUT/'scripts/evaluate_mesh.py','--mesh-dir',folder,'--input',inp,'--method',name,'--stages',*stages],folder/'unified_evaluation/complete.json')
p1=OUT/'P1_GGGS_Tetrahedra';wait(p1/'complete.json');evaluate('P1',p1,OLD/'surface_inputs/e10_gggs_surface',['raw','binary','post'])
p4=OUT/'P4_2DGS_Depth_Mesh';wait(p4/'complete.json')
for name in ['expected_mask','median_mask','expected_mask_alpha','median_mask_alpha','median_mask_alpha_voxel075','median_mask_alpha_voxel125']:
    evaluate('P4_'+name,p4/name,OLD/'surface_inputs/e11_2dgs_surface')
p2=OUT/'P2_GGGS_Cleaning';wait(p2/'surface_inputs/render_complete.json')
run('P2_TSDF',[OUT/'scripts/fuse_gggs.py','--input',p2/'surface_inputs','--out',p2/'TSDF'],p2/'TSDF/complete.json')
evaluate('P2_TSDF',p2/'TSDF',p2/'surface_inputs');wait(p2/'Tetrahedra/complete.json');evaluate('P2_Tetra',p2/'Tetrahedra',p2/'surface_inputs',['raw','binary','post'])
p3=OUT/'P3_GS2Mesh';wait(p3/'fusion_shared_depth_range_complete.json')
run('P3_evaluation',[p3/'scripts/evaluate_p3.py'],p3/'meshes/evaluation/complete.json')
run('P3_shared_evaluation',[p3/'scripts/evaluate_p3.py','--shared-depth-range'],p3/'meshes_shared_depth_range/evaluation/complete.json')
run('P3_visualization',[p3/'scripts/build_p3_views.py'],p3/'visualizations/protocol.json')
e12=OUT/'E12_GGGS_EO';wait(e12/'surface_inputs/render_complete.json')
run('E12_TSDF',[OUT/'scripts/fuse_gggs.py','--input',e12/'surface_inputs','--out',e12/'TSDF'],e12/'TSDF/complete.json')
evaluate('E12_TSDF',e12/'TSDF',e12/'surface_inputs');wait(e12/'Tetrahedra/complete.json');evaluate('E12_Tetra',e12/'Tetrahedra',e12/'surface_inputs',['raw','binary','post'])
write(OUT/'CPU_QUEUE_STATUS.json',dict(status='new_mesh_queue_complete',pid=os.getpid()))

