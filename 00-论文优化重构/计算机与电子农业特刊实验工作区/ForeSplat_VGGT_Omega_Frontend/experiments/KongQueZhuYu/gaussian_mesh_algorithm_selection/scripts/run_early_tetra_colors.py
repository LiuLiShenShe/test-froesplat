import subprocess,json,time,os
from pathlib import Path
p=Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES']='1'
py=str(p.parents[2]/'.venv/bin/python')
for name,folder,ply in [('P1',p/'P1_GGGS_Tetrahedra',p.parent/'geometry_rgb_dual_frontend/E10_Standard_Corrected_Interface/checkpoints/point_cloud/iteration_30000/point_cloud.ply'),('P2',p/'P2_GGGS_Cleaning/Tetrahedra',p/'P2_GGGS_Cleaning/cleaned_gaussians.ply'),('E12',p/'E12_GGGS_EO/Tetrahedra',p/'E12_GGGS_EO/checkpoints/point_cloud/iteration_30000/point_cloud.ply')]:
 dest=folder/'recon_post_vertex_color.ply'
 if dest.with_suffix('.colour.json').exists():continue
 cmd=['uv','run','--no-project','--python',py,'python',str(p/'scripts/color_tetra.py'),'--mesh',str(folder/'recon_post.ply'),'--ply',str(ply),'--out',str(dest)]
 start=time.time()
 with (p/'logs'/('early_'+name+'_tetra_color.log')).open('a') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=p).returncode
 with (p/'actual_commands.jsonl').open('a') as f:f.write(json.dumps(dict(stage='early_'+name+'_tetra_color',command=cmd,returncode=rc,wall_seconds=time.time()-start))+'\n')
 if rc:raise SystemExit(rc)
 print(name,'native SH vertex color complete',flush=True)
