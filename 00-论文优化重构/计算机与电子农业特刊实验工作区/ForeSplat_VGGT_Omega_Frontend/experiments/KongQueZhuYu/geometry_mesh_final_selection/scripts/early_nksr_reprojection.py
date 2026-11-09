import os,time,subprocess,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
python='/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'
for method in ['M1_Omega_NKSR','M2_Standard_NKSR','M3_E10_GGGS_NKSR','E8_GGGS_NKSR','E11_2DGS_NKSR']:
 folder=root/'meshes'/method if method.startswith('M') else root/'meshes/supplementary'/method
 while not (folder/'complete.json').exists():
  if (folder/'RESOURCE_LIMIT.json').exists():
   print(method,'optional resource-limited, recorded',flush=True);raise SystemExit(0)
  time.sleep(5)
 command=['uv','run','--no-project','--python',python,'python',str(root/'scripts/evaluate_meshes.py'),'--mesh-dir',str(folder),'--method',method]
 rec=dict(stage='early_reprojection_'+method,command=command,started_unix=time.time(),execution='Read complete mesh; independent CPU only; per-evaluation output lock prevents duplicates')
 with (root/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 with (root/'logs'/('early_reprojection_'+method+'.log')).open('a') as f: rc=subprocess.call(command,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
 rec.update(returncode=rc,ended_unix=time.time())
 with (root/'configs/actual_commands.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 if rc:raise RuntimeError((method,rc))
 print(method,'early208raw/post evaluation complete',flush=True)
