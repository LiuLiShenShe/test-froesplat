from pathlib import Path
import subprocess,os,json,time,sys
ROOT=Path(__file__).resolve().parents[1];SEL=ROOT.parent;PY=ROOT/'.venv/bin/python'
tasks=[('P2_baseline',SEL/'P2_GGGS_Cleaning/cleaned_gaussians.ply',ROOT/'task0_gaussian_cleaning/diagnosis/P2_preview')]
for name in ['round_01','round_02','round_03','additional_rounds/aggressive_delete']:
    d=ROOT/'task0_gaussian_cleaning'/name;tasks.append((name,d/'gaussians.ply',d/'preview'))
for name,ply,out in tasks:
    command=['uv','run','--no-project','--python',str(PY),'python',str(ROOT/'scripts/preview.py'),'--ply',str(ply),'--out',str(out)]
    log=ROOT/'logs'/('preview_'+name.replace('/','_')+'.log');start=time.time()
    print('starting',name,flush=True)
    with log.open('a') as f:code=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
    with (ROOT/'actual_commands.jsonl').open('a') as f:f.write(json.dumps(dict(stage='preview_'+name,argv=command,returncode=code,seconds=time.time()-start,log=str(log)))+'\n')
    if code:sys.exit(code)
    print('finished',name,flush=True)
