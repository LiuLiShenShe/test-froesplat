from common import *
import subprocess,time
def main():
    p=argparse.ArgumentParser();p.add_argument('--gaussian-ply',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    py=ROOT/'.venv/bin/python'
    stages=[('render',['render_cleaned_stereo.py','--gaussian-ply',str(a.gaussian_ply.resolve()),'--output',str(a.output)]),('stereo',['run_cleaned_stereo_fusion.py','--output',str(a.output),'--stage','stereo']),('fusion',['run_cleaned_stereo_fusion.py','--output',str(a.output),'--stage','fusion','--shared-depth-range'])]
    for name,args in stages:
        cmd=['uv','run','--no-project','--python',str(py),'python',str(ROOT/'scripts'/args[0]),*args[1:]];log=a.output/(name+'.log');start=time.time();write(a.output/'STATUS.json',dict(stage=name,status='running',gaussian_sha256=sha(a.gaussian_ply)))
        with log.open('a') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        with (ROOT/'actual_commands.jsonl').open('a') as f:f.write(json.dumps(dict(stage=a.output.name+'_'+name,argv=cmd,returncode=code,seconds=time.time()-start,log=str(log)))+'\n')
        if code:raise RuntimeError(f'{name} failed: {log}')
        print(a.output.name,name,'complete',flush=True)
    cmd=['uv','run','--no-project','--python',str(py),'python',str(SEL/'scripts/evaluate_mesh.py'),'--mesh-dir',str(a.output/'meshes_shared_depth_range'),'--input',str(OLD/'surface_inputs/e10_gggs_surface'),'--method',a.output.name]
    write(a.output/'STATUS.json',dict(stage='208_mesh_evaluation',status='running'))
    with (a.output/'evaluation.log').open('a') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
    if code:raise RuntimeError('mesh evaluation failed')
    write(a.output/'STATUS.json',dict(status='extraction_and208_evaluation_complete',gaussian_sha256=sha(a.gaussian_ply)))
if __name__=='__main__':main()
