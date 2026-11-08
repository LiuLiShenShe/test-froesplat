"""One CPU launcher for independently isolated geometry backends.

The subprocess imports only its own model environment. Consumers share Bundle v1.
"""
import argparse
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BACKENDS={
    'fast3r':('.venv-fast3r',['frontends/fast3r/run.py']),
    'amb3r-off':('.venv-amb3r',['-m','frontends.amb3r.base','--mode','off']),
    'amb3r-on':('.venv-amb3r',['-m','frontends.amb3r.base','--mode','on']),
    'amb3r-pair':('.venv-amb3r',['-m','frontends.amb3r.base','--mode','both']),
    'amb3r-sfm':('.venv-amb3r',['frontends/sfm/run.py','--model','amb3r']),
    'fast3r-sfm':('.venv-fast3r',['frontends/sfm/run.py','--model','fast3r']),
    'evaluate':('.venv-fast3r',['frontends/evaluate.py']),
}

def run_frontend(method,args=(),check=True):
    environment,entrypoint=BACKENDS[method]
    python=ROOT/environment/'bin'/'python'
    if not python.is_file():raise FileNotFoundError(python)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1')
    return subprocess.run(['uv','run','--no-project','--python',str(python),*entrypoint,*args],
                          cwd=ROOT,env=env,check=check)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('method',choices=list(BACKENDS))
    options,extra=p.parse_known_args();raise SystemExit(run_frontend(options.method,extra,check=False).returncode)
