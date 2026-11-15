"""Durable serial driver for exactly the three authorized plants."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
import argparse
import fcntl
import json
import subprocess
import time
import traceback
from .common import ROOT,HISTORY,OUT,PHENOTYPE,PLANTS,guard_memory,hash_file,stamp,write


def initial_protocol():
    inputs={}
    for plant in PLANTS:
        scene=ROOT/f'configs/scenes/{plant}.json'
        spec=json.loads(scene.read_text())
        inputs[plant]=dict(original=stamp(spec['model']),scene=stamp(scene),
            C1_raw=stamp(ROOT/f'experiments/C1_C2_all19/{plant}/C1_raw/cleaned_gaussians.ply'))
    sources=[HISTORY/'scripts'/name for name in ['render_surface.py','prepare_rendered_inputs.py',
        'surface_protocol.py','extract_nksr.py','mesh_artifacts.py','evaluate_meshes.py']]
    sources.append(PHENOTYPE/'scripts/phenotype_core.py')
    return dict(plants=PLANTS,inputs=inputs,historical_source_sha256={str(p):hash_file(p) for p in sources},
        gpu='physical GPU1 only; serial stages and plants',io_limit_mib_s=20,
        surface_rss_guard_gib=16,other_stage_rss_guard_gib=24,minimum_system_available_gib=10,
        no_new_training=True,no_new_cleaning=True,no_new_nksr_install_or_build=True,
        software_reference=str(HISTORY/'configs/nksr_environment_ready.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plants',nargs='+',choices=PLANTS,default=PLANTS)
    args=parser.parse_args()
    summary=OUT/'summary';summary.mkdir(parents=True,exist_ok=True)
    lock=(summary/'execution.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (summary/'protocol.json').exists():
        write(summary/'protocol.json',initial_protocol())
    protocol=json.loads((summary/'protocol.json').read_text())
    errors=[]
    mapping=[('surface','surface'),('reconstruct','mesh'),('evaluate','visualizations'),('phenotype','phenotype')]
    for plant in args.plants:
        scene=ROOT/f'configs/scenes/{plant}.json'
        input_ply=ROOT/f'experiments/C1_C2_all19/{plant}/C1_raw/cleaned_gaussians.ply'
        assert stamp(input_ply)==protocol['inputs'][plant]['C1_raw']
        for module,folder in mapping:
            if (OUT/plant/folder/'complete.json').exists():
                print('reuse completed',plant,module,flush=True);continue
            guard_memory()
            python=HISTORY/'.venv-nksr/bin/python' if module=='reconstruct' else ROOT.parent/'.venv/bin/python'
            command=['uv','run','--no-project','--python',str(python),'python','-m',f'nksr_pipeline.{module}',
                '--scene',str(scene),'--input-ply',str(input_ply),'--output',str(OUT/plant)]
            record=dict(plant=plant,stage=module,command=command,started_unix=time.time(),physical_gpu=1 if module!='phenotype' else None)
            write(summary/'progress.json',dict(status='running',driver_pid=os.getpid(),**record))
            print('START',plant,module,flush=True)
            with (summary/'actual_commands.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            try:
                before=time.perf_counter()
                with (OUT/plant/f'{module}_run.log').open('a') as log:
                    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                record.update(returncode=result.returncode,entrypoint_wall_s=time.perf_counter()-before,ended_unix=time.time())
                with (summary/'actual_commands.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                if result.returncode:raise RuntimeError(f'{plant}/{module} failed ({result.returncode}); inspect {module}_run.log')
                assert (OUT/plant/folder/'complete.json').exists()
                print('DONE',plant,module,flush=True)
            except Exception:
                errors.append(dict(plant=plant,stage=module,error=traceback.format_exc()))
                write(summary/'errors.json',errors)
                print(errors[-1]['error'],flush=True)
                break
    for plant in PLANTS:
        scene=json.loads((ROOT/f'configs/scenes/{plant}.json').read_text())
        assert stamp(scene['model'])==protocol['inputs'][plant]['original']
        assert stamp(ROOT/f'experiments/C1_C2_all19/{plant}/C1_raw/cleaned_gaussians.ply')==protocol['inputs'][plant]['C1_raw']
    complete=all((OUT/p/f/'complete.json').exists() for p in PLANTS for _,f in mapping)
    write(summary/'run_complete.json',dict(completed=complete and not errors,errors=errors,plants=PLANTS))
    if complete and not errors:
        # CPU-only image preparation runs after all serial GPU stages.
        from .references import main as references
        references()
    print('THREE PLANT PIPELINE FINISHED',complete,'errors',len(errors),flush=True)


if __name__=='__main__':main()
