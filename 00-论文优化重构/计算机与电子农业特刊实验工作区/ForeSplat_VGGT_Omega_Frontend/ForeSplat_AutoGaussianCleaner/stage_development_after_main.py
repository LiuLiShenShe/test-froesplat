#!/usr/bin/env python3
"""Start the authorized development split only after a complete improving main comparison."""
import argparse,json,os,subprocess,sys,time
import yaml
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from ForeSplat_AutoGaussianCleaner.execution_provenance import completed_identity,resolve_variant,scene_input_fingerprint,native_renderer_fingerprint
from ForeSplat_AutoGaussianCleaner.cleaner.io import write_json,sha

def process_state(path):
    try:
        fields=path.read_text().rsplit(')',1)[1].split()
        return fields[19],fields[0]
    except (OSError,IndexError):
        return None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-output',type=Path,required=True)
    parser.add_argument('--main-pid',type=int,required=True)
    parser.add_argument('--entrypoint',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--jobs',type=int,choices=[1,2],default=1)
    args=parser.parse_args();out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    proc=Path('/proc')/str(args.main_pid)/'stat'
    state=process_state(proc);identity=state[0] if state else None
    while not(args.main_output/'COMPLETE.json').exists():
        for name in ['INVALIDATED.json','FULL_VALIDATION_FAILURE.json','FAILED.json']:
            if(args.main_output/name).exists():raise ValueError('Main comparison failed: '+str(args.main_output/name))
        state=process_state(proc)
        if state is None or state[0]!=identity or state[1]=='Z':
            raise ValueError('Main producer exited before certified completion')
        time.sleep(10)
    # COMPLETE is written by the engine before the CLI finishes its immutable
    # source/input audit. Wait for that producer to finish, then verify again.
    state=process_state(proc)
    while state is not None and state[0]==identity and state[1]!='Z':
        time.sleep(10)
        state=process_state(proc)
    if (args.main_output/'INVALIDATED.json').exists() or not(args.main_output/'COMPLETE.json').exists():
        raise ValueError('Main result failed its final immutable-source audit')
    marker=json.loads((args.main_output/'COMPLETE.json').read_text())
    report=json.loads((args.main_output/'cleanup_metrics.json').read_text())
    manifest=json.loads(args.manifest.read_text())
    job=next(j for j in manifest['plants'] if j['plant']==report['plant'] and j['group']=='development')
    specification=json.loads(Path(job['scene']).read_text())
    base=args.entrypoint.resolve().parent
    requested=yaml.safe_load((base/'configs/default.yaml').read_text())
    requested.update(yaml.safe_load(args.config.read_text()))
    requested['implementation_sources']={str(f.relative_to(base)):sha(f) for f in sorted(
        list((base/'cleaner').rglob('*.py'))+list((base/'evaluation').glob('*.py'))+[base/'auto_clean_gaussians.py',base/'execution_provenance.py'])}
    requested['scene_input_fingerprint']=scene_input_fingerprint(specification)
    requested['native_renderer_fingerprint']=native_renderer_fingerprint([specification['gggs_repo']])
    variant=resolve_variant('C5',requested)
    if not completed_identity(args.main_output,requested,specification,variant) or report['config']!=requested:
        raise ValueError('Completed main result differs from requested development source/config/scene/native renderer or resolved C5 variant')
    write_json(out/'MAIN_SOURCE_BINDING.json',dict(entrypoint=str(args.entrypoint.resolve()),entrypoint_sha256=sha(args.entrypoint),
        config_path=str(args.config.resolve()),config_sha256=sha(args.config),configuration=requested,variant=variant,
        main_output=str(args.main_output.resolve()),scope='Exact requested execution identity checked before reusing the main output or launching any development scene'))
    if (report['removed_gaussians']+report.get('refined_gaussians',0)<=0 or report['outcome']!='measured_improvement_with_residuals'
            or max(report['artifact_reduction'],report.get('free_artifact_reduction',0))<=0):
        raise ValueError('Main comparison has no actual improving Gaussian modification')
    if not report['initial']['views'] or len(report['final']['views'])!=len(report['initial']['views']):
        raise ValueError('Main comparison lacks complete matched-view evaluation')
    write_json(out/'MAIN_COMPARISON_GATE.json',dict(main_output=str(args.main_output.resolve()),
        native_sha256=marker['native_sha256'],removed_gaussians=report['removed_gaussians'],
        background_reduction=report['artifact_reduction'],independent_test_started=False,
        scope='The unchanged C0-C4 baselines and this complete C5 main experiment precede running the fixed candidate version on the development split. This gate is not the final global-method decision.'))
    if json.loads(Path(job['scene']).read_text())!=marker['scene']:
        raise ValueError('Main scene differs from the recorded development scene')
    reuse=out/'development'/report['plant'];reuse.parent.mkdir(parents=True,exist_ok=True)
    if reuse.exists() or reuse.is_symlink():raise FileExistsError(reuse)
    reuse.symlink_to(args.main_output.resolve(),target_is_directory=True)
    write_json(out/'MAIN_RESULT_REUSE.json',dict(plant=report['plant'],output=str(reuse),
        completed_main_output=str(args.main_output.resolve()),input_sha256=marker['input_sha256'],native_sha256=marker['native_sha256'],
        scope='Reuse this exact already completed same-source/config/native-scene result. The batch resume path checks every exported artifact and full configuration; the cleaned model is never used as original input.'))
    command=['uv','run','--no-project','--python',sys.executable,'python','-u',str(args.entrypoint.resolve()),
        '--manifest',str(args.manifest.resolve()),'--method','C5','--config',str(args.config.resolve()),
        '--group','development','--output-root',str(out),'--resume','--jobs',str(args.jobs)]
    write_json(out/'STAGED_DEVELOPMENT_COMMAND.json',dict(command=command,started=time.time()))
    with (out/'development_batch.log').open('a') as log:
        code=subprocess.call(command,cwd=Path(__file__).resolve().parent.parent,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},stdout=log,stderr=subprocess.STDOUT)
    write_json(out/'STAGED_DEVELOPMENT_RESULT.json',dict(exit_code=code,finished=time.time()))
    raise SystemExit(code)

if __name__=='__main__':main()
