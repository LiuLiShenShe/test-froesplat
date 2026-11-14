#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import argparse
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
import yaml
import numpy as np
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha,write_json
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.runner import run
from ForeSplat_AutoGaussianCleaner.cleaner.adaptive.memory_monitor import ProcessGpuMemory
from ForeSplat_AutoGaussianCleaner.execution_provenance import scene_input_fingerprint,native_renderer_fingerprint,completed_identity,invalidate_output,resolve_variant,gaussian_count,assert_output_separate


def validate_initial_reference(reference, spec, config, original_sha):
    old=json.loads(reference.read_text())
    if old.get('input_model_sha256',old.get('input_sha256'))!=original_sha:
        raise ValueError('Initial cache belongs to another GGGS input model')
    if old['config']['evaluation_max_side']!=config['evaluation_max_side']:
        raise ValueError('Initial cache resolution mismatch')
    for key in ['scene_input_fingerprint','native_renderer_fingerprint']:
        if not old['config'].get(key) or old['config'][key]!=config[key]:
            raise ValueError(f'Initial cache lacks current content provenance: {key}; recompute from the original GGGS input')
    for key in ['boundary_relative_width','evidence_views','initial_evidence_views','geometry']:
        if old['config'].get(key)!=config.get(key):
            raise ValueError(f'Initial cache evidence protocol mismatch: {key}')
    for name in ['cleaner/runtime.py','cleaner/contribution_analysis.py','cleaner/foreground_protection.py','cleaner/geometry_consistency.py']:
        if old['config'].get('implementation_sources',{}).get(name)!=config['implementation_sources'].get(name):
            raise ValueError(f'Initial cache derived-source mismatch: {name}')
    for key in ['dataset','gggs_repo','geometry_cameras','geometry_dir']:
        before,after=old['scene'].get(key),spec.get(key)
        if bool(before)!=bool(after) or (before and Path(before).resolve()!=Path(after).resolve()):
            raise ValueError(f'Initial cache uses another scene interface: {key}')
    return old


def main():
    p=argparse.ArgumentParser(description='ForeSplat Two-Stage Gaussian Artifact Suppression — native GGGS only')
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--strategy',choices=['historical_two_stage','adaptive'])
    p.add_argument('--method',choices=['C5','C5-A','C5-B','C5-C','C5-D'])
    p.add_argument('--stage2-input',type=Path,help='Optional native historical reference with original-row lineage; defaults to scene R5 baseline, otherwise Stage1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--baseline-cache',type=Path)
    p.add_argument('--evidence-cache',type=Path)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();base=Path(__file__).resolve().parent
    if a.strategy=='historical_two_stage' or (a.strategy is None and a.method is None):
        if a.method or a.baseline_cache or a.evidence_cache:
            p.error('Historical strategy uses its own current-model evidence and stage references')
        from ForeSplat_AutoGaussianCleaner.cleaner.historical.entry import execute
        execute(a,base)
        return
    a.method=a.method or 'C5'
    config=yaml.safe_load((base/'configs/default.yaml').read_text())
    config.update(yaml.safe_load((a.config or base/'configs/adaptive.yaml').read_text()))
    config['implementation_sources']={str(f.relative_to(base)):sha(f) for f in sorted(
        list((base/'cleaner').rglob('*.py'))+list((base/'evaluation').glob('*.py'))+[Path(__file__).resolve(),base/'execution_provenance.py'])}
    variant=resolve_variant(a.method,config)
    spec=json.loads(a.scene.read_text())
    assert_output_separate(a.output,spec)
    original_sha=sha(spec['model'])
    config['scene_input_fingerprint']=scene_input_fingerprint(spec)
    config['native_renderer_fingerprint']=native_renderer_fingerprint([spec['gggs_repo']])
    if a.baseline_cache:
        validate_initial_reference(a.baseline_cache.parents[1]/'original_reference.json',spec,config,original_sha)
    if a.evidence_cache:
        # Cache is accepted only with explicit same-input provenance and row identity.
        if a.evidence_cache.parent.name!='01' or a.evidence_cache.parent.parent.name!='rounds':
            raise ValueError('Only round-01 evidence from the untouched original input may be reused')
        ref=a.evidence_cache.parents[2]/'original_reference.json'
        validate_initial_reference(ref,spec,config,original_sha)
        with np.load(a.evidence_cache) as z:
            cached_ids=z['original_ids']
            if len(cached_ids)!=gaussian_count(spec['model']) or not np.array_equal(cached_ids,np.arange(len(cached_ids))):
                raise ValueError('Evidence cache must retain untouched original row identities')
    if (a.output/'COMPLETE.json').exists():
        old=json.loads((a.output/'COMPLETE.json').read_text())
        if a.resume and completed_identity(a.output,config,spec,variant):
            print('Already complete',a.output);return
        raise FileExistsError('Output already complete with another configuration; choose another path')
    if (a.output/'original_reference.json').exists():
        if not a.resume:
            raise FileExistsError('Incomplete output exists; use --resume to archive and reproducibly restart it')
        old=json.loads((a.output/'original_reference.json').read_text())
        if old['config']!=config or old['input_sha256']!=sha(spec['model']) or old['scene']!=spec or old.get('variant')!=variant:
            raise ValueError('Interrupted output belongs to another configuration/input')
        import time
        archive=a.output.with_name(a.output.name+f'.interrupted_{int(time.time())}')
        a.output.rename(archive)
        print('Archived interrupted output before restart:',archive,flush=True)
    if torch.cuda.device_count()!=1:raise RuntimeError('Expected physical GPU1 only')
    torch.set_num_threads(8)
    write_json(a.output/'execution.json',dict(argv=sys.argv,physical_gpu=1,torch_version=torch.__version__,variant=variant))
    import shutil
    for name in config['implementation_sources']:
        path=a.output/'source_snapshot'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(base/name,path)
    memory=ProcessGpuMemory(a.output).start()
    try:
        result=run(spec,config,a.output,variant,a.baseline_cache,a.evidence_cache)
    finally:
        memory_usage=memory.finish()
        write_json(a.output/'GPU_MEMORY_SUMMARY.json',memory_usage)
    result['gpu_memory']=memory_usage
    write_json(a.output/'cleanup_metrics.json',result)
    try:
        intact=(scene_input_fingerprint(spec)==config['scene_input_fingerprint'] and
            native_renderer_fingerprint([spec['gggs_repo']])==config['native_renderer_fingerprint'] and
            sha(spec['model'])==original_sha and
            all(sha(base/name)==digest for name,digest in config['implementation_sources'].items()))
    except (OSError,ValueError,KeyError):
        intact=False
    if not intact:
        invalidate_output(a.output,'Source, native renderer, original GGGS input or scene contents changed during execution')
        raise RuntimeError('Completed output invalidated because its frozen input/protocol changed during execution')
    print(json.dumps({k:result[k] for k in ['plant','outcome','removed_gaussians','artifact_reduction','seconds']}),flush=True)


if __name__=='__main__':main()
