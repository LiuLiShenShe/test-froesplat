#!/usr/bin/env python3
"""Archive the exact cleaner source, native GGGS source and observed environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation-root',type=Path,required=True)
    parser.add_argument('--support-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--native-gggs-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--decision',type=Path,help='Required to label a snapshot as production validated')
    args=parser.parse_args();implementation=args.implementation_root.resolve();support=args.support_root.resolve()
    native=args.native_gggs_root.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Source deliveries are immutable; choose a new archive path')
    if not all((implementation/name).is_file() for name in ['auto_clean_gaussians.py','auto_clean_batch.py','configs/default.yaml']):
        raise ValueError('Incomplete existing cleaner implementation')
    if not (native/'scene/gaussian_model.py').is_file():raise ValueError('Actual native GGGS source required')
    decision=json.loads(args.decision.read_text()) if args.decision else None
    files={}
    allowed={'.py','.yaml','.yml','.json','.md','.txt','.toml','.cu','.cuh','.cpp','.c','.h','.hpp'}
    excluded={'experiments','__pycache__','.git','.venv','build','dist'}
    def source_files(base):
        for directory,folders,names in os.walk(base):
            folders[:]=sorted(name for name in folders if name not in excluded)
            for name in sorted(names):yield Path(directory)/name
    def add(path,name):
        # No credentials or environment/index settings belong in source delivery.
        if path.name.startswith('.env') or path.name in ['credentials.json','auth.json']:return
        files[name]=path
    for base in [support,implementation]:
        for path in source_files(base):
            relative=path.relative_to(base)
            if not path.is_file() or set(relative.parts)&excluded:continue
            if base==implementation and base!=support and relative.parts[0] not in ['cleaner','evaluation','configs'] and str(relative) not in ['auto_clean_gaussians.py','auto_clean_batch.py','execution_provenance.py']:
                continue
            if path.suffix in allowed or path.name.startswith('LICENSE'):
                add(path,'ForeSplat_AutoGaussianCleaner/'+str(relative))
    # The source used for every native render is retained with its license.
    for path in source_files(native):
        relative=path.relative_to(native)
        if not path.is_file() or set(relative.parts)&excluded:continue
        if path.suffix in allowed or path.name.startswith('LICENSE') or path.name=='CMakeLists.txt':
            add(path,'native_gggs/'+str(relative))
    project=support.parent
    for name in ['pyproject.toml','uv.lock','LICENSE','tools/clean_trained_gggs.py']:
        path=project/name
        if path.is_file():add(path,'project/'+name)
    observed=support/'experiments/GGGS_Cleaner_Comparison/comparisons'
    for name in ['UV_ENVIRONMENT_SNAPSHOT.json','observed_python_requirements.txt']:
        path=observed/name
        if path.is_file():add(path,'environment/'+name)
    native_observers=project/'experiments/KongQueZhuYu/geometry_rgb_dual_frontend/scripts'
    for name in ['train_observed.py','train_observed_resume.py']:
        path=native_observers/name
        if path.is_file():add(path,'training_observers/'+name)
    if decision:
        expected=decision.get('development_protocol',{}).get('sources',{})
        if not expected:raise ValueError('A recorded full development decision is required')
        for name,value in expected.items():
            path=implementation/name
            if not path.is_file() or digest(path)!=value:raise ValueError('Selected source differs from development decision: '+name)
        add(args.decision.resolve(),'validation/DEVELOPMENT_DECISION.json')
    source_hashes={name:digest(path) for name,path in files.items()}
    manifest=dict(label=args.label,source_root=str(implementation),native_source_root=str(native),files=source_hashes,
        scope='Exact source archive of the existing GGGS cleaner and actual native GGGS backend; no models, datasets, frontend inference, other reconstruction backends or virtual environments bundled',
        production_validation='Recorded development decision attached; independent-test result must be assessed separately' if decision else 'Development source snapshot only; no production/generalization success claimed',
        environment='Observed imported dependencies and native compilation sources retained. No uv.lock exists unless explicitly bundled; observed requirements are not a portable dependency lock.',
        entrypoints=['ForeSplat_AutoGaussianCleaner/auto_clean_gaussians.py','ForeSplat_AutoGaussianCleaner/auto_clean_batch.py'])
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix(output.suffix+'.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in files.items():archive.write(path,name)
        archive.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        archive.writestr('HOW_TO_RUN.md','''# GGGS cleaner source snapshot

Use the existing ForeSplat uv environment on GPU1. Native Gaussian input must preserve XYZ, all three log-scales, quaternion, opacity logit, full SH and filter_3D.
Provide a scene JSON with the actual trained GGGS model, corrected native renderer, RGB/masks/COLMAP cameras and existing optional depth/confidence. Adapt absolute scene paths for your workspace.

Run `CUDA_VISIBLE_DEVICES=1 uv run --no-sync python auto_clean_gaussians.py --scene <scene.json> --method C5 --config <selected_config.yaml> --output <new_directory>` from the cleaner directory.
Batch uses `auto_clean_batch.py --manifest <manifest.json> --method C5 --output-root <new_directory>`. `--jobs 2` optionally overlaps two isolated scenes on GPU1; shared-workstation timing is recorded.

Native GGGS rendering uses the local CUDA extension built for the observed PyTorch/Python. Its exact source and license are included under native_gggs; models/data and compiled binaries are supplied by the existing project environment.
Use `cleaned_gaussians_supersplat.ply` for standard viewer import and `cleaned_gaussians.ply` for the native GGGS/downstream pipeline. A zero-modification report is not cleanup success. See SOURCE_MANIFEST.json for snapshot/validation scope.
''')
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:raise ValueError('Source archive integrity check failed')
    # Verify no source changed during archive construction.
    if any(digest(path)!=source_hashes[name] for name,path in files.items()):raise ValueError('Source changed while packaging')
    temporary.replace(output)
    sidecar=dict(archive=str(output),sha256=digest(output),bytes=output.stat().st_size,source_files=len(files),label=args.label,source_manifest=manifest)
    output.with_suffix('.json').write_text(json.dumps(sidecar,ensure_ascii=False,indent=2)+'\n')
    print(output,'files',len(files),'bytes',output.stat().st_size)


if __name__=='__main__':main()
