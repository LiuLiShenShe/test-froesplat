"""Post-training ForeSplat pipeline node; invokes the existing GGGS cleaner."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import importlib.util


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scene_camera_names(scene: dict) -> list[str]:
    """Read the same native COLMAP inventory as Scene, without creating CUDA state."""
    loader=Path(scene['gggs_repo'])/'scene/colmap_loader.py'
    spec=importlib.util.spec_from_file_location('foresplat_result_camera_inventory',loader)
    if spec is None or spec.loader is None:raise ValueError('Missing native GGGS COLMAP camera reader')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sparse=Path(scene['dataset'])/'sparse/0'
    cameras=(module.read_extrinsics_binary(str(sparse/'images.bin')) if (sparse/'cameras.bin').exists()
             else module.read_extrinsics_text(str(sparse/'images.txt')))
    return [Path(v.name).stem for v in sorted(cameras.values(),key=lambda v:v.name)]


def read_cleaning_result(scene_config: str | Path, output: str | Path) -> dict:
    """Require certified, complete native output and preserve a truthful outcome."""
    scene_config, output = Path(scene_config).resolve(), Path(output).resolve()
    if (output/'INVALIDATED.json').exists():
        raise ValueError('Cleaning output has an explicit protocol invalidation')
    scene = json.loads(scene_config.read_text())
    complete = json.loads((output / 'COMPLETE.json').read_text())
    report = json.loads((output / 'cleanup_metrics.json').read_text())
    if report.get('method')=='historical_two_stage':
        return read_historical_result(scene,output,complete,report)
    required=['cleaned_gaussians.ply','cleaned_gaussians_supersplat.ply','removed_gaussians.ply',
              'removed_gaussians_supersplat.ply','original_gaussian_ids.npy','removed_original_gaussian_ids.npy','viewer_baking.json']
    hashes=complete.get('artifact_hashes',{})
    for name in required:
        if name not in hashes or not (output/name).is_file() or _sha(output/name)!=hashes[name]:
            raise ValueError(f'Completed cleaning artifact missing or hash mismatch: {name}')
    native = output / 'cleaned_gaussians.ply'
    viewer = output / 'cleaned_gaussians_supersplat.ply'
    if complete['scene'] != scene or complete['input_sha256'] != _sha(Path(scene['model'])):
        raise ValueError('Completed cleaning belongs to another input scene/model')
    if complete['native_sha256'] != _sha(native):
        raise ValueError('Completed native model hash mismatch')
    if not viewer.is_file():
        raise FileNotFoundError(viewer)
    if report['method'] == 'C5-A':
        raise ValueError('Appearance-only ablation is not a certified pipeline model')
    if report['method'] not in ['C5-B','C5-C','C5-D'] or complete.get('variant')!=report['method']:
        raise ValueError('Pipeline requires an actual quality-evaluated GGGS cleaner result')
    if report.get('config')!=complete.get('config') or report.get('input_sha256')!=complete['input_sha256']:
        raise ValueError('Cleaning report differs from its completed execution')
    initial,final=report.get('initial',{}),report.get('final',{})
    original_views=[v['name'] for v in initial.get('views',[])]
    if (not original_views or original_views!=scene_camera_names(scene)
            or [v['name'] for v in final.get('views',[])]!=original_views
            or not initial.get('metrics_complete',True) or not final.get('metrics_complete',True)):
        raise ValueError('Pipeline requires complete matched original-view quality evaluation')
    modified = report['removed_gaussians'] > 0 or report['refined_gaussians'] > 0
    improving = max(report['artifact_reduction'], report.get('free_artifact_reduction', 0)) > 0
    if modified and complete['native_sha256']==complete['input_sha256']:
        raise ValueError('Reported Gaussian modification exported an identical original native model')
    if modified and not improving:
        raise ValueError('Edited final model has no certified net background improvement')
    expected_outcome='measured_improvement_with_residuals' if modified and improving else 'no_accepted_modification'
    if report.get('outcome')!=expected_outcome:
        raise ValueError('Cleaning outcome does not match actual model modification and measured benefit')
    return dict(stage='Automatic Gaussian Cleaning', plant=scene['plant'],
        status='measured_improvement_with_residuals' if modified and improving else 'no_accepted_modification',
        modified=modified, method=report['method'], native_model=str(native),
        supersplat_model=str(viewer), removed_model=str(output / 'removed_gaussians.ply'),
        report=str(output / 'cleanup_metrics.json'), original_gaussian_ids=str(output / 'original_gaussian_ids.npy'),
        input_model=scene['model'], removed_gaussians=report['removed_gaussians'],
        refined_gaussians=report['refined_gaussians'], seconds=report['seconds'],
        training_background_improvement=report['artifact_reduction'],
        auxiliary_free_background_improvement=report.get('free_artifact_reduction', 0),
        downstream_gaussian_model=str(native), input_sha256=complete['input_sha256'],
        native_sha256=complete['native_sha256'])


def read_historical_result(scene,output,complete,report):
    """Require actual whole-view, frozen-input two-stage output before chaining."""
    if (not complete.get('source_and_input_verification') or not complete.get('full_view_evaluation')
            or complete.get('scene')!=scene or complete.get('config')!=report.get('config')
            or complete.get('input_sha256')!=_sha(Path(scene['model']))):
        raise ValueError('Historical cleaner lacks verified complete input/protocol identity')
    required=['cleaned_gaussians.ply','cleaned_gaussians_supersplat.ply','removed_gaussians.ply',
              'removed_gaussians_supersplat.ply','original_gaussian_ids.npy','removed_original_gaussian_ids.npy','viewer_baking.json']
    for name in required:
        if not (output/name).is_file() or complete.get('artifact_hashes',{}).get(name)!=_sha(output/name):
            raise ValueError(f'Historical cleaning artifact missing or changed: {name}')
    expected=scene_camera_names(scene)
    for stage in ['stage1','stage2']:
        s=report[stage]
        if not s.get('metrics_complete'):
            raise ValueError('Historical stage lacks complete final evaluation')
        for side in ['initial','final']:
            if [v['name'] for v in s[side]['views']]!=expected:
                raise ValueError('Historical stage has unmatched/incomplete observed cameras')
    return dict(stage='Automatic Gaussian Cleaning',plant=scene['plant'],method='historical_two_stage',
        status='measured_improvement_with_residuals' if report['stage2']['score']>0 else 'stage2_no_accepted_modification',
        native_model=str(output/'cleaned_gaussians.ply'),supersplat_model=str(output/'cleaned_gaussians_supersplat.ply'),
        downstream_gaussian_model=str(output/'cleaned_gaussians.ply'),
        original_gaussian_ids=str(output/'original_gaussian_ids.npy'),
        report=str(output/'cleanup_metrics.json'),seconds=report['seconds'],
        input_sha256=complete['input_sha256'],native_sha256=complete['native_sha256'])


def clean_trained_gggs(scene_config: str | Path, output: str | Path, *,
                       method: str | None = None, config: str | Path | None = None,
                       resume: bool = False) -> dict:
    """Call after GGGS training, before Gaussian export or the existing mesh stage."""
    module = Path(__file__).resolve().parent
    project = module.parent
    scene_config, output = Path(scene_config).resolve(), Path(output).resolve()
    scene = json.loads(scene_config.read_text())
    for key in ['model', 'dataset', 'gggs_repo']:
        if not Path(scene[key]).exists():
            raise FileNotFoundError(scene[key])
    uv = shutil.which('uv')
    if uv is None:
        raise RuntimeError('uv is required by the GGGS pipeline')
    command = [uv, 'run', '--no-sync', 'python', str(module / 'auto_clean_gaussians.py'),
               '--scene', str(scene_config), '--output', str(output)]
    command += ['--method',method] if method else ['--strategy','historical_two_stage']
    if config is not None:
        command += ['--config', str(Path(config).resolve())]
    if resume:
        command += ['--resume']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
    subprocess.run(command, cwd=project, env=env, check=True)
    result = read_cleaning_result(scene_config, output)
    (output / 'PIPELINE_STAGE.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result
