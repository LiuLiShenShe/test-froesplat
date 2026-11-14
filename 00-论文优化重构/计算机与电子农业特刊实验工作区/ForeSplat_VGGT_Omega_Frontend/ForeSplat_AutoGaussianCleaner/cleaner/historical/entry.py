"""Public CLI dispatch, frozen-source capture and reproducible resume checks."""
from pathlib import Path
import json
import shutil
import sys
import torch
import yaml
from ..io import sha, write_json
from .runner import run
from ...execution_provenance import assert_output_separate, scene_input_fingerprint, native_renderer_fingerprint


def execute(args, base):
    config = yaml.safe_load((args.config or base/'configs/historical_two_stage.yaml').read_text())
    if config.get('method_version') not in ['historical-two-stage-1','historical-two-stage-2','historical-two-stage-3','historical-two-stage-4','historical-two-stage-5','historical-two-stage-6','historical-two-stage-7','historical-two-stage-8']:
        raise ValueError('Historical strategy needs an explicitly versioned historical two-stage configuration')
    for key in ['evidence_views', 'confirmation_views', 'stage1_max_rounds', 'stage2_max_rounds']:
        if config[key] <= 0:
            raise ValueError(f'Positive {key} required')
    if not 16 <= config['evidence_views'] <= 24 or not 48 <= config['confirmation_views'] <= 64:
        raise ValueError('Two-stage strategy requires 16–24 quick and 48–64 confirmation cameras')
    spec = json.loads(args.scene.read_text())
    assert_output_separate(args.output, spec)
    sources = sorted(list((base/'cleaner').rglob('*.py'))+list((base/'evaluation').glob('*.py'))+
                     [base/'auto_clean_gaussians.py', base/'execution_provenance.py'])
    config['implementation_sources'] = {str(f.relative_to(base)): sha(f) for f in sources}
    config['scene_input_fingerprint'] = scene_input_fingerprint(spec)
    config['native_renderer_fingerprint'] = native_renderer_fingerprint([spec['gggs_repo']])
    baseline = args.stage2_input or spec.get('baseline_models', {}).get('R5')
    if baseline:
        baseline = Path(baseline).resolve()
        config['stage2_input_sha256'] = sha(baseline)
        config['stage2_row_ids_sha256'] = sha(baseline.parent/'original_gaussian_ids.npy')
    reference = dict(scene=spec, config=config, input_sha256=sha(spec['model']))
    output = args.output.resolve()
    marker = output/'COMPLETE.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if (args.resume and old.get('config') == config and old.get('input_sha256') == reference['input_sha256']
                and old.get('native_sha256') == sha(output/'cleaned_gaussians.ply')):
            print('Already complete:', output, flush=True)
            return
        raise FileExistsError('Existing completed two-stage output has different inputs, source or configuration')
    execution = output/'execution.json'
    if execution.exists():
        old = json.loads(execution.read_text())
        if not args.resume or old['reference'] != reference:
            raise FileExistsError('Incomplete output requires matching --resume or a new output directory')
        # Existing files never become fabricated completed artifacts. Keep the
        # entire failed attempt inspectable before a reproducible fresh run.
        import time
        archive = output.with_name(output.name+f'.interrupted_{time.time_ns()}')
        output.rename(archive)
        print('Archived interrupted attempt:', archive, flush=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected physical GPU1 only')
    torch.set_num_threads(8)
    write_json(execution, dict(argv=sys.argv, physical_gpu=1, reference=reference))
    for name in config['implementation_sources']:
        destination = output/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base/name, destination)
    try:
        result = run(spec, config, output, args.stage2_input)
        if (scene_input_fingerprint(spec) != config['scene_input_fingerprint'] or
                native_renderer_fingerprint([spec['gggs_repo']]) != config['native_renderer_fingerprint'] or
                any(sha(base/name) != digest for name, digest in config['implementation_sources'].items()) or
                (baseline and (sha(baseline) != config['stage2_input_sha256'] or
                              sha(baseline.parent/'original_gaussian_ids.npy') != config['stage2_row_ids_sha256']))):
            raise RuntimeError('Inputs, implementation or historical baseline changed during execution')
        # The outer entry certifies frozen sources after all exports and renders.
        complete = json.loads(marker.read_text())
        complete['source_and_input_verification'] = True
        write_json(marker, complete)
    except BaseException as error:
        marker.unlink(missing_ok=True)
        write_json(output/'FAILED.json', dict(error=type(error).__name__, message=str(error)))
        raise
    print(json.dumps(dict(plant=result['plant'], strategy='historical_two_stage',
        stage1_gaussians=result['stage1']['final_gaussians'], stage2_gaussians=result['stage2']['final_gaussians'],
        seconds=result['seconds'], final=result['models']['final']), ensure_ascii=False), flush=True)
