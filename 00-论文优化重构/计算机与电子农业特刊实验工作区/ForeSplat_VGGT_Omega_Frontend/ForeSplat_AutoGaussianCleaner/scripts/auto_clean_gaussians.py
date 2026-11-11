#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import yaml
from ForeSplat_AutoGaussianCleaner.cleaner.iterative_cleanup import run


def main():
    p = argparse.ArgumentParser(description='Automatic native GGGS artifact suppression; GPU1 only')
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/default.yaml')
    p.add_argument('--variant', choices=['A_once', 'A', 'B_once', 'B', 'C'], default='C')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    spec = json.loads(a.scene.read_text())
    config = yaml.safe_load(a.config.read_text())
    from ForeSplat_AutoGaussianCleaner.cleaner.io import sha, write_json
    module = Path(__file__).resolve().parents[1]
    config['implementation_sources'] = {str(x.relative_to(module)): sha(x) for x in sorted(
        list((module/'cleaner').glob('*.py'))+list((module/'evaluation').glob('*.py'))+list((module/'scripts').glob('*.py')))}
    if a.variant.startswith('A'):
        config.update(geometry=False, refinement=False)
    elif a.variant.startswith('B'):
        config['refinement'] = False
    if a.variant.endswith('_once'):
        config['max_rounds'] = 1
    if config['max_rounds'] < 1 or config['evaluation_max_side'] < 16:
        raise ValueError('Invalid round/resolution limit')
    complete = a.output/'COMPLETE.json'
    if complete.exists():
        old = json.loads(complete.read_text())
        if a.resume and old['config'] == config and old['scene'] == spec and old['input_sha256'] == sha(spec['model']):
            if sha(a.output/'cleaned_gaussians.ply') != old['native_sha256']:
                raise ValueError('Completed output hash mismatch')
            print('Already complete:', a.output)
            return
        raise FileExistsError('Output already completed; choose a new output or --resume with identical input/config')
    if (a.output/'original_reference.json').exists():
        # Incomplete runs are reproducibly restarted, retaining prior candidates under an archive.
        import time
        archive = a.output.with_name(a.output.name+f'.interrupted_{int(time.time())}')
        a.output.rename(archive)
    torch.set_num_threads(8)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected CUDA_VISIBLE_DEVICES=1')
    write_json(a.output/'execution.json', dict(argv=sys.argv, cuda_visible_devices='1', torch_version=torch.__version__, variant=a.variant))
    result = run(spec, config, a.output)
    print(json.dumps({k: result[k] for k in ['plant', 'outcome', 'artifact_reduction', 'cleaned_gaussians', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
