from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from common import *


def capture(seed, native, path):
    import_paths()
    import numpy as np
    import torch
    from protocol import set_seed, rng_hashes
    from utils.general_utils import safe_state
    if native:
        assert seed == 0
        safe_state(False)
        initial = rng_hashes()
    else:
        initial = set_seed(seed)
    names=[f'camera_{i:04d}' for i in range(213)]
    random.shuffle(names)
    sequence=[names.pop(random.randint(0,len(names)-1)) for _ in range(32)]
    cpu=torch.rand(32); cuda=torch.rand(32,device='cuda')
    result={'seed':seed,'native_legacy_path':native,'initial_rng_sha256':initial,'camera_draws':sequence,
            'numpy_draws':np.random.rand(32).tolist(),'cpu_draws':cpu.tolist(),'cuda_draws':cuda.cpu().tolist(),
            'post_draw_rng_sha256':rng_hashes(), 'torch_version':torch.__version__,
            'manual_seed_all_explicit_equivalence':None}
    # Test explicit CUDA seeding equivalence without affecting recorded initial state.
    torch.manual_seed(seed)
    before=[torch.cuda.get_rng_state(i).clone() for i in range(torch.cuda.device_count())]
    torch.cuda.manual_seed_all(seed)
    result['manual_seed_all_explicit_equivalence']=all(torch.equal(state,torch.cuda.get_rng_state(i)) for i,state in enumerate(before))
    write_json(path,result)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--capture',action='store_true'); parser.add_argument('--seed',type=int)
    parser.add_argument('--native',action='store_true'); parser.add_argument('--output'); args=parser.parse_args()
    if args.capture:
        capture(args.seed,args.native,Path(args.output)); return
    results={}
    specs=[('legacy0',0,True),('stage0',0,False),('stage1a',1,False),('stage1b',1,False),('stage2',2,False)]
    for label,seed,native in specs:
        path=STAGE/f'reports/RNG_{label}.json'
        command=[str(PYTHON),str(Path(__file__)), '--capture','--seed',str(seed),'--output',str(path)]
        if native: command.append('--native')
        subprocess.run(command,env=os.environ.copy(),cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        results[label]=read_json(path)
    compare_keys=('initial_rng_sha256','camera_draws','numpy_draws','cpu_draws','cuda_draws','post_draw_rng_sha256')
    checks={
        'legacy_seed0_state_and_draws_equivalent':all(results['legacy0'][k]==results['stage0'][k] for k in compare_keys),
        'same_seed_state_and_draws_reproducible':all(results['stage1a'][k]==results['stage1b'][k] for k in compare_keys),
        'seed1_seed2_all_rng_streams_differ':all(results['stage1a']['initial_rng_sha256'][k]!=results['stage2']['initial_rng_sha256'][k] for k in ('python','numpy','torch_cpu','torch_cuda_0')),
        'explicit_cuda_reseed_same_state':all(r['manual_seed_all_explicit_equivalence'] for r in results.values()),
        'frozen_complete_configuration_equal':True,
        'historical_runtime_seed0':all(read_json(E1/f'reports/{s}_{g}_RUNTIME.json')['seed']==0 for s in SCENES for g in GROUPS),
    }
    from protocol import config_invariant
    values=[config_invariant(e0_config(s,g)) for s in SCENES for g in GROUPS]
    checks['frozen_complete_configuration_equal']=all(v==values[0] for v in values)
    data={'timestamp':now(),'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
          'legacy_source_commit':E1_TRAIN_COMMIT,'legacy_entry':str(E0/'scripts/run_phase20e1.py'),
          'legacy_observer':str(E1/'scripts/train_observed.py'),
          'native_safe_state_sha256':sha256(GS/'utils/general_utils.py'),
          'seed0_rng_at_historical_run':'NOT_CAPTURED; independently re-executed source path equivalence, not retroactive state measurement',
          'comparability':'Same RNG-setting logic and unchanged packages/source; historical per-run RNG hashes unavailable. CUDA training is not guaranteed bitwise deterministic.',
          'runtime_pair_gate':'Full loaded camera order, points, initialized parameters and before/after optimizer RNG compared in each actual B/C pair before first optimization step.'}
    write_json(STAGE/'reports/SEED_CONTROL_VERIFICATION.json',data)
    assert all(checks.values()),checks
    print(json.dumps(data,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
