from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import *
from protocol import config_invariant


def check_hash(row):
    p = Path(row['path'])
    if not p.is_absolute():
        p = ROOT / p
    try:
        actual = sha256(p)
        return {**row, 'actual_sha256': actual, 'status': 'PASS' if actual == row['sha256'] else 'FAIL'}
    except Exception as exc:
        return {**row, 'actual_sha256': '', 'status': 'FAIL', 'error': repr(exc)}


def freeze():
    plan_path = STAGE / 'configs/EXPERIMENT_PLAN.json'
    assert not plan_path.exists(), 'Never redefine an already frozen protocol'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == BASELINE_COMMIT
    config_hashes = read_json(E0 / 'reports/FROZEN_TRAIN_CONFIG_HASHES.json')
    for path, digest in config_hashes.items():
        assert sha256(ROOT / path) == digest, path
    configs = []
    for scene, group, seed in TRAIN_ORDER:
        config = e0_config(scene, group)
        config['seed'] = seed
        path = output_path(scene, group, seed)
        for section in ('model', 'all_parameters'):
            config[section]['model_path'] = str(path)
        config['e2'] = {'source_frozen_config': str(E0 / f'reports/{scene}_{group}_TRAIN_CONFIG.json'),
                        'source_frozen_config_sha256': sha256(E0 / f'reports/{scene}_{group}_TRAIN_CONFIG.json'),
                        'physical_gpu_uuid': GPU_UUID, 'seed_entry': 'protocol.set_seed before native training()',
                        'historical_native_optimizer_updates': 29999}
        assert not path.exists()
        write_json(e2_config_path(scene, group, seed), config)
        configs.append({'sample':scene, 'group':group, 'seed':seed, 'path':str(e2_config_path(scene, group, seed)),
                        'sha256':sha256(e2_config_path(scene, group, seed))})
    invariants = [config_invariant(e0_config(scene, group)) for scene in SCENES for group in GROUPS]
    assert all(v == invariants[0] for v in invariants)
    write_json(plan_path, {
        'frozen_at':now(), 'baseline_commit':BASELINE_COMMIT, 'training_order':TRAIN_ORDER,
        'seeds':[0,1,2], 'seed0':'inherit unmodified E1 final checkpoints and float renders',
        'additional_trainings':8, 'GPU_UUID':GPU_UUID, 'configs':configs,
        'iterations':30000, 'checkpoint_policy':'Only native final iteration 30000; no best-checkpoint selection',
        'retry_policy':{'max_full_attempts':MAX_FULL_ATTEMPTS, 'trigger':'documented infrastructure interruption/OOM/NaN only',
                        'action':'preserve entire failed attempt; restart from iteration 0 with identical seed/config/GPU',
                        'no_quality_based_retry':True, 'no_incomplete_resume':True},
        'gate_after_first_pair':'Both WanNianQing1 seed1 models complete and reload before next six',
        'analysis_unit':'scene and independently trained seed-pair; heldout views are correlated',
        'paired_sd':'sample standard deviation, ddof=1', 'significance_tests':'none; n=3 is descriptive',
        'decision_rule':'ROBUST only if both full and ROI PSNR effects share direction across 0/1/2 and absolute paired mean exceeds both B and C cross-seed sample SD; otherwise MIXED if endpoint disagreement else INCONCLUSIVE',
        'directions':'Higher PSNR/SSIM, lower LPIPS/errors/background energy preferred; geometry distributions descriptive only',
        'outlier_rule':'Flag a seed only when its absolute distance from other-two mean exceeds 3 times other-two sample SD; descriptive, do not remove',
        'figures':'fixed by camera-name order first/middle/last BEFORE new metrics; seed0 ranked best/worst/closest-zero may be additional explicitly exploratory',
        'foreground_roi':'E0 B_R0 heldout masks; nearest resize; same for all models; no independent GT',
        'foreground_coverage':'mean(alpha>0.5) inside fixed ROI; missing = 1-coverage',
        'geometry':'native-scale finite alpha-supported depth quantiles and rendered-vs-surface-normal cosine proxy, no metric GT',
        'statistical_limit':'same seed is controlled initial RNG, not a guarantee of deterministic custom CUDA training',
        'no_extra_scenes_no_A_no_HongZhang_training':True,
    })
    fixed=[]
    for scene in SCENES:
        refs=[r for r in read_csv(E0/'reports/EVALUATION_REFERENCE_MANIFEST.csv') if r['sample']==scene]
        refs.sort(key=lambda r:r['loader_image_name'])
        for category,i in [('first',0),('middle',len(refs)//2),('last',len(refs)-1)]:
            fixed.append({'sample':scene,'loader_image_name':refs[i]['loader_image_name'],'selection':category,'policy':'PREDECLARED_CAMERA_NAME_ORDER'})
    write_csv(STAGE/'configs/FIXED_FIGURE_VIEWS.csv',fixed)
    protected=[]
    for folder in (E0,E1):
        for p in folder.rglob('*'):
            if p.is_file(): protected.append({'path':str(p),'sha256':sha256(p),'kind':'HISTORICAL_STAGE_FILE'})
    existing = STAGE/'scripts/common.py'
    protected.append({'path':str(existing),'sha256':sha256(existing),'kind':'PREEXISTING_UNTRACKED_E2_COMMON'})
    write_csv(STAGE/'reports/HISTORICAL_STAGE_SHA256.csv',protected)
    print('PROTOCOL_FROZEN',sha256(plan_path),flush=True)


def check(label):
    plan=read_json(STAGE/'configs/EXPERIMENT_PLAN.json')
    records=read_csv(E0/'reports/MANAGED_ARTIFACT_HASHES.csv')
    records += read_csv(E1/'reports/LARGE_ARTIFACT_MANIFEST.csv')
    protection=read_csv(E0/'reports/SOURCE_PROTECTION.csv')
    sources=[r for r in protection if r['kind']=='PRODUCTION_CODE']
    for r in protection: records.append({'path':str(ROOT/r['path']),'sha256':r['sha256'],'kind':r['kind']})
    records += read_csv(STAGE/'reports/HISTORICAL_STAGE_SHA256.csv')
    records += [{'path':r['path'],'sha256':r['sha256']} for r in plan['configs']]
    unique={r['path']:r for r in records}
    with ThreadPoolExecutor(max_workers=6) as pool:
        checked=list(pool.map(check_hash,unique.values()))
    bad=[r for r in checked if r['status']!='PASS']
    assert read_json(E1/'reports/FINAL_ACCEPTANCE.json')['status']=='PASS'
    import_paths()
    import torch
    versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','lpips','numpy','Pillow','plyfile','matplotlib','tensorboard')}
    previous=read_json(E1/'reports/ENVIRONMENT.json')['versions']
    assert all(versions[k]==previous[k] for k in versions)
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,driver_version','--format=csv'],text=True)
    assert GPU_UUID in hardware
    env={'timestamp':now(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
         'versions':versions,'torch_cuda':torch.version.cuda,'torch_cudnn':torch.backends.cudnn.version(),
         'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
         'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
         'hardware':hardware,'disk_free_bytes':shutil.disk_usage(E2_DATA.parent).free,
         'extensions':{},'search':'Bocha HTTP403 insufficient quota; Context7 official docs separately retrieved',
         'RTK':'/home/test/.codex/RTK.md absent'}
    legacy_runtime=read_json(E1/'reports/PRETRAINING_INPUT_RECHECK.json')
    assert torch.version.cuda==legacy_runtime['cuda']
    assert GPU_UUID==legacy_runtime['physical_gpu_uuid']
    legacy_extensions={r['path']:r['sha256'] for r in legacy_runtime['extensions']}
    for modname in ('diff_surfel_rasterization._C','simple_knn._C'):
        module=importlib.import_module(modname)
        digest=sha256(module.__file__)
        assert legacy_extensions[module.__file__]==digest,modname
        env['extensions'][modname]={'path':module.__file__,'sha256':digest,'matches_E1_binary':True}
    write_json(STAGE/f'reports/{label}_ENVIRONMENT.json',env)
    write_csv(STAGE/f'reports/{label}_HASH_RECHECK.csv',checked,list(dict.fromkeys(k for r in checked for k in r)))
    result={'timestamp':now(),'status':'PASS' if not bad else 'FAIL','unique_hashes_checked':len(checked),
            'E0_managed_rows':6831,'production_sources':len(sources),
            'source_protection_rows':len(protection),'failed':bad,
            'protocol_sha256':sha256(STAGE/'configs/EXPERIMENT_PLAN.json')}
    write_json(STAGE/f'reports/{label}_INPUT_RECHECK.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    assert not bad

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); parser.add_argument('--label',default='PRETRAINING')
    args=parser.parse_args()
    E2_DATA.mkdir(parents=True,exist_ok=True)
    if args.freeze: freeze()
    check(args.label)
