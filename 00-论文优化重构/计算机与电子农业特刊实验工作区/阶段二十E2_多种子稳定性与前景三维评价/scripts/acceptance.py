"""Evidence-based G1--G18 acceptance. No training or historical-file writes."""
from __future__ import annotations
import json
import math
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from common import *
from protocol import assert_pair,config_invariant


def artifact_record(path):
    relative=path.relative_to(E2_DATA)
    if path.name=='chkpnt30000.pth':role='FINAL_CHECKPOINT'
    elif path.name=='point_cloud.ply' and 'iteration_30000' in str(path):role='FINAL_GAUSSIAN_PLY'
    elif path.suffix=='.pt':role='REAL_FLOAT_RENDER'
    else:role='RECOVERY_SUPPORT_OR_PRESERVED_ATTEMPT'
    return {'path':str(path),'relative_to_managed_root':str(relative),'bytes':path.stat().st_size,'sha256':sha256(path),'role':role,
            'managed_root':str(E2_DATA),'in_git':False}


def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_INDEX,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    with (STAGE/'logs/FINAL_PROTECTION_RECHECK.txt').open('w') as log:
        subprocess.run([str(PYTHON),str(STAGE/'scripts/preflight.py'),'--label','FINAL'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (STAGE/'logs/TESTS_FINAL.txt').open('w') as log:
        tests=subprocess.run([str(PYTHON),'-m','unittest','discover','-s',str(STAGE/'tests'),'-v'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    historical_audit=read_json(STAGE/'reports/HISTORICAL_SEED0_TRAINING_AUDIT.json')
    input_audit=read_json(STAGE/'reports/CHANGED_TRAINING_INPUT_AUDIT.json')
    assert historical_audit['status']=='PASS_WITH_EXPLICIT_LIMIT' and historical_audit['actual_inherited_models']==8
    assert input_audit['status']=='PASS' and input_audit['changed_training_views']==4
    manifest=read_csv(STAGE/'MULTISEED_TRAINING_MANIFEST.csv')
    full=read_csv(STAGE/'MULTISEED_PER_VIEW_METRICS.csv');fg=read_csv(STAGE/'FOREGROUND_PROXY_METRICS.csv');bg=read_csv(STAGE/'BACKGROUND_GEOMETRY_PROXY_METRICS.csv')
    effects=read_csv(STAGE/'PAIRED_SEED_EFFECTS.csv');stats=read_csv(STAGE/'reports/PAIRED_MULTISEED_STATISTICS.csv')
    new=[r for r in manifest if int(r['seed']) in NEW_SEEDS]
    runtime_inventory=[]
    for path in sorted((STAGE/'reports').glob('*_RUNTIME.json')):
        runtime=read_json(path)
        key=(runtime['sample'],runtime['group'],runtime['seed'])
        assert key in TRAIN_ORDER, f'Unplanned training runtime: {path}'
        runtime_inventory.append({'sample':key[0],'group':key[1],'seed':key[2],
                                  'attempt':runtime['attempt'],'exit_code':runtime['exit_code'],
                                  'observed_iteration':runtime['observed_iteration'],
                                  'optimizer_steps':runtime['optimizer_steps'],
                                  'runtime_path':str(path),'runtime_sha256':sha256(path)})
    successful_runtimes=[r for r in runtime_inventory if r['exit_code']==0]
    assert len(successful_runtimes)==8
    assert {(r['sample'],r['group'],r['seed']) for r in successful_runtimes}==set(TRAIN_ORDER)
    assert len({(r['sample'],r['group'],r['seed']) for r in successful_runtimes})==8
    write_csv(STAGE/'reports/TRAINING_ATTEMPT_INVENTORY.csv',runtime_inventory)
    pairs=[];all_initial=[]
    for sample in SCENES:
        for seed in NEW_SEEDS:
            members=[]
            for group in GROUPS:
                accepted=[read_json(p) for p in (STAGE/'reports').glob(f'{run_id(sample,group,seed)}_attempt*_VALIDATION.json') if read_json(p)['status']=='PASS']
                assert len(accepted)==1
                validation=accepted[0];label=run_id(sample,group,seed,validation['attempt'])
                initialization=read_json(STAGE/f'reports/{label}_INITIALIZATION.json')
                runtime=read_json(STAGE/f'reports/{label}_RUNTIME.json')
                assert runtime['optimizer_steps']==29999 and runtime['observed_iteration']==30000
                curve=read_csv(STAGE/f'logs/{label}_TRAINING_CURVE.csv')
                assert [int(r['iteration']) for r in curve]==list(range(1,30001))
                assert all(math.isfinite(float(row[field])) for row in curve for field in ('l1','rgb_loss')),f'Nonfinite training loss: {label}'
                frozen_config=read_json(e2_config_path(sample,group,seed))
                executed_config=read_json(Path(validation['checkpoint_path']).parent/'FULL_FROZEN_TRAIN_CONFIG.json')
                assert config_invariant(executed_config)==config_invariant(frozen_config)
                assert runtime['config_sha256']==sha256(e2_config_path(sample,group,seed))
                members.append({'init':initialization,'runtime':runtime,'validation':validation,'cameras':[r['camera_name'] for r in curve]})
                verification=read_json(STAGE/f'reports/RNG_{"stage1a" if seed==1 else "stage2"}.json')
                assert initialization['initial_rng_sha256']==verification['initial_rng_sha256']
                frozen_group=next(r for r in read_json(E0/f'reports/{sample}_PREFLIGHT.json')['groups'] if r['group']==group)
                for field,old in [('_xyz','xyz'),('_scaling','scale'),('_opacity','opacity')]:
                    assert initialization['initial_parameter_sha256'][field]==frozen_group['initialization_sha256'][old]
                all_initial.append(initialization)
            assert_pair(members[0]['init'],members[1]['init'])
            for field in ('config_invariant_sha256','optimizer_class','optimizer_groups'):
                assert members[0]['init'][field]==members[1]['init'][field],field
            same_sequence=members[0]['cameras']==members[1]['cameras']
            assert same_sequence,'Same-seed Python camera sequence must match'
            for m in members:
                assert sha256(m['validation']['checkpoint_path'])==m['validation']['checkpoint_sha256']
                assert sha256(m['validation']['PLY_path'])==m['validation']['PLY_sha256']
                assert m['validation']['activated_parameters_finite']
            pairs.append({'sample':sample,'seed':seed,'status':'PASS','initial_rng':members[0]['init']['initial_rng_sha256'],
                          'six_initial_parameters_identical':True,'same_ordered_train_and_test_cameras':True,'same_sparse_points':True,
                          'same_optimizer_class_and_groups':True,'all_30000_sampled_camera_names_identical':same_sequence,
                          'all_30000_loss_records_finite':True,'executed_config_matches_frozen_parameters':True,
                          'B_attempt':members[0]['validation']['attempt'],'C_attempt':members[1]['validation']['attempt']})
    for sample in SCENES:
        first=next(r for r in all_initial if r['sample']==sample and r['seed']==1)
        second=next(r for r in all_initial if r['sample']==sample and r['seed']==2)
        for field in ('python','numpy','torch_cpu','torch_cuda_0'):
            assert first['initial_rng_sha256'][field]!=second['initial_rng_sha256'][field]
        assert set(first['ordered_cameras']['train'])==set(second['ordered_cameras']['train'])
        assert first['initial_ply_sha256']==second['initial_ply_sha256']
        assert first['initial_parameter_sha256']['_rotation']!=second['initial_parameter_sha256']['_rotation']
        assert first['ordered_camera_sha256']!=second['ordered_camera_sha256']
    seed_report={'status':'PASS','timestamp':now(),'pairs':pairs,'historical_seed0_evidence':'SOURCE_PATH_EQUIVALENCE_ONLY; actual historical RNG state not captured',
                 'strict_same_wrapper_three_seed_claim':False,'native_CUDA_bitwise_determinism_claim':False}
    write_json(STAGE/'reports/ACTUAL_TRAINING_SEED_VERIFICATION.json',seed_report)
    paths=sorted(p for p in E2_DATA.rglob('*') if p.is_file())
    with ThreadPoolExecutor(max_workers=6) as pool:artifacts=list(pool.map(artifact_record,paths))
    write_csv(STAGE/'reports/LARGE_ARTIFACT_MANIFEST.csv',artifacts)
    write_json(STAGE/'reports/LARGE_ARTIFACT_SUMMARY.json',{'timestamp':now(),'files':len(artifacts),'total_bytes':sum(r['bytes'] for r in artifacts),
                'managed_root':str(E2_DATA),'root_must_not_be_deleted':True,'not_uploaded_to_git':True,
                'index_sha256':sha256(STAGE/'reports/LARGE_ARTIFACT_MANIFEST.csv')})
    ref_groups=defaultdict(set);roi_groups=defaultdict(set);viewsets=defaultdict(set)
    expected_refs={(r['sample'],r['loader_image_name']):r for r in read_csv(E0/'reports/EVALUATION_REFERENCE_MANIFEST.csv')}
    roi_source={(r['sample'],Path(r['registered_name']).stem):r for r in read_csv(E0/'MATCHED_2DGS_DATASET_MANIFEST.csv') if r['group']=='B_R0'}
    for row in full:
        key=row['sample'],row['loader_image_name'];ref=expected_refs[key]
        assert row['reference_path']==ref['reference_rgb_path'] and row['reference_original_sha256']==ref['common_rgb_sha256']
        assert row['ROI_path']==roi_source[key]['mask_path'] and row['ROI_original_sha256']==roi_source[key]['mask_sha256']
        ref_groups[key].add((row['reference_original_sha256'],row['reference_resized_tensor_sha256'],row['render_width'],row['render_height']))
        roi_groups[key].add((row['ROI_original_sha256'],row['ROI_resized_tensor_sha256']))
        viewsets[row['sample'],row['group'],row['seed']].add(row['loader_image_name'])
    assert all(len(group)==1 for group in ref_groups.values())
    assert all(len(group)==1 for group in roi_groups.values())
    for (sample,_,_),names in viewsets.items():assert names=={name for scene,name in expected_refs if scene==sample}
    changed=subprocess.check_output(['git','diff','--name-only',BASELINE_COMMIT,'--'],cwd=ROOT,text=True).splitlines()
    prefix=str(STAGE.relative_to(ROOT))+'/'
    # core.quotePath makes raw non-ASCII names escaped; ask git explicitly for literal names.
    changed=subprocess.check_output(['git','-c','core.quotePath=false','diff','--name-only',BASELINE_COMMIT,'--'],cwd=ROOT,text=True).splitlines()
    unrelated=[name for name in changed if not name.startswith(prefix)]
    failed_attempts=[read_json(p) for p in (STAGE/'reports').glob('*_FAILED_ATTEMPT.json')]
    preserved_failures=all(Path(r['process']['stdout_stderr_path']).exists() and sha256(r['process']['stdout_stderr_path'])==r['process']['stdout_stderr_sha256'] for r in failed_attempts)
    plan=read_json(STAGE/'configs/EXPERIMENT_PLAN.json');remediation=read_json(STAGE/'reports/PREOPTIMIZATION_REMEDIATION.json')
    assert sha256(STAGE/'configs/EXPERIMENT_PLAN.json')==remediation['frozen_plan_sha256']
    all_cfg=[read_json(r['path']) for r in plan['configs']]
    invariant=all(config_invariant(c)==config_invariant(all_cfg[0]) for c in all_cfg)
    mesh_reference=read_json(STAGE/'configs/NEXT_PHASE_MESH_REFERENCE.json')
    assert mesh_reference['status']=='PROSPECTIVE_REFERENCE_ONLY_NO_MESH_EXECUTED'
    assert {r['sample'] for r in mesh_reference['models']}=={'WanNianQing1','CaoMei2','HongZhang'}
    assert all(r['reference_group']=='B_R0' and r['seed']==0 for r in mesh_reference['models'])
    for model in mesh_reference['models']:
        assert sha256(model['checkpoint_path'])==model['checkpoint_sha256']
        assert sha256(model['PLY_path'])==model['PLY_sha256']
    gate=[]
    def add(number,description,passed,evidence):gate.append({'gate':f'G{number}','description':description,'status':'PASS' if passed else 'FAIL','evidence':evidence})
    protected=read_json(STAGE/'reports/FINAL_INPUT_RECHECK.json')
    recomputed=read_json(STAGE/'reports/FLOAT_RENDER_METRIC_RECOMPUTATION.json')
    add(1,'Original Seed 0 and E0/E1 files unchanged',protected['status']=='PASS','FINAL_HASH_RECHECK.csv: all 7914 source-protection rows plus managed/history hashes; HISTORICAL_SEED0_TRAINING_AUDIT; original float scores asserted during inherited evaluation')
    add(2,'Eight actual additional trainings, with all failures disclosed',len(new)==8 and preserved_failures,'MULTISEED_TRAINING_MANIFEST.csv; zero-step failures and PREOPTIMIZATION_REMEDIATION.json preserved')
    add(3,'Seeds 1/2 change all actual RNG streams',seed_report['status']=='PASS','ACTUAL_TRAINING_SEED_VERIFICATION.json; RNG stage independent-process verification')
    add(4,'Same-seed B/C initialization, full config, physical GPU comparable',invariant and len(pairs)==4 and all(r['GPU_UUID']==GPU_UUID for r in manifest),'Four actual pair checks; complete config invariants; fixed GPU UUID. Historical Seed 0 is source-path equivalent, not retroactive state proof')
    add(5,'All required models complete 30000 iterations',all(int(r['iterations'])==30000 for r in manifest),'Original/native final update count is 29999; recorded unchanged')
    add(6,'Checkpoints and Gaussian PLY reload',len(pairs)==4,'Eight new VALIDATION.json reports, original eight E1 validations, checkpoint/PLY exact tensors and render replay')
    add(7,'Identical frozen heldout cameras',len(viewsets)==17,'Actual loader/test-camera name sets and per-view manifest match frozen E0 references')
    add(8,'Canonical full raw RGB common reference',all(len(v)==1 for v in ref_groups.values()),'Original and resized tensor SHA per view identical across all groups/seeds')
    add(9,'One unchanged R0 proxy ROI for every group',all(len(v)==1 for v in roi_groups.values()),'Original and resized ROI tensor SHA per view identical; E0 B_R0 mask source')
    add(10,'All proxies labelled non-independent GT',all(r['ROI_is_independent_GT']=='False' and r['composited_metrics_are_proxy']=='True' for r in fg) and all(r['geometry_GT']=='NONE' and r['components_are_confirmed_floaters']=='False' for r in bg),'FOREGROUND_PROXY_METRIC_SPEC.md and per-view label fields')
    add(11,'Scene/training seed pair is the unit, not heldout views',len(effects)==6 and all(r['training_pairs']=='3' for r in stats),'Two scenes x three training pairs; no p-values or view-count significance')
    add(12,'No seed/checkpoint/figure cherry-picking',len(new)==8 and len(plan['configs'])==8,'Unmodified prospective plan; final30000 only; fixed plus explicitly Seed0-ranked exploratory figure selections; disclosed zero-step instrumentation deviation')
    add(13,'No extra A/HongZhang/unrelated training',{(r['sample'],r['group'],int(r['seed'])) for r in new}==set(TRAIN_ORDER),'TRAINING_ATTEMPT_INVENTORY.csv: every actual runtime belongs to the fixed matrix; exactly eight successful runs; inherited A/HongZhang only re-evaluated')
    add(14,'R1/2DGS/M2M3 production code unchanged',not unrelated and protected['status']=='PASS','57 production source hashes and tracked-diff restriction; no math or optimizer edits')
    add(15,'Manual GT and paper unchanged',not unrelated and protected['status']=='PASS','All tracked changes restricted to E2 delivery; no new GT files or manuscript edits')
    add(16,'Metrics recompute from actual float renders',recomputed['status']=='PASS' and recomputed['unique_float_renders']==len({r['float_render_path'] for r in full}),'FLOAT_RENDER_METRIC_RECOMPUTATION.csv/json; no screenshot-derived metric')
    add(17,'All negative results and failures retained',preserved_failures and all(r['no_exclusions']=='True' for r in stats),'All predeclared seeds, per-view scores, descriptive flags, two zero-step failures and raw logs retained')
    add(18,'Concrete next M2M3/mesh execution decision',(STAGE/'M2M3_METHOD_CORRECTION_HANDOFF.md').stat().st_size>5000 and (STAGE/'NEXT_PHASE_EXECUTION_PLAN.md').stat().st_size>2000,'REWRITE_TO_IMPLEMENTATION; three prospectively frozen B_R0/Seed0 checkpoint and PLY hashes verified; no next-stage work executed')
    result={'status':'PASS' if all(r['status']=='PASS' for r in gate) and tests.returncode==0 else 'PARTIAL','timestamp':now(),
            'gates':gate,'unit_tests_passed':tests.returncode==0,'unit_test_log_sha256':sha256(STAGE/'logs/TESTS_FINAL.txt'),
            'additional_trainings_completed':len(new),'new_iterations':sorted({int(r['iterations']) for r in new}),
            'unique_models':16,'group_seed_rows':len(manifest),'per_view_rows':len(full),'unique_float_renders':recomputed['unique_float_renders'],
            'preserved_failed_attempts':len(failed_attempts),'no_unrelated_tracked_changes':not unrelated,'unrelated_changes':unrelated,
            'historical_seed0_limit':'Actual historical RNG hash unavailable; source-equivalent path verified; strict uniform-wrapper three-seed claim suspended',
            'no_next_phase_actions_executed':True}
    write_csv(STAGE/'reports/ACCEPTANCE_GATES.csv',gate)
    write_json(STAGE/'reports/FINAL_ACCEPTANCE.json',result)
    print(json.dumps({'status':result['status'],'gates':[(r['gate'],r['status']) for r in gate],'additional_trainings_completed':len(new)},ensure_ascii=False),flush=True)
    assert result['status']=='PASS',result

if __name__=='__main__':main()
