"""Archive actual E1 provenance without claiming historical RNG-state capture."""
from __future__ import annotations
import ast
from common import *
from protocol import config_invariant


def main():
    observer=E1/'scripts/train_observed.py'
    launcher=E0/'scripts/run_phase20e1.py'
    native=GS/'utils/general_utils.py'
    tree=ast.parse(launcher.read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
    seed_calls=[node for node in calls if node.func.id=='safe_state']
    training_calls=[node for node in calls if node.func.id=='training']
    assert len(seed_calls)==len(training_calls)==1 and seed_calls[0].lineno<training_calls[0].lineno
    assert isinstance(seed_calls[0].args[0],ast.Constant) and seed_calls[0].args[0].value is False
    observed=ast.parse(observer.read_text())
    paths=[node for node in ast.walk(observed) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run_path']
    assert len(paths)==1
    rows=[]
    for old in read_csv(E1/'TRAINING_RUN_MANIFEST.csv'):
        sample,group=old['sample'],old['group'];label=f'{sample}_{group}'
        runtime=read_json(E1/f'reports/{label}_RUNTIME.json')
        process=read_json(E1/f'reports/{label}_PROCESS.json')
        validation=read_json(E1/f'reports/{label}_VALIDATION.json')
        config_path=E0/f'reports/{label}_TRAIN_CONFIG.json'
        actual=Path(validation['checkpoint_path']).parent/'FULL_FROZEN_TRAIN_CONFIG.json'
        assert runtime['observer_source_sha256']==sha256(observer)
        assert runtime['config_sha256']==sha256(config_path)
        assert runtime['seed']==0 and runtime['physical_gpu']==1
        assert runtime['observed_iteration']==validation['iteration']==30000
        assert runtime['exit_code']==process['exit_code']==0 and validation['status']=='PASS'
        assert config_invariant(read_json(actual))==config_invariant(e0_config(sample,group))
        curve=E1/f'metrics/{label}_TRAINING_CURVE.csv'
        assert sha256(curve)==runtime['curve_sha256']
        assert [int(r['iteration']) for r in read_csv(curve)]==list(range(1,30001))
        rows.append({'sample':sample,'group':group,'seed':0,'iterations':30000,'native_optimizer_updates':29999,
            'start_time':runtime['start_time'],'end_time':runtime['end_time'],'GPU_UUID':GPU_UUID,
            'actual_command_argv_json':json.dumps(process['observer_command'],ensure_ascii=False),
            'frozen_config_path':str(config_path),'frozen_config_sha256':sha256(config_path),
            'executed_config_path':str(actual),'executed_config_sha256':sha256(actual),
            'runtime_path':str(E1/f'reports/{label}_RUNTIME.json'),'runtime_sha256':sha256(E1/f'reports/{label}_RUNTIME.json'),
            'stdout_stderr_path':process['stdout_stderr_path'],'stdout_stderr_sha256':process['stdout_stderr_sha256'],
            'checkpoint_path':validation['checkpoint_path'],'checkpoint_sha256':validation['checkpoint_sha256'],
            'PLY_path':validation['final_PLY_path'],'PLY_sha256':validation['final_PLY_sha256'],
            'curve_path':str(curve),'curve_sha256':runtime['curve_sha256'],'initial_sparse_PLY_sha256':runtime['initial_ply_sha256'],
            'historical_initial_RNG_sha256':'NOT_CAPTURED','comparison_scope':'SOURCE_EQUIVALENT_NATIVE_SEED_0_PATH; NOT_RETROACTIVE_STATE_PROOF'})
    assert len(rows)==8
    write_csv(STAGE/'reports/HISTORICAL_SEED0_TRAINING_AUDIT.csv',rows)
    write_json(STAGE/'reports/HISTORICAL_SEED0_TRAINING_AUDIT.json',{'status':'PASS_WITH_EXPLICIT_LIMIT','timestamp':now(),
        'actual_inherited_models':8,'native_seed':0,'source_chain':[{ 'path':str(p),'sha256':sha256(p)} for p in (observer,launcher,native)],
        'seed_call_before_training_line':seed_calls[0].lineno,'native_training_call_line':training_calls[0].lineno,
        'same_native_seed0_path_reexecution':'SEED_CONTROL_VERIFICATION.json',
        'actual_historical_initial_RNG_hash_available':False,'strict_uniform_wrapper_three_seed_inference_suspended':True,
        'current_uniform_wrapper_analysis_seeds':[1,2],'historical_models_modified_or_retrained':False})
    print('HISTORICAL_SEED0_AUDIT_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
