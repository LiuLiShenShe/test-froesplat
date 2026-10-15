from __future__ import annotations
import argparse
import csv
import json
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path
from common import *
from protocol import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sample',required=True); parser.add_argument('--group',choices=GROUPS,required=True)
    parser.add_argument('--seed',type=int,choices=NEW_SEEDS,required=True); parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    assert (args.sample,args.group,args.seed) in MATRIX
    assert read_json(STAGE/'reports/PRETRAINING_INPUT_RECHECK.json')['status']=='PASS'
    assert read_json(STAGE/'reports/SEED_CONTROL_VERIFICATION.json')['status']=='PASS'
    assert_gpu()
    frozen=read_json(STAGE/'configs/EXPERIMENT_PLAN.json')
    selected=next(r for r in frozen['configs'] if (r['sample'],r['group'],r['seed'])==(args.sample,args.group,args.seed))
    assert sha256(selected['path'])==selected['sha256']
    config=read_json(selected['path'])
    output=output_path(args.sample,args.group,args.seed,args.attempt)
    assert not output.exists(), 'Preserve successful and failed attempts'
    for section in ('model','all_parameters'): config[section]['model_path']=str(output)
    import_paths()
    import numpy as np
    import torch
    import train
    from plyfile import PlyData
    logger_compatibility()
    label=run_id(args.sample,args.group,args.seed,args.attempt)
    runtime={'sample':args.sample,'group':args.group,'seed':args.seed,'attempt':args.attempt,'start_time':now(),
             'config_sha256':selected['sha256'],'output_path':str(output),'iterations':30000,
             'GPU_UUID':GPU_UUID,'physical_gpu':GPU_INDEX,'native_final_optimizer_update':29999,
             'observer_source_sha256':sha256(Path(__file__)),'seed_source_sha256':sha256(STAGE/'scripts/protocol.py'),
             'optimizer_steps':0,'logging_compatibility':'Same E1 Agg.buffer_rgba RGB compatibility; no production edits'}
    curve_path=STAGE/f'logs/{label}_TRAINING_CURVE.csv'
    curve=curve_path.open('x',newline='')
    writer=csv.writer(curve)
    writer.writerow(['iteration','camera_name','l1','rgb_loss','gaussians','native_iteration_ms','wall_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'])
    started=time.perf_counter(); state={'iteration':0,'camera_stack':[]}
    original_scene=train.Scene; original_report=train.training_report; original_randint=train.randint
    original_setup=train.GaussianModel.training_setup
    initialization={}

    def observed_scene(dataset,gaussians,*positional,**keywords):
        scene=original_scene(dataset,gaussians,*positional,**keywords)
        initialization.update(verify_scene(scene,config,args.sample))
        state['scene']=scene
        assert initialization['initial_ply_sha256']==read_json(E1/f'reports/{args.sample}_{args.group}_RUNTIME.json')['initial_ply_sha256']
        return scene

    def observed_setup(model,opt):
        initialization['pre_optimizer_rng_sha256']=rng_hashes()
        value=original_setup(model,opt)
        initialization['post_optimizer_rng_sha256']=rng_hashes()
        initialization.update({'sample':args.sample,'group':args.group,'seed':args.seed,'attempt':args.attempt,
                               'config_invariant_sha256':hashlib.sha256(json.dumps(config_invariant(config),sort_keys=True).encode()).hexdigest(),
                               'optimizer_class':type(model.optimizer).__name__,
                               'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in model.optimizer.param_groups]})
        if args.group=='C_R1':
            evidence=sorted((STAGE/'reports').glob(f'{run_id(args.sample,"B_R0",args.seed)}_attempt*_INITIALIZATION.json'))
            assert evidence, 'B must have initialized first'
            assert_pair(read_json(evidence[-1]),initialization)
            initialization['paired_B_initialization_verified']=str(evidence[-1])
        write_json(STAGE/f'reports/{label}_INITIALIZATION.json',initialization)
        original_step=model.optimizer.step
        def observed_step(*pos,**kw):
            result=original_step(*pos,**kw)
            runtime['optimizer_steps']+=1
            return result
        model.optimizer.step=observed_step
        return value

    def observed_randint(low,high):
        index=original_randint(low,high)
        if not state['camera_stack']: state['camera_stack']=state['scene'].getTrainCameras().copy()
        state['camera_name']=state['camera_stack'].pop(index).image_name
        return index

    def observed_report(tb,iteration,l1,loss,loss_function,elapsed,tests,scene,renderer,render_args):
        state.update({'iteration':iteration,'scene':scene})
        writer.writerow([iteration,state['camera_name'],l1.item(),loss.item(),len(scene.gaussians.get_xyz),elapsed,
                         time.perf_counter()-started,torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()])
        if iteration%100==0: curve.flush()
        return original_report(tb,iteration,l1,loss,loss_function,elapsed,tests,scene,renderer,render_args)

    train.Scene=observed_scene; train.training_report=observed_report; train.randint=observed_randint
    train.GaussianModel.training_setup=observed_setup
    failure=None
    try:
        torch.cuda.reset_peak_memory_stats()
        initialization['initial_rng_sha256']=set_seed(args.seed)
        runtime['initial_rng_sha256']=initialization['initial_rng_sha256']
        # No RNG draws or reseeding between this line and native model/Scene initialization.
        train.training(Namespace(**config['model']),Namespace(**config['optimization']),Namespace(**config['pipeline']),
                       [7000,30000],[7000,30000],[],None)
        torch.cuda.synchronize()
        assert state['iteration']==30000 and runtime['optimizer_steps']==29999
        runtime['training_wall_seconds']=time.perf_counter()-started
        model=state['scene'].gaussians
        assert all(torch.isfinite(getattr(model,k)).all() for k in ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'))
        torch.save((model.capture(),30000),output/'chkpnt30000.pth')
        runtime['final_gaussians']=len(model.get_xyz)
        runtime['status']='TRAINING_COMPLETED_PENDING_RELOAD_VALIDATION'
        write_json(output/'FULL_FROZEN_TRAIN_CONFIG.json',config)
    except BaseException as exc:
        failure=exc; runtime.update({'status':'FAILED','exception':repr(exc)}); traceback.print_exc()
    finally:
        curve.close()
        runtime.update({'end_time':now(),'observed_iteration':state['iteration'],'process_wall_seconds':time.perf_counter()-started,
                        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                        'exit_code':int(failure is not None),'curve_path':str(curve_path),'curve_sha256':sha256(curve_path)})
        write_json(STAGE/f'reports/{label}_RUNTIME.json',runtime)
        print(json.dumps(runtime,ensure_ascii=False),flush=True)
    if failure is not None: raise failure

if __name__=='__main__': main()
