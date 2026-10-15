"""Observation only; all forward/backward, optimizer and pruning math are native."""
from common import *
from argparse import ArgumentParser,Namespace
import pickle,random,time,traceback

def rng_hashes():
    import torch,numpy as np
    values={'python':hashlib.sha256(pickle.dumps(random.getstate(),protocol=4)).hexdigest(),
            'numpy':hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=4)).hexdigest(),
            'torch_cpu':tensor_sha(torch.get_rng_state()),'torch_cuda':tensor_sha(torch.cuda.get_rng_state())}
    return values

def main():
    p=ArgumentParser();p.add_argument('--sample',choices=SCENES,required=True);p.add_argument('--method',choices=['M1','M2'],required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    gpu_idle();verify_frozen();import_paths(branch=True)
    import numpy as np,torch,train
    from plyfile import PlyData
    from PIL import Image
    from utils.general_utils import safe_state,PILtoTorch
    c=read_json(STAGE/f'configs/{a.sample}_{a.method}_TRAIN.json')
    output=DATA/'smoke'/a.sample/a.method if a.smoke else model_path(a.sample,a.method)
    assert not output.exists(),'Preserve every attempted run'
    c['model']['model_path']=str(output)
    if a.smoke:c['optimization']['iterations']=20
    opt=Namespace(**c['optimization'])
    label=f'{a.sample}_{a.method}'+('_SMOKE' if a.smoke else '')
    runtime={'sample':a.sample,'method':a.method,'smoke':a.smoke,'seed':0,'GPU_UUID':GPU_UUID,'start_time':now(),
             'command':sys.argv,'model_path':str(output),'config_sha256':sha(STAGE/f'configs/{a.sample}_{a.method}_TRAIN.json'),
             'production_train_sha256':sha(BRANCH/'train.py'),'wrapper_sha256':sha(__file__),'optimizer_steps':0}
    state={'iteration':0,'camera_stack':[],'extra':False,'score':{}}
    events=[];standard=[];camera_sequence=[]
    old_scene=train.Scene;old_setup=train.GaussianModel.training_setup;old_report=train.training_report
    old_randint=train.randint;old_score=train.m2m3_prune_scores;old_prune=train.maybe_prune_gaussians
    old_prune_points=train.GaussianModel.prune_points
    initialization={}
    started=time.perf_counter()
    curve=(STAGE/f'logs/{label}_CURVE.csv').open('x',newline='')
    writer=csv.writer(curve,lineterminator='\n');writer.writerow(['iteration','camera','l1','rgb_loss','Gaussian_count','wall_seconds'])
    def observed_scene(dataset,g,*args,**kwargs):
        scene=old_scene(dataset,g,*args,**kwargs);state['scene']=scene
        sparse=Path(dataset.source_path)/'sparse/0/points3D.ply'
        assert sha(output/'input.ply')==sha(sparse)==sha(Path(reference(a.sample)['checkpoint_path']).parent/'input.ply')
        split=rows(E0/'TRAIN_VALIDATION_SPLIT.csv')
        ordered={}
        for splitname,cameras in [('train',scene.getTrainCameras()),('validation',scene.getTestCameras())]:
            ordered[splitname]=[x.image_name for x in cameras]
            assert set(ordered[splitname])=={r['loader_image_name'] for r in split if r['sample']==a.sample and r['split']==splitname}
            assert all(x.gt_alpha_mask is not None for x in cameras)
        alpha=[]
        for cam in scene.getTrainCameras():
            with Image.open(Path(dataset.source_path)/dataset.images/f'{cam.image_name}.png') as image:
                expected=PILtoTorch(image.split()[3],(cam.image_width,cam.image_height))
            assert torch.equal(cam.gt_alpha_mask.cpu(),expected)
            alpha.append({'camera':cam.image_name,'alpha_sha256':tensor_sha(expected),'foreground_fraction':float((expected>.5).float().mean())})
        initialization.update(ordered_cameras=ordered,initial_ply_sha256=sha(sparse),initial_count=len(g.get_xyz),
                              parameter_hashes={f:tensor_sha(getattr(g,f)) for f in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']},
                              post_scene_rng=rng_hashes(),alpha=alpha)
        return scene
    def observed_setup(model,optimization):
        value=old_setup(model,optimization);initialization['post_optimizer_rng']=rng_hashes()
        if a.method=='M2':
            other_label=f'{a.sample}_M1'+('_SMOKE' if a.smoke else '')
            other=read_json(STAGE/f'reports/{other_label}_INITIALIZATION.json')
            assert initialization==other,'Same-seed M1/M2 initialization must be identical'
        write_json(STAGE/f'reports/{label}_INITIALIZATION.json',initialization)
        old_step=model.optimizer.step
        def step(*args,**kwargs):
            value=old_step(*args,**kwargs);runtime['optimizer_steps']+=1;return value
        model.optimizer.step=step
        return value
    def randint(low,high):
        index=old_randint(low,high)
        if not state['camera_stack']:state['camera_stack']=state['scene'].getTrainCameras().copy()
        camera_sequence.append(state['camera_stack'].pop(index).image_name)
        return index
    def report(tb,iteration,l1,loss,*rest):
        assert torch.isfinite(l1) and torch.isfinite(loss)
        state['iteration']=iteration
        writer.writerow([iteration,camera_sequence[-1],l1.item(),loss.item(),len(state['scene'].gaussians.get_xyz),time.perf_counter()-started])
        if iteration%100==0:curve.flush()
        return old_report(tb,iteration,l1,loss,*rest)
    def score(g,optimization,legacy,mask_scores=None):
        result=old_score(g,optimization,legacy,mask_scores)
        assert torch.isfinite(result).all()
        quantiles=torch.quantile(result,torch.tensor([0.,.25,.5,.75,.95,1.],device='cuda')).cpu().tolist()
        state['score']={f'score_{q}':v for q,v in zip(['min','p25','p50','p75','p95','max'],quantiles)}
        state['score']['score_mean']=float(result.mean())
        state['score']['mask_score_available']=mask_scores is not None
        if mask_scores is not None:
            assert torch.isfinite(mask_scores).all()
            state['score'].update(mask_score_min=float(mask_scores.min()),mask_score_mean=float(mask_scores.mean()),mask_score_max=float(mask_scores.max()))
        return result
    def prune_points(g,mask):
        before=len(g.get_xyz);value=old_prune_points(g,mask)
        if not state['extra']:
            standard.append({'iteration':state['iteration'],'before':before,'removed':int(mask.sum()),'after':len(g.get_xyz),'floor_applies':False})
        return value
    def prune(g,optimization,iteration,path,cameras=None,capacity_state=None):
        state['extra']=True;state['score']={}
        try:r=old_prune(g,optimization,iteration,path,cameras,capacity_state)
        finally:state['extra']=False
        if r is not None:
            event={'sample':a.sample,'method':a.method,'iteration':iteration,'Gaussian_count_before':r['gaussians_before'],
                   'candidate_count':r.get('m2m3_requested',0),'requested_removal':r.get('m2m3_requested',0),
                   'budget_limited_count':r.get('m2m3_after_budget',0),'budget_blocked_count':r.get('m2m3_budget_blocked',0),
                   'floor_count':r.get('floor_count',0),'floor_active':r.get('floor_active',False),
                   'floor_evaluated': 'floor_active' in r,
                   'floor_blocked_count':r.get('floor_blocked',0),'removed_count':r['removed'],'Gaussian_count_after':r['gaussians_after'],
                   'score_status':'EVALUATED' if state['score'] else 'NOT_EVALUATED_NO_CANDIDATES',**state['score']}
            assert event['Gaussian_count_after']==event['Gaussian_count_before']-event['removed_count']
            assert event['removed_count']<=int(event['Gaussian_count_before']*optimization.m2m3_max_remove_ratio)
            events.append(event);write_json(output/'OBSERVED_PRUNING_EVENTS.json',events)
        return r
    train.Scene=observed_scene;train.GaussianModel.training_setup=observed_setup;train.training_report=report
    train.randint=randint;train.m2m3_prune_scores=score;train.maybe_prune_gaussians=prune;train.GaussianModel.prune_points=prune_points
    failure=None
    try:
        torch.cuda.reset_peak_memory_stats();safe_state(False);initialization['initial_rng']=rng_hashes()
        # Native safe_state(0) immediately precedes model and Scene creation.
        train.training(Namespace(**c['model']),opt,Namespace(**c['pipeline']),[7000,30000] if not a.smoke else [],[7000,30000] if not a.smoke else [],[],None)
        g=state['scene'].gaussians
        if a.smoke:
            # Diagnostic only: invoke unchanged pruning at its real schedule gate, on the discarded short model.
            s=train.init_capacity_state(g)
            diagnostic=prune(g,Namespace(**{**c['optimization'],'iterations':30000}),18000,str(output),state['scene'].getTrainCameras(),s)
            assert diagnostic is not None and events[-1]['mask_score_available']
            runtime['diagnostic_iteration']=18000
        else:
            assert state['iteration']==30000 and runtime['optimizer_steps']==29999
            assert [x['iteration'] for x in events]==[18000,21000,24000,27000,30000]
            if a.method=='M2':
                assert camera_sequence==read_json(STAGE/f'reports/{a.sample}_M1_CAMERA_SEQUENCE.json')
            assert all(torch.isfinite(getattr(g,f)).all() for f in ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'])
            torch.save((g.capture(),30000),output/'chkpnt30000.pth')
            # Native branch already re-saves final PLY after final extra prune.
            reloaded=train.GaussianModel(3);reloaded.load_ply(str(output/'point_cloud/iteration_30000/point_cloud.ply'))
            captured,iteration=torch.load(output/'chkpnt30000.pth',map_location='cuda',weights_only=False)
            for i,f in enumerate(['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'],1):
                assert torch.equal(getattr(reloaded,f),captured[i]),f
            runtime.update(checkpoint_sha256=sha(output/'chkpnt30000.pth'),PLY_sha256=sha(output/'point_cloud/iteration_30000/point_cloud.ply'),reload_verified=True)
        torch.cuda.synchronize()
        runtime.update(status='PASS',completed_iterations=state['iteration'],Gaussian_count=len(g.get_xyz),
                       training_wall_seconds=time.perf_counter()-started,pruning_events=len(events))
        write_json(output/'FULL_FROZEN_TRAIN_CONFIG.json',c)
    except BaseException as e:
        failure=e;runtime.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        curve.close();runtime.update(end_time=now(),observed_iteration=state['iteration'],wall_seconds=time.perf_counter()-started,
                                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                                     curve_sha256=sha(STAGE/f'logs/{label}_CURVE.csv'))
        write_json(STAGE/f'reports/{label}_TRAIN_RUN.json',runtime)
        write_json(STAGE/f'reports/{label}_CAMERA_SEQUENCE.json',camera_sequence)
        write_json(STAGE/f'reports/{label}_STANDARD_PRUNING.json',standard)
        write_csv(STAGE/f'metrics/{label}_PRUNING.csv',events)
        print('TRAIN_RESULT',json.dumps(runtime,ensure_ascii=False),flush=True)
    if failure:raise failure

if __name__=='__main__':main()
