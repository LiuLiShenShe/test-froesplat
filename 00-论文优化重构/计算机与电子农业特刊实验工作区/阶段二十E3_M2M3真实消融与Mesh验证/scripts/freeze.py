"""CPU-only preregistration; no model outputs are inspected."""
from common import *
import ast, copy, difflib, numpy as np

def main():
    assert not (STAGE/'configs/FREEZE.json').exists()
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASELINE
    protected={}
    def protect(p):
        p=Path(p).resolve();protected[str(p)]=sha(p)
    # Git covers every tracked historical file; SHA256 additionally covers all
    # relevant source, reports, manuscript and untracked frozen training assets.
    assert not subprocess.check_output(['git','diff','--name-only'])
    assert not subprocess.check_output(['git','diff','--cached','--name-only'])
    (STAGE/'reports/BASELINE_GIT_INDEX.txt').write_bytes(subprocess.check_output(['git','ls-files','--stage']))
    for raw in subprocess.check_output(['git','ls-files','-z'] ).split(b'\0'):
        if raw:
            p=ROOT/os.fsdecode(raw)
            relevant=(p.is_relative_to(GS) or p.is_relative_to(BRANCH)
                      or p.is_relative_to(E0) or p.is_relative_to(E1) or p.is_relative_to(E2)
                      or p.suffix in {'.tex','.bib'}
                      or (p.is_relative_to(WORK) and any(x.startswith(('阶段二十A','阶段二十B')) for x in p.parts)))
            if p.is_file() and relevant:protect(p)
    refs=[];mesh_configs=[];train_configs=[]
    split=rows(E0/'TRAIN_VALIDATION_SPLIT.csv')
    matched=[r for r in rows(E0/'MATCHED_2DGS_DATASET_MANIFEST.csv') if r['group']=='B_R0']
    for r in matched:
        for field,hashfield in [('loader_image_path','image_sha256'),('mask_path','mask_sha256'),('common_rgb_path','common_rgb_sha256')]:
            assert sha(r[field])==r[hashfield];protect(r[field])
    import_paths()
    from utils.render_utils import focus_point_fn
    from argparse import ArgumentParser
    # Read branch argument defaults without importing GPU training or allocating CUDA.
    import importlib.util
    spec=importlib.util.spec_from_file_location('branch_arguments',BRANCH/'arguments/__init__.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    parser=ArgumentParser();mod.OptimizationParams(parser)
    branch_defaults=vars(parser.parse_args([]))
    for scene in SCENES:
        r=reference(scene)
        for pathkey,hashkey in [('checkpoint_path','checkpoint_sha256'),('PLY_path','PLY_sha256')]:
            assert sha(r[pathkey])==r[hashkey];protect(r[pathkey])
        output=Path(r['checkpoint_path']).parent
        c=config(scene)
        for p in (Path(c['model']['source_path'])/'sparse/0').iterdir():
            if p.is_file():protect(p)
        protect(output/'input.ply');protect(output/'cameras.json');protect(output/'cfg_args')
        cameras=read_json(output/'cameras.json')
        train_names={x['loader_image_name'] for x in split if x['sample']==scene and x['split']=='train'}
        cams=[x for x in cameras if x['img_name'] in train_names]
        assert len(cams)==len(train_names)
        poses=np.stack([np.concatenate([np.array(x['rotation']),np.array(x['position'])[:,None]],1) for x in cams])
        poses=poses@np.diag([1,-1,-1,1])
        center=focus_point_fn(poses)
        radius=float(np.linalg.norm(np.array([x['position'] for x in cams])-center,axis=-1).min())
        depth=2*radius
        mc={'sample':scene,'mode':'bounded','depth_ratio':0.0,'depth_trunc':depth,'voxel_size':depth/1024,
            'sdf_trunc':5*depth/1024,'mesh_res':1024,'camera_focus':center.tolist(),'camera_radius':radius,
            'scale_source':'Frozen train camera poses; native focus_point_fn and min radius; no new model quality',
            'integration_cameras':sorted(train_names),'mask_background':True,'mask_threshold':0.5,
            'background_rule':'Native gt_alpha_mask from frozen RGBA; no rendered-alpha threshold',
            'num_cluster':1,'postprocessing':'Native post_process_mesh top 1 cluster threshold and minimum 50 triangles; ties retained',
            'SH_for_fusion':0,'sensitivity_voxel_factors':[0.5,2.0],
            'sensitivity_scope':'All three M0 references, supplementary only; same depth_trunc, sdf_trunc=5*voxel'}
        mp=STAGE/f'configs/{scene}_MESH.json';write_json(mp,mc);mesh_configs.append(mp)
        refs.append({**r,'method':'M0','input_ply_path':str(output/'input.ply'),'input_ply_sha256':sha(output/'input.ply'),
                     'camera_json_path':str(output/'cameras.json'),'camera_json_sha256':sha(output/'cameras.json'),
                     'mesh_config_path':str(mp),'mesh_config_sha256':sha(mp),'independent_3D_GT':'NONE','physical_scale':'UNAVAILABLE'})
        for method in ['M1','M2']:
            tc=copy.deepcopy(c);tc['model']['model_path']=str(model_path(scene,method))
            opt={**branch_defaults,**c['optimization']}
            opt.update(pruning_mode='mask',pruning_start_iter=18000,pruning_interval=3000,
                       pruning_opacity_threshold=.005,pruning_brightness_threshold=.01,pruning_mask_threshold=.45,
                       pruning_mask_max_views=12,pruning_max_remove_ratio=.03,pruning_mask_score_weight=3.,
                       capacity_control_mode='m2m3' if method=='M1' else 'm2m3_floor',
                       capacity_floor_ratio=.4,capacity_floor_reference='max_seen',capacity_floor_count=0,
                       capacity_floor_start_iter=18000,capacity_floor_end_iter=0,
                       m2m3_score_mode='topology',m2m3_region_mode='foreground',m2m3_opacity_weight=1.,
                       m2m3_brightness_weight=1.,m2m3_mask_weight=3.,m2m3_scale_weight=.5,m2m3_view_weight=.5,
                       m2m3_max_remove_ratio=.03,save_pruning_report=True,save_capacity_report=True,capacity_report_interval=1000)
            tc['optimization']=opt;tc['all_parameters']={**tc['model'],**opt,**tc['pipeline']}
            tc['e3']={'method':method,'seed':0,'sample':scene,'GPU_UUID':GPU_UUID,
                      'parameter_source':'20260610_151144_full/CaoMei2/fg2dgs_m2m3_floor40/command.txt plus source defaults',
                      'old_pilot_losses_and_init_NOT_inherited':True,'heldout_quality_early_stop':False}
            p=STAGE/f'configs/{scene}_{method}_TRAIN.json';write_json(p,tc);train_configs.append(p)
    old=ROOT/'00-论文优化重构/m2m3实验/实验输出/runs/20260610_151144_full/CaoMei2/fg2dgs_m2m3_floor40/command.txt'
    (STAGE/'configs/PILOT_PARAMETER_SOURCE.txt').write_text(old.read_text())
    diff={}
    shared=['scene/__init__.py','scene/gaussian_model.py','scene/dataset_readers.py','scene/cameras.py','utils/camera_utils.py','gaussian_renderer/__init__.py','utils/loss_utils.py']
    for f in shared:
        assert (GS/f).read_bytes()==(BRANCH/f).read_bytes();diff[f]='BYTE_IDENTICAL'
    a=ast.parse((GS/'train.py').read_text());b=ast.parse((BRANCH/'train.py').read_text())
    fa={x.name:x for x in a.body if isinstance(x,ast.FunctionDef)}
    fb={x.name:x for x in b.body if isinstance(x,ast.FunctionDef)}
    for name in ['rgb_reconstruction_loss','mask_loss_terms','read_view_weights','get_view_weight','gaussian_brightness']:
        assert ast.dump(fa[name],include_attributes=False)==ast.dump(fb[name],include_attributes=False)
        diff[name]='AST_IDENTICAL'
    for name in ['train.py','arguments/__init__.py','utils/general_utils.py']:
        text=''.join(difflib.unified_diff((GS/name).read_text().splitlines(True),(BRANCH/name).read_text().splitlines(True),fromfile='main/'+name,tofile='branch/'+name))
        (STAGE/'reports'/('BRANCH_DIFF_'+name.replace('/','_')+'.txt')).write_text(text)
    write_json(STAGE/'reports/BASELINE_EQUIVALENCE.json',{'status':'SOURCE_EQUIVALENT_DISABLED_PATH','shared':diff,
        'disabled_conditions':{'pruning_mode':'none','capacity_control_mode':'none'},
        'training_diff_review':'Only capacity validation/state/no-op update/extra pruning and logging added. Disabled prune returns before tensors or RNG. update_capacity_state(None) returns. No change to optimizer, loss, densification or schedules.',
        'general_utils_difference':'colormap Agg tostring_rgb fallback only, no seed change; use same E1 compatibility wrapper',
        'historical_limit':'Seed0 initial RNG hash absent; source-path equivalence, not claimed historical state identity',
        'new_training_count':6,'M0_branch_control_required':False})
    write_csv(STAGE/'MESH_REFERENCE_MANIFEST.csv',refs)
    write_csv(STAGE/'M2M3_ABLATION_CONFIGS.csv',[{'sample':read_json(p)['e3']['sample'],'method':read_json(p)['e3']['method'],
                'seed':0,'iterations':30000,'config_path':str(p),'config_sha256':sha(p),'GPU_UUID':GPU_UUID} for p in train_configs])
    all_cfg=mesh_configs+train_configs+[STAGE/'reports/BASELINE_EQUIVALENCE.json',STAGE/'configs/PILOT_PARAMETER_SOURCE.txt']
    freeze={'time':now(),'baseline_commit':BASELINE,'status':'FROZEN_BEFORE_FIRST_E3_GPU_WORK','GPU_UUID':GPU_UUID,
            'training_order':[(s,m) for s in SCENES for m in ['M1','M2']],
            'failed_run_policy':'Keep all evidence. At most one full fresh same-config retry after execution fault, never quality rerun. No checkpoint selection.',
            'short_test_policy':'Discard independent 20-iteration test; run actual schedule gates at virtual iteration18000 only in diagnostic process; formal runs always fresh native30000.',
            'protected':[{'path':p,'sha256':h} for p,h in protected.items()],
            'configs':[{'path':str(p),'sha256':sha(p)} for p in all_cfg]}
    write_json(STAGE/'configs/FREEZE.json',freeze)
    print('FROZEN',len(protected),'protected files; 3 references; 6 new trainings; main TSDF and supplementary sensitivity')

if __name__=='__main__':main()
