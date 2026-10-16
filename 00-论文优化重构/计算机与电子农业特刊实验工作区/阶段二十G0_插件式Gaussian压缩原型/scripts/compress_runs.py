"""Independent CPU pruning, followed by untouched native PLY reloading on GPU 1."""
from common import *
import gc
import shutil
from argparse import Namespace


def main():
    assert_gpu();p=plan();import_paths()
    import numpy as np
    import torch
    from plugins.gaussian_compression import compress_model
    from plugins.gaussian_compression.io import read_model
    from plugins.gaussian_compression.evaluation.native import load_native
    from plugins.gaussian_compression.scoring import AdaptationBlocked
    if (STAGE/'COMPRESSION_RUN_MANIFEST.csv').exists():raise FileExistsError('Do not overwrite experiment results')
    assert sha(p['source_model'])==p['source_model_sha256']
    src=read_model(p['source_model']);manifest=[];validations=[]
    for run in p['runs']:
        output=DATA/'models'/run['run_id']
        result=compress_model(p['source_model'],output,run['method'],run['prune_ratio'],
            {'enabled':True,'seed':p['random_seed'],'active_sh_degree':p['SH_degree'],
             'expected_source_sha256':p['source_model_sha256'],'recovery':False})
        meta=read_json(result.manifest_path)
        keep=np.load(result.retained_indices_path,allow_pickle=False)
        removed=np.load(result.removed_indices_path,allow_pickle=False)
        reload=read_model(result.model_path)
        assert np.array_equal(src.vertices[keep],reload.vertices)
        assert np.array_equal(np.sort(np.r_[keep,removed]),np.arange(src.count))
        g=load_native(result.model_path,GS,p['SH_degree'])
        count=len(g.get_xyz);assert count==meta['output_count']
        shapes={k:list(getattr(g,k).shape) for k in ['_xyz','_opacity','_scaling','_rotation','_features_dc','_features_rest']}
        # Conventional model-directory layout for the original Scene/Viewer loader.
        native=output/'native_model';cloud=native/'point_cloud/iteration_30000';cloud.mkdir(parents=True)
        (cloud/'point_cloud.ply').symlink_to('../../../pruned_model.ply')
        native_config={**p['model'],'model_path':str(native)}
        (native/'cfg_args').write_text(str(Namespace(**native_config))+'\n')
        shutil.copyfile(Path(p['source_native_model_dir'])/'cameras.json',native/'cameras.json')
        assert sha(cloud/'point_cloud.ply')==meta['output_model_sha256']
        validations.append({'run_id':run['run_id'],'status':'PASS','count':count,'native_shapes':shapes,
                            'schema_equal':src.schema==reload.schema,'SH_degree':reload.sh_degree,'active_SH_degree':g.active_sh_degree,
                            'every_retained_scalar_equal':True,'partition_no_duplicates_or_illegal_indices':True,
                            'native_loader':'untouched scene.gaussian_model.GaussianModel.load_ply',
                            'native_model_directory':str(native)})
        manifest.append({**run,'scene':p['scene'],'group':p['group'],'training_seed':0,'M2M3':'M0',
                         'source_model_path':p['source_model'],'source_model_sha256':p['source_model_sha256'],
                         'source_count':src.count,'Gaussian_count':count,'removed_count':len(removed),
                         'actual_prune_ratio':meta['actual_prune_ratio'],'source_PLY_bytes':meta['source_bytes'],
                         'PLY_bytes':meta['output_bytes'],'PLY_byte_reduction_ratio':1-meta['output_bytes']/meta['source_bytes'],
                         'pruned_model_path':str(result.model_path),'output_model_sha256':meta['output_model_sha256'],
                         'retained_indices_path':str(result.retained_indices_path),'retained_indices_sha256':sha(result.retained_indices_path),
                         'removed_indices_path':str(result.removed_indices_path),'removed_indices_sha256':sha(result.removed_indices_path),
                         'importance_scores_path':str(output/'importance_scores.npy'),'importance_scores_sha256':sha(output/'importance_scores.npy'),
                         'compression_config_path':str(result.manifest_path),'compression_config_sha256':sha(result.manifest_path),
                         'scoring_seconds':meta['scoring_seconds'],'compression_wall_seconds':meta['compression_wall_seconds'],
                         'recovery_steps':0,'model_load_success':True,'input_unchanged':True,'active_SH_degree':p['SH_degree']})
        del g;gc.collect();torch.cuda.empty_cache()
        print('COMPRESSED',run['run_id'],count,meta['output_bytes'],flush=True)
    # Exercise the public disabled path on the actual source model.
    off=compress_model(p['source_model'],DATA/'disabled_identity','PUP_2DGS_ADAPTED',.6,{'enabled':False})
    assert off.output_model_sha256==p['source_model_sha256']
    blocked=[]
    for method in ['PUP_2DGS_ADAPTED','LIGHTGAUSSIAN_2DGS_ADAPTED']:
        out=DATA/'negative_attempts'/method
        try:compress_model(p['source_model'],out,method,.4,{'enabled':True})
        except AdaptationBlocked as e:
            blocked.append({'method':method,'error':str(e),'manifest':str(out/'compression_config.json'),
                            'manifest_sha256':sha(out/'compression_config.json'),'produced_pruned_PLY':(out/'pruned_model.ply').exists()})
        else:raise AssertionError('Blocked method silently ran')
    write_csv(STAGE/'COMPRESSION_RUN_MANIFEST.csv',manifest)
    write_json(STAGE/'reports/MODEL_VALIDATION.json',{'status':'PASS','runs':validations,
        'disabled_identity':{'status':'PASS','model_path':str(off.model_path),'sha256':off.output_model_sha256},
        'blocked_method_attempts':blocked,'source_final_sha256':sha(p['source_model'])})

if __name__=='__main__':main()
