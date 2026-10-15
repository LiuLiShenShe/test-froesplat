from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import math
import time
from argparse import Namespace
from pathlib import Path
from common import *
from protocol import assert_gpu,tensor_sha
from metric_core import resize_reference,load_roi,foreground_metrics,background_geometry_metrics


def model_info(sample,group,seed):
    if seed==0:
        cfg=e0_config(sample,group)
        legacy_group='C_R1' if sample=='HongZhang' and group=='B_R0' else group
        legacy=read_json(E1/f'reports/{sample}_{legacy_group}_VALIDATION.json')
        output=Path(legacy['checkpoint_path']).parent
        cfg['model']['model_path']=str(output)
        return cfg,output,legacy
    validations=sorted((STAGE/'reports').glob(f'{run_id(sample,group,seed)}_attempt*_VALIDATION.json'))
    valid=[read_json(p) for p in validations if read_json(p)['status']=='PASS']
    assert len(valid)==1,'Each predeclared run has exactly one accepted final model'
    output=Path(valid[0]['checkpoint_path']).parent
    return read_json(output/'FULL_FROZEN_TRAIN_CONFIG.json'),output,valid[0]


def model_distribution(model,output):
    import numpy as np
    from plyfile import PlyData
    sparse=PlyData.read(output/'input.ply')['vertex']
    centroid=np.stack([sparse[k] for k in ('x','y','z')],axis=1).mean(0)
    xyz=model.get_xyz.detach().cpu().numpy()
    values={'opacity':model.get_opacity.detach().cpu().numpy().ravel(),
            'scale':model.get_scaling.detach().cpu().numpy().ravel(),
            'distance_from_sparse_centroid':np.linalg.norm(xyz-centroid,axis=1)}
    return {'Gaussian_count':len(xyz),'scale_axes':int(model.get_scaling.shape[-1]),
            **{f'{field}_{label}':float(np.quantile(array,q)) for field,array in values.items()
               for label,q in [('p01',.01),('p50',.5),('p95',.95),('p99',.99)]}}


def evaluate(sample,group,seed):
    import_paths();assert_gpu()
    import numpy as np
    import torch
    import lpips
    from PIL import Image
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    label=run_id(sample,group,seed)
    report=STAGE/f'reports/{label}_EVALUATION.json'
    assert not report.exists(),'Never replace a completed evaluation'
    config,model_path,validation=model_info(sample,group,seed)
    if sample=='HongZhang' and group=='B_R0':
        raise ValueError('HongZhang B must be an explicit C alias at summary time')
    render_dir=E2_DATA/'evaluation'/sample/group/f'seed_{seed}'
    assert not render_dir.exists(),'Retain failed evaluation evidence; use explicit recovery for an incomplete evaluation'
    render_dir.mkdir(parents=True)
    started=time.perf_counter()
    captured,iteration=torch.load(validation['checkpoint_path'],map_location='cuda',weights_only=False)
    assert iteration==30000
    model=GaussianModel(config['model']['sh_degree'])
    scene=Scene(Namespace(**config['model']),model,load_iteration=30000,shuffle=False)
    for index,field in enumerate(('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'),1):
        assert torch.equal(getattr(model,field),captured[index]),field
    del captured;gc.collect();torch.cuda.empty_cache()
    cameras={c.image_name:c for c in scene.getTestCameras()}
    refs=[r for r in read_csv(E0/'reports/EVALUATION_REFERENCE_MANIFEST.csv') if r['sample']==sample]
    assert set(cameras)=={r['loader_image_name'] for r in refs}
    proxies={Path(r['registered_name']).stem:r for r in read_csv(E0/'MATCHED_2DGS_DATASET_MANIFEST.csv') if r['sample']==sample and r['group']=='B_R0'}
    original={r['loader_image_name']:r for r in read_csv(E1/'HELDOUT_PER_VIEW_METRICS.csv') if r['sample']==sample and r['group']==group}
    perceptual=lpips.LPIPS(net='vgg',version='0.1',verbose=False).eval().cuda()
    weights=Path(lpips.__file__).parent/'weights/v0.1/vgg.pth'
    vgg=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
    metric_versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','lpips','Pillow','numpy')}
    metric_versions.update({'SSIM_source_sha256':sha256(GS/'utils/loss_utils.py'),'PSNR_source_sha256':sha256(GS/'utils/image_utils.py'),
                           'LPIPS_weights_sha256':sha256(weights),'VGG_weights_sha256':sha256(vgg),
                           'range':'float32 [0,1]; native RGB clamp','LPIPS':'VGG v0.1 normalize=True','background':[0,0,0]})
    legacy_metric=read_json(E1/f'reports/{sample}_{group}_VALIDATION.json')['metric_versions'] if seed==0 else read_json(E1/'reports/WanNianQing1_B_R0_VALIDATION.json')['metric_versions']
    for package in ('torch','torchvision','lpips','Pillow','numpy'):
        assert metric_versions[package]==legacy_metric[package],package
    for new,old in [('SSIM_source_sha256','ssim_source_sha256'),('PSNR_source_sha256','psnr_source_sha256'),('LPIPS_weights_sha256','lpips_weights_sha256'),('VGG_weights_sha256','vgg16_weights_sha256')]:
        assert metric_versions[new]==legacy_metric[old],new
    distribution=model_distribution(model,model_path)
    results=[];fg_results=[];bg_results=[]
    with torch.no_grad():
        for ref in refs:
            name=ref['loader_image_name'];camera=cameras[name]; proxy=proxies[name]
            assert sha256(ref['reference_rgb_path'])==ref['common_rgb_sha256']
            assert sha256(proxy['mask_path'])==proxy['mask_sha256']
            reference_cpu=resize_reference(ref['reference_rgb_path'],camera.image_width,camera.image_height)
            roi_cpu=load_roi(proxy['mask_path'],camera.image_width,camera.image_height)
            reference=reference_cpu.cuda();roi=roi_cpu.cuda()
            package=render(camera,model,Namespace(**config['pipeline']),torch.zeros(3,device='cuda'))
            replay_rgb=package['render'].clamp(0,1);replay_alpha=package['rend_alpha'][0]
            assert torch.isfinite(replay_rgb).all() and torch.isfinite(replay_alpha).all()
            if seed==0:
                old=original[name]; assert sha256(old['float_render_path'])==old['float_render_sha256']
                frozen=torch.load(old['float_render_path'],map_location='cpu',weights_only=True)
                rgb=frozen['rgb'].cuda();alpha=frozen['alpha'].cuda()
                assert tensor_sha(reference_cpu)==old['reference_resized_tensor_sha256']
                replay_error=(rgb-replay_rgb).abs().max().item();alpha_error=(alpha-replay_alpha).abs().max().item()
                legacy_float=old['float_render_path'];legacy_sha=old['float_render_sha256']
                assert proxy['mask_sha256']==old['R0_proxy_mask_sha256']
            else:
                rgb=replay_rgb;alpha=replay_alpha;replay_error=0.;alpha_error=0.;legacy_float='';legacy_sha=''
            identity={'sample':sample,'group':group,'seed':seed,'loader_image_name':name,'registered_name':ref['registered_name'],
                      'registered_image_id':ref['image_id'],'render_width':camera.image_width,'render_height':camera.image_height,
                      'reference_path':ref['reference_rgb_path'],'reference_original_sha256':ref['common_rgb_sha256'],
                      'reference_resized_tensor_sha256':tensor_sha(reference_cpu),'ROI_path':proxy['mask_path'],
                      'ROI_original_sha256':proxy['mask_sha256'],'ROI_resized_tensor_sha256':tensor_sha(roi_cpu),
                      'ROI_is_independent_GT':False,'status':'PASS','model_checkpoint_sha256':validation['checkpoint_sha256']}
            primary={**identity,'PSNR':psnr(rgb[None],reference[None]).mean().item(),'SSIM':ssim(rgb,reference).item(),
                     'LPIPS':perceptual(rgb[None],reference[None],normalize=True).item(),
                     'float_source_policy':'E1_ORIGINAL_FLOAT' if seed==0 else 'NEW_REAL_FLOAT_RENDER',
                     'legacy_float_path':legacy_float,'legacy_float_sha256':legacy_sha,
                     'geometry_replay_RGB_max_abs_error':replay_error,'geometry_replay_alpha_max_abs_error':alpha_error,
                     'empty_render':bool(alpha.max().item()<=1e-6)}
            if seed==0:
                for metric in ('PSNR','SSIM','LPIPS'):
                    assert abs(primary[metric]-float(original[name][metric]))<=1e-6,(name,metric)
            foreground={**identity,**foreground_metrics(rgb,reference,alpha,roi,ssim,perceptual)}
            # Geometry replay uses its own coherent alpha with depth and normal.
            background={**identity,**background_geometry_metrics(rgb,alpha,roi,package['surf_depth'][0],package['rend_normal'],package['surf_normal']),
                        **distribution,'geometry_alpha_support_source':'GEOMETRY_REPLAY_ALPHA' if seed==0 else 'SAME_RENDER_ALPHA'}
            if seed==0:
                geometry=background_geometry_metrics(replay_rgb,replay_alpha,roi,package['surf_depth'][0],package['rend_normal'],package['surf_normal'])
                for key in geometry:
                    if key.startswith(('depth_','normal_')): background[key]=geometry[key]
                assert abs(foreground['foreground_PSNR']-float(original[name]['R0_proxy_foreground_psnr']))<1e-6
            tensor_path=render_dir/f'{name}_FLOAT_RENDER.pt'
            torch.save({'rgb':rgb.cpu(),'alpha':alpha.cpu(),'depth':package['surf_depth'][0].cpu(),
                        'normal':package['rend_normal'].cpu(),'surface_normal':package['surf_normal'].cpu(),
                        'geometry_replay_alpha':replay_alpha.cpu()},tensor_path)
            for row in (primary,foreground,background):row.update({'float_render_path':str(tensor_path),'float_render_sha256':sha256(tensor_path)})
            results.append(primary);fg_results.append(foreground);bg_results.append(background)
            print(f'EVALUATED {label} {name} raw_PSNR={primary["PSNR"]:.6f} foreground_PSNR={foreground["foreground_PSNR"]:.6f}',flush=True)
            del package,reference,roi,rgb,alpha,replay_rgb,replay_alpha
    write_csv(STAGE/f'metrics/{label}_FULL.csv',results)
    write_csv(STAGE/f'metrics/{label}_FOREGROUND.csv',fg_results)
    write_csv(STAGE/f'metrics/{label}_BACKGROUND_GEOMETRY.csv',bg_results)
    result={'sample':sample,'group':group,'seed':seed,'status':'PASS','heldout':len(results),'model_path':str(model_path),
            'checkpoint_sha256':validation['checkpoint_sha256'],'metric_versions':metric_versions,'Gaussian_distribution':distribution,
            'evaluation_wall_seconds':time.perf_counter()-started,'timestamp':now(),'GPU_UUID':GPU_UUID,
            'geometry_replay_max_RGB_error':max(r['geometry_replay_RGB_max_abs_error'] for r in results),
            'geometry_replay_max_alpha_error':max(r['geometry_replay_alpha_max_abs_error'] for r in results),
            'evaluator_sha256':sha256(Path(__file__)),'metric_core_sha256':sha256(STAGE/'scripts/metric_core.py')}
    write_json(report,result)
    print('EVALUATION_COMPLETE',label,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--sample');p.add_argument('--group');p.add_argument('--seed',type=int);p.add_argument('--all',action='store_true')
    a=p.parse_args()
    if a.all:
        matrix=[(r['sample'],r['group'],0) for r in read_csv(E1/'TRAINING_RUN_MANIFEST.csv')]
        matrix+=list(TRAIN_ORDER)
        for sample,group,seed in matrix:
            if (STAGE/f'reports/{run_id(sample,group,seed)}_EVALUATION.json').exists():continue
            subprocess_args=[str(PYTHON),str(Path(__file__)),'--sample',sample,'--group',group,'--seed',str(seed)]
            import subprocess
            log=STAGE/f'logs/{run_id(sample,group,seed)}_EVALUATION.txt'
            with log.open('xb') as h: subprocess.run(subprocess_args,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,check=True)
    else:evaluate(a.sample,a.group,a.seed)

if __name__=='__main__':main()
