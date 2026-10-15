"""Common-reference held-out evaluator; legacy M0 floats remain immutable."""
from common import *
from argparse import ArgumentParser,Namespace
import gc,time,importlib.metadata

def distribution(g):
    import numpy as np
    result={'Gaussian_count':len(g.get_xyz)}
    for field,array in [('opacity',g.get_opacity.detach().cpu().numpy()),('scale',g.get_scaling.detach().cpu().numpy())]:
        for label,q in [('p01',.01),('p50',.5),('p95',.95),('p99',.99)]:result[f'{field}_{label}']=float(np.quantile(array,q))
    return result

def main():
    p=ArgumentParser();p.add_argument('--sample',choices=SCENES,required=True);p.add_argument('--method',choices=['M0','M1','M2'],required=True);a=p.parse_args()
    verify_frozen();label=f'{a.sample}_{a.method}'
    report=STAGE/f'reports/{label}_EVALUATION.json';assert not report.exists()
    core=metric_module()
    if a.method=='M0':
        group=reference(a.sample)['actual_group'];base=f'{a.sample}_{group}_seed0'
        # Locate by immutable table identity, without inventing a render replay.
        identity={'sample':a.sample,'method':'M0','seed':0,'status':'PASS','float_source_policy':'E2_REUSED_E1_FLOAT_RGB_ALPHA'}
        full=[r for r in rows(E2/'MULTISEED_PER_VIEW_METRICS.csv') if r['sample']==a.sample and r['group']==group and r['seed']=='0']
        fg=[r for r in rows(E2/'FOREGROUND_PROXY_METRICS.csv') if r['sample']==a.sample and r['group']==group and r['seed']=='0']
        bg=[r for r in rows(E2/'BACKGROUND_GEOMETRY_PROXY_METRICS.csv') if r['sample']==a.sample and r['group']==group and r['seed']=='0']
        assert full and len(full)==len(fg)==len(bg)
        for values,kind in [(full,'FULL'),(fg,'FOREGROUND'),(bg,'BACKGROUND_GEOMETRY')]:
            for r in values:
                assert sha(r['float_render_path'])==r['float_render_sha256'];r.update(identity)
            write_csv(STAGE/f'metrics/{label}_{kind}.csv',values)
        write_json(report,{**identity,'heldout':len(full),'reused_from':str(E2),'checkpoint_sha256':reference(a.sample)['checkpoint_sha256'],
                           'geometry_limit':'E2 retained historical RGB/alpha with depth/normal replay. Geometry alpha stored independently; no independent 3D GT.'})
        print('REUSED_M0_METRICS',label,len(full),flush=True);return
    gpu_idle();import_paths()
    import torch,numpy as np,lpips
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    c=read_json(STAGE/f'configs/{label}_TRAIN.json');output=model_path(a.sample,a.method)
    run=read_json(STAGE/f'reports/{label}_TRAIN_RUN.json');assert run['status']=='PASS' and run['completed_iterations']==30000
    target=DATA/'evaluation'/a.sample/a.method;assert not target.exists();target.mkdir(parents=True)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    g=GaussianModel(3);scene=Scene(Namespace(**c['model']),g,load_iteration=30000,shuffle=False)
    captured,it=torch.load(output/'chkpnt30000.pth',map_location='cpu',weights_only=False);assert it==30000
    for i,f in enumerate(['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'],1):assert torch.equal(getattr(g,f).detach().cpu(),captured[i]),f
    del captured;gc.collect()
    cams={x.image_name:x for x in scene.getTestCameras()}
    refs=[r for r in rows(E0/'reports/EVALUATION_REFERENCE_MANIFEST.csv') if r['sample']==a.sample]
    proxies={Path(r['registered_name']).stem:r for r in rows(E0/'MATCHED_2DGS_DATASET_MANIFEST.csv') if r['sample']==a.sample and r['group']=='B_R0'}
    assert set(cams)=={r['loader_image_name'] for r in refs}
    lp=lpips.LPIPS(net='vgg',version='0.1',verbose=False).eval().cuda()
    legacy=read_json(E1/'reports/WanNianQing1_B_R0_VALIDATION.json')['metric_versions']
    versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','lpips','Pillow','numpy']}
    for n in versions:assert versions[n]==legacy[n],n
    for path,key in [(GS/'utils/loss_utils.py','ssim_source_sha256'),(GS/'utils/image_utils.py','psnr_source_sha256'),
                     (Path(lpips.__file__).parent/'weights/v0.1/vgg.pth','lpips_weights_sha256'),
                     (Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth','vgg16_weights_sha256')]:assert sha(path)==legacy[key],key
    dist=distribution(g);full=[];fg=[];bg=[];inference=[]
    with torch.no_grad():
        for ref in refs:
            name=ref['loader_image_name'];cam=cams[name];proxy=proxies[name]
            assert sha(ref['reference_rgb_path'])==ref['common_rgb_sha256']
            assert sha(proxy['mask_path'])==proxy['mask_sha256']
            ref_cpu=core.resize_reference(ref['reference_rgb_path'],cam.image_width,cam.image_height)
            roi_cpu=core.load_roi(proxy['mask_path'],cam.image_width,cam.image_height)
            reference_gpu=ref_cpu.cuda();roi=roi_cpu.cuda()
            torch.cuda.synchronize();mark=time.perf_counter()
            package=render(cam,g,Namespace(**c['pipeline']),torch.zeros(3,device='cuda'))
            torch.cuda.synchronize();inference.append(time.perf_counter()-mark)
            rgb=package['render'].clamp(0,1);alpha=package['rend_alpha'][0]
            data={'rgb':rgb.cpu(),'alpha':alpha.cpu(),'depth':package['surf_depth'][0].cpu(),'normal':package['rend_normal'].cpu(),
                  'surface_normal':package['surf_normal'].cpu(),'geometry_replay_alpha':alpha.cpu()}
            assert all(torch.isfinite(v).all() for v in data.values())
            path=target/f'{name}_FLOAT_RENDER.pt';torch.save(data,path)
            identity={'sample':a.sample,'method':a.method,'seed':0,'loader_image_name':name,'registered_name':ref['registered_name'],
                      'render_width':cam.image_width,'render_height':cam.image_height,'reference_path':ref['reference_rgb_path'],
                      'reference_original_sha256':ref['common_rgb_sha256'],'reference_resized_tensor_sha256':tensor_sha(ref_cpu),
                      'ROI_path':proxy['mask_path'],'ROI_original_sha256':proxy['mask_sha256'],'ROI_resized_tensor_sha256':tensor_sha(roi_cpu),
                      'ROI_is_independent_GT':False,'status':'PASS','model_checkpoint_sha256':run['checkpoint_sha256'],
                      'float_render_path':str(path),'float_render_sha256':sha(path),'float_source_policy':'NEW_REAL_FLOAT_RENDER'}
            full.append({**identity,'PSNR':psnr(rgb[None],reference_gpu[None]).mean().item(),'SSIM':ssim(rgb,reference_gpu).item(),
                         'LPIPS':lp(rgb[None],reference_gpu[None],normalize=True).item()})
            fg.append({**identity,**core.foreground_metrics(rgb,reference_gpu,alpha,roi,ssim,lp)})
            bg.append({**identity,**core.background_geometry_metrics(rgb,alpha,roi,package['surf_depth'][0],package['rend_normal'],package['surf_normal']),**dist})
            print('EVALUATED',label,name,full[-1]['PSNR'],fg[-1]['foreground_PSNR'],flush=True)
    for data,kind in [(full,'FULL'),(fg,'FOREGROUND'),(bg,'BACKGROUND_GEOMETRY')]:write_csv(STAGE/f'metrics/{label}_{kind}.csv',data)
    write_json(report,{'sample':a.sample,'method':a.method,'status':'PASS','heldout':len(full),'seed':0,'GPU_UUID':GPU_UUID,
                       'metric_versions':versions,'model_distribution':dist,'wall_seconds':time.perf_counter()-start,
                       'inference_render_mean_seconds':float(np.mean(inference)),'inference_first_view_seconds':inference[0],
                       'inference_median_seconds':float(np.median(inference)),
                       'inference_limit':'One unsmoothed pass; includes first-call overhead; not a benchmark speed claim',
                       'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'timestamp':now()})
    print('EVALUATION_COMPLETE',label,flush=True)

if __name__=='__main__':main()
