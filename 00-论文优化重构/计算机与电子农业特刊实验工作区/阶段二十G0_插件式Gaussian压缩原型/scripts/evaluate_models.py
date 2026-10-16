"""Same held-out cameras and frozen E1/E2 metrics for every true pruned PLY."""
from common import *
from native_context import cameras,reference_records
from argparse import Namespace
import gc
import math
import time
import traceback


def main():
    assert_gpu();p=plan();import_paths()
    import cv2
    import numpy as np
    import torch
    import lpips
    import PIL
    import importlib.metadata as metadata
    from gaussian_renderer import render
    from plugins.gaussian_compression.evaluation.native import load_native
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    torch.set_num_threads(4)
    metric=module(E2/'scripts/metric_core.py','frozen_phase20e2_metric_core')
    if (STAGE/'reports/QUALITY_EVALUATION.json').exists():raise FileExistsError('Quality evaluation already recorded')
    result={'status':'RUNNING','started_at':now(),'gpu_uuid':GPU_UUID,'plan_sha256':sha(STAGE/'configs/EXPERIMENT_PLAN.json')}
    allrows=[];view_manifest=[];replay=[]
    try:
        scene,views=cameras(p);refs,rois=reference_records(p)
        gc.collect();torch.cuda.empty_cache()
        perceptual=lpips.LPIPS(net='vgg',version='0.1',verbose=False).eval().cuda()
        bg=torch.tensor(p['background'],dtype=torch.float32,device='cuda')
        pipe=Namespace(**p['pipeline'])
        original={r['loader_image_name']:r for r in rows(E1/'HELDOUT_PER_VIEW_METRICS.csv') if r['sample']==p['scene'] and r['group']=='B_R0'}
        camera_rows=[]
        for cam in views:
            ref=metric.resize_reference(refs[cam.image_name]['reference_rgb_path'],cam.image_width,cam.image_height)
            roi=metric.load_roi(rois[cam.image_name]['mask_path'],cam.image_width,cam.image_height)
            assert tensor_sha(ref)==original[cam.image_name]['reference_resized_tensor_sha256']
            assert rois[cam.image_name]['mask_sha256']==original[cam.image_name]['R0_proxy_mask_sha256']
            camera_rows.append({'camera':cam.image_name,'width':cam.image_width,'height':cam.image_height,
                'world_view_transform_sha256':tensor_sha(cam.world_view_transform),'full_proj_transform_sha256':tensor_sha(cam.full_proj_transform),
                'FoVx':cam.FoVx,'FoVy':cam.FoVy,'reference_rgb_path':refs[cam.image_name]['reference_rgb_path'],
                'reference_file_sha256':refs[cam.image_name]['common_rgb_sha256'],'reference_tensor_sha256':tensor_sha(ref),
                'ROI_path':rois[cam.image_name]['mask_path'],'ROI_sha256':rois[cam.image_name]['mask_sha256'],
                'ROI_pixels':int(roi.sum()),'ROI_is_independent_GT':False})
        write_csv(STAGE/'configs/HELDOUT_CAMERA_REFERENCE_INDEX.csv',camera_rows)
        runrows=rows(STAGE/'COMPRESSION_RUN_MANIFEST.csv')
        for run in runrows:
            start=time.perf_counter();name=run['run_id'];out=DATA/'renders'/name;out.mkdir(parents=True,exist_ok=False)
            assert sha(run['pruned_model_path'])==run['output_model_sha256']
            g=load_native(run['pruned_model_path'],GS,p['SH_degree'])
            with torch.no_grad():
                for cam in views:
                    key=cam.image_name;r=refs[key];roi_rec=rois[key]
                    reference=metric.resize_reference(r['reference_rgb_path'],cam.image_width,cam.image_height).cuda()
                    roi=metric.load_roi(roi_rec['mask_path'],cam.image_width,cam.image_height).cuda()
                    package=render(cam,g,pipe,bg)
                    assert all(torch.isfinite(package[k]).all() for k in ['render','rend_alpha','surf_depth','rend_normal','surf_normal'])
                    raw=package['render'];rgb=raw.clamp(0,1);alpha=package['rend_alpha'].squeeze(0)
                    record={'run_id':name,'method':run['method'],'prune_ratio':run['prune_ratio'],'camera':key,
                            'width':cam.image_width,'height':cam.image_height,'PSNR':psnr(rgb[None],reference[None]).mean().item(),
                            'SSIM':ssim(rgb,reference).item(),'LPIPS':perceptual(rgb[None],reference[None],normalize=True).item(),
                            'unclamped_RGB_min':raw.min().item(),'unclamped_RGB_max':raw.max().item()}
                    record.update(metric.foreground_metrics(rgb,reference,alpha,roi,ssim,perceptual))
                    outside=~roi
                    record.update(outside_ROI_RGB_energy=rgb[:,outside].square().mean().item(),
                        outside_ROI_alpha_mean=alpha[outside].mean().item(),alpha_coverage=(alpha>.5).float().mean().item(),
                        ROI_alpha_mean=alpha[roi].mean().item(),valid_depth_fraction=(package['surf_depth']>0).float().mean().item())
                    roi_cpu=roi.cpu().numpy().astype(np.uint8)
                    inner=cv2.erode(roi_cpu,np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
                    edge=roi & ~torch.from_numpy(inner).cuda()
                    edge_mse=(rgb-reference)[:,edge].square().mean().item()
                    record.update(proxy_inner_edge_pixels=int(edge.sum()),proxy_inner_edge_MSE=edge_mse,
                        proxy_inner_edge_PSNR=-10*math.log10(edge_mse) if edge_mse else 999.,
                        proxy_inner_edge_alpha_coverage=(alpha[edge]>.5).float().mean().item())
                    save={k:package[k].detach().cpu() for k in ['render','rend_alpha','surf_depth']}
                    path=out/f'{key}.pt';torch.save(save,path)
                    view_manifest.append({'run_id':name,'camera':key,'float_render_path':str(path),'float_render_sha256':sha(path),
                        'bytes':path.stat().st_size,'model_sha256':run['output_model_sha256'],
                        'reference_tensor_sha256':tensor_sha(reference),'ROI_file_sha256':roi_rec['mask_sha256']})
                    if run['prune_ratio']=='0':
                        replay.append({'run_id':name,'camera':key,**{f'{k}_difference_from_E1':record[k]-float(original[key][k]) for k in ['PSNR','SSIM','LPIPS']}})
                    assert all(math.isfinite(record[k]) for k in ['PSNR','SSIM','LPIPS','foreground_PSNR','fixed_mask_composited_SSIM','fixed_mask_composited_LPIPS'])
                    allrows.append(record)
                    del save,package,raw,rgb,alpha,reference,roi,edge
            write_csv(STAGE/f'reports/{name}_PER_VIEW_METRICS.csv',[r for r in allrows if r['run_id']==name])
            print('QUALITY_DONE',name,round(time.perf_counter()-start,3),flush=True)
            del g;gc.collect();torch.cuda.empty_cache()
        write_csv(STAGE/'reports/HELDOUT_PER_VIEW_METRICS.csv',allrows)
        write_csv(STAGE/'configs/FLOAT_RENDER_INDEX.csv',view_manifest)
        write_csv(STAGE/'reports/UNPRUNED_E1_REPLAY_DELTA.csv',replay)
        weights=Path(lpips.__file__).parent/'weights/v0.1/vgg.pth'
        vgg=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
        result.update(status='PASS',completed_at=now(),runs=len(runrows),views_per_run=len(views),measurements=len(allrows),
            GPU_name=torch.cuda.get_device_name(),torch_version=torch.__version__,numpy_version=np.__version__,
            Pillow_version=PIL.__version__,lpips_version=metadata.version('lpips'),LPIPS_network='vgg',LPIPS_version='0.1',
            LPIPS_normalize=True,LPIPS_weights_sha256=sha(weights),VGG_weights_sha256=sha(vgg),
            PSNR_source_sha256=sha(GS/'utils/image_utils.py'),SSIM_source_sha256=sha(GS/'utils/loss_utils.py'),
            foreground_metric_source_sha256=sha(E2/'scripts/metric_core.py'),all_metrics_finite=True,
            ROI_is_independent_GT=False,recovery_steps=0,background=p['background'],active_SH_degree=p['SH_degree'])
    except BaseException as e:
        result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:write_json(STAGE/'reports/QUALITY_EVALUATION.json',result)

if __name__=='__main__':main()
