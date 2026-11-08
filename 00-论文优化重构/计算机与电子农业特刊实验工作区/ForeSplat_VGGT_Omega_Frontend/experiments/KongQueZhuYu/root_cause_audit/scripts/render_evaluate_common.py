#!/usr/bin/env python3
"""One metric implementation for actual E1/E5/E6 checkpoint-render float data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity

AUDIT=Path(__file__).resolve().parents[1]
SOURCES=[0,19,38,57,76,95,114,133,152,170,189,209]


def psnr_region(error, selection):
    mse=float(np.mean(error[selection],dtype=np.float64))
    return None if mse==0 else float(-10*np.log10(mse))


def gradients(image):
    gray=image@np.asarray([.2126,.7152,.0722],dtype=np.float32)
    gx=ndimage.sobel(gray,axis=1,mode='reflect')/8
    gy=ndimage.sobel(gray,axis=0,mode='reflect')/8
    return np.hypot(gx,gy)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-root',type=Path,default=AUDIT/'render_error_analysis/checkpoint_rerender_v1')
    parser.add_argument('--output',type=Path,default=AUDIT/'render_error_analysis/common_evaluation_v1')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError(f'Preserve existing metrics: {args.output}')
    args.output.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    import lpips,skimage
    model=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
    torch.set_num_threads(8)
    rows=[];parity=[];depth_values=[];crop_records=[]
    for source in SOURCES:
        inputs={}
        for experiment in ('E1','E5','E6'):
            directory=args.render_root/experiment/'views'/f'crop_{source:04d}'
            with np.load(directory/'raw_render_outputs.npz') as raw:
                inputs[experiment]={key:raw[key] for key in ('render_rgb','gt_rgb','foreground_mask','alpha','depth_median','normal_camera')}
        reference=inputs['E1']
        assert all(np.array_equal(reference['gt_rgb'],v['gt_rgb']) for v in inputs.values()),f'GT mismatch at source{source}'
        assert all(np.array_equal(reference['foreground_mask'],v['foreground_mask']) for v in inputs.values()),f'Mask mismatch at source{source}'
        parity.append(dict(source_id=source,gt_identical=True,mask_identical=True,shape=list(reference['gt_rgb'].shape),
                           gt_sha256=hashlib.sha256(reference['gt_rgb'].tobytes()).hexdigest()))
        gt=reference['gt_rgb'];fg=reference['foreground_mask'];shape=fg.shape
        boundary=ndimage.binary_dilation(fg,iterations=5)^ndimage.binary_erosion(fg,iterations=5)
        inner=fg&boundary;all_pixels=np.ones(shape,dtype=bool)
        valid_window=all_pixels.copy();valid_window[:3]=False;valid_window[-3:]=False;valid_window[:,:3]=False;valid_window[:,-3:]=False
        yy,xx=np.nonzero(fg);box=np.zeros(shape,dtype=bool);box[yy.min():yy.max()+1,xx.min():xx.max()+1]=True
        gt_gradient=gradients(gt)
        error_panels=[];normal_panels=[];alpha_panels=[];rgb_panels=[gt]
        for experiment,value in inputs.items():
            rendered=value['render_rgb'];error=(rendered-gt)**2
            score,ssim_map=structural_similarity(gt,rendered,channel_axis=2,data_range=1.,win_size=7,
                gaussian_weights=False,use_sample_covariance=True,full=True)
            ssim_map=ssim_map.mean(axis=2)
            with torch.no_grad():
                ta=torch.from_numpy(rendered.copy()).permute(2,0,1)[None].cuda()*2-1
                tb=torch.from_numpy(gt.copy()).permute(2,0,1)[None].cuda()*2-1
                perceptual=float(model(ta,tb).item())
            rendered_gradient=gradients(rendered)
            alpha50=value['alpha']>.5
            alpha_true_positive=int((alpha50&fg).sum())
            alpha_union=int((alpha50|fg).sum())
            row=dict(experiment=experiment,source_id=source,width=shape[1],height=shape[0],
                full_psnr=psnr_region(error,all_pixels),fg_mask_psnr=psnr_region(error,fg),
                boundary_band5_psnr=psnr_region(error,boundary),inner_fg_edge5_psnr=psnr_region(error,inner),
                legacy_fg_bbox_psnr=psnr_region(error,box),ssim=float(score),
                fg_mask_ssim_map=float(ssim_map[fg&valid_window].mean()),
                boundary_ssim_map=float(ssim_map[boundary&valid_window].mean()),lpips_alex=perceptual,
                foreground_pixels=int(fg.sum()),boundary_pixels=int(boundary.sum()),bbox_pixels=int(box.sum()),
                gt_fg_gradient_mean=float(gt_gradient[fg].mean()),render_fg_gradient_mean=float(rendered_gradient[fg].mean()),
                fg_gradient_retention=float(rendered_gradient[fg].mean()/gt_gradient[fg].mean()),
                gt_edge_gradient_mean=float(gt_gradient[inner].mean()),render_edge_gradient_mean=float(rendered_gradient[inner].mean()),
                edge_gradient_retention=float(rendered_gradient[inner].mean()/gt_gradient[inner].mean()),
                fg_gradient_mae=float(np.abs(rendered_gradient-gt_gradient)[fg].mean()),
                foreground_alpha_mean=float(value['alpha'][fg].mean()),background_alpha_mean=float(value['alpha'][~fg].mean()),
                background_alpha_gt01_fraction=float((value['alpha'][~fg]>.1).mean()),
                background_alpha_gt05_fraction=float((value['alpha'][~fg]>.5).mean()),
                background_alpha_gt09_fraction=float((value['alpha'][~fg]>.9).mean()),
                alpha_mask05_iou=alpha_true_positive/alpha_union,
                alpha_mask05_precision=alpha_true_positive/max(1,int(alpha50.sum())),
                alpha_mask05_recall=alpha_true_positive/int(fg.sum()),
                background_rgb_mean_abs_error=float(np.abs(rendered-gt)[~fg].mean()))
            rows.append(row)
            directory=args.output/experiment/f'crop_{source:04d}';directory.mkdir(parents=True)
            absolute=np.abs(rendered-gt).mean(axis=2)
            np.savez_compressed(directory/'error_regions.npz',absolute_rgb_mean=absolute,
                foreground_error=absolute*fg,boundary_error=absolute*boundary,
                squared_rgb_error=error,foreground_mask=fg,boundary_band5=boundary,
                inner_foreground_edge5=inner,gt_gradient=gt_gradient,render_gradient=rendered_gradient)
            for name,err in [('absolute_error',absolute),('foreground_error',absolute*fg),('boundary_error',absolute*boundary)]:
                heat=np.clip(err/.5,0,1);image=np.stack([heat,np.zeros_like(heat),1-heat],axis=2)
                if name!='absolute_error':image[err==0]=0
                Image.fromarray(np.round(image*255).astype(np.uint8)).save(directory/f'{name}.png')
            normal=value['normal_camera'];length=np.linalg.norm(normal,axis=2,keepdims=True)
            normal=np.divide(normal,length,out=np.zeros_like(normal),where=length>1e-8)
            normal_image=np.clip(normal*.5+.5,0,1);normal_image[value['alpha']<=.01]=0
            Image.fromarray(np.round(normal_image*255).astype(np.uint8)).save(directory/'normal_camera.png')
            rgb_panels.append(rendered);error_panels.append(np.repeat((np.clip(absolute/.5,0,1)*boundary)[...,None],3,axis=2));normal_panels.append(normal_image);alpha_panels.append(np.repeat(np.clip(value['alpha'],0,1)[...,None],3,axis=2))
            sampled=value['depth_median'][fg&(value['alpha']>.5)&np.isfinite(value['depth_median'])&(value['depth_median']>0)]
            depth_values.append(sampled[::max(1,len(sampled)//10000)])
            print(f'{experiment} source{source}: PSNR{row["full_psnr"]:.3f} trueFG{row["fg_mask_psnr"]:.3f} boundary{row["boundary_band5_psnr"]:.3f}',flush=True)
        def montage(panels,name):
            pictures=[Image.fromarray(np.round(np.clip(p,0,1)*255).astype(np.uint8)).resize((270,480),Image.Resampling.LANCZOS) for p in panels]
            canvas=Image.new('RGB',(270*len(pictures),480));[canvas.paste(image,(i*270,0)) for i,image in enumerate(pictures)];canvas.save(args.output/f'source_{source:04d}_{name}.png')
        montage(rgb_panels,'GT_E1_E5_E6');montage(error_panels,'E1_E5_E6_boundary_error');montage(normal_panels,'E1_E5_E6_camera_normals');montage(alpha_panels,'E1_E5_E6_alpha')
        # A common GT-selected upper-plant boundary patch, never a model-selected crop.
        candidate=inner.copy();candidate[yy.min()+int((yy.max()-yy.min())*.60):]=False
        energy=ndimage.uniform_filter(gt_gradient,size=15)*candidate
        cy,cx=np.unravel_index(np.argmax(energy),shape)
        side=320;x0=max(0,min(int(cx)-side//2,shape[1]-side));y0=max(0,min(int(cy)-side//2,shape[0]-side))
        crop_records.append(dict(source_id=source,x0=x0,y0=y0,x1=x0+side,y1=y0+side,
            selection='GT foreground inner 5px boundary in upper60% of foreground bbox; strongest15px-mean GT Sobel',
            interpretation='leaf-edge candidate crop, no leaf instance annotation'))
        for panels,name in ((rgb_panels,'GT_E1_E5_E6_leafedge_native320'),(error_panels,'E1_E5_E6_leafedge_error_native320')):
            canvas=Image.new('RGB',(side*len(panels),side))
            for i,panel in enumerate(panels):
                picture=Image.fromarray(np.round(np.clip(panel[y0:y0+side,x0:x0+side],0,1)*255).astype(np.uint8));canvas.paste(picture,(side*i,0))
            canvas.save(args.output/f'source_{source:04d}_{name}.png')
    with (args.output/'per_view_common_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    for experiment in ('E1','E5','E6'):
        subset=[r for r in rows if r['experiment']==experiment]
        summary=dict(experiment=experiment,views=len(subset))
        for key in rows[0]:
            if key not in ('experiment','source_id','width','height'):summary[key]=float(np.mean([r[key] for r in subset if r[key] is not None]))
        summaries.append(summary)
    with (args.output/'summary_common_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    omega=AUDIT.parents[2]
    standard=omega.parent/'ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison'
    historical_paths={'E1':omega/'experiments/KongQueZhuYu/E1_GGGS_Baseline/metrics',
        'E5':standard/'E5_Standard_VGGT_GGGS_Baseline/metrics',
        'E6':standard/'E6_2DGS_Standard_VGGT/metrics'}
    historical=[];old_reports={}
    for experiment,directory in historical_paths.items():
        with (directory/'per_view_metrics.csv').open() as stream:
            old={int(Path(r['image_name']).stem.split('_')[-1]):r for r in csv.DictReader(stream)}
        old_reports[experiment]=dict(path=str(directory/'render_metrics.json'),
            file_sha256=hashlib.sha256((directory/'render_metrics.json').read_bytes()).hexdigest(),
            report=json.loads((directory/'render_metrics.json').read_text()))
        for row in (r for r in rows if r['experiment']==experiment):
            previous=old[row['source_id']]
            compared=dict(experiment=experiment,source_id=row['source_id'])
            for new_key,old_key in [('full_psnr','psnr'),('ssim','ssim'),('lpips_alex','lpips'),('legacy_fg_bbox_psnr','fg_psnr')]:
                compared[new_key+'_historical_saved']=float(previous[old_key])
                compared[new_key+'_checkpoint_rerender']=row[new_key]
                compared[new_key+'_difference']=row[new_key]-float(previous[old_key])
            historical.append(compared)
    with (args.output/'historical_same12_vs_checkpoint_rerender.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(historical[0]));writer.writeheader();writer.writerows(historical)
    (args.output/'historical_all208_metrics_snapshot.json').write_text(json.dumps(old_reports,indent=2,allow_nan=False))
    settings=dict(scope='12 representative training views, actual checkpoint re-render, float32 RGB native black-background composition',
        source_ids=SOURCES,ssim='same scikit-image function for all: channel_axis2, win7, gaussianFalse, sampleCovTrue, data_range1; regional SSIM is mean of full map over selected valid pixels',
        foreground='exact external binary mask, nearest interpolation; old bbox PSNR saved separately',boundary='5 actual-render-pixel dilation XOR erosion, inner boundary reported separately; evaluation only',
        gradients='Rec709 RGB grayscale, reflect-mode Sobel /8; energy retention can also reflect noise so gradient MAE saved',
        lpips='same cached AlexNet LPIPS full native resolution [-1,1] for all; no input resize',
        background=[0,0,0],metric_png_requantization=False,skimage_version=skimage.__version__,lpips_version=getattr(lpips,'__version__',None),
        summaries=summaries,input_parity=parity)
    settings['leafedge_crop_records']=crop_records
    settings['historical_comparison_scope']='Previous saved per-view float metrics on the same12 source IDs, separate from old all208 aggregate; previous fixed-view PNGs are not this new evidence.'
    settings['historical_same12_max_abs_difference']={experiment:{key:float(max(abs(r[key+'_difference']) for r in historical if r['experiment']==experiment)) for key in ('full_psnr','ssim','lpips_alex','legacy_fg_bbox_psnr')} for experiment in ('E1','E5','E6')}
    settings['physical_gpu']=1
    settings['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    settings['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    settings['wall_seconds_including_cpu_metrics_visualizations']=time.perf_counter()-begin
    (args.output/'common_metric_settings.json').write_text(json.dumps(settings,indent=2,allow_nan=False))
    # Numeric depth mapping pooled across methods; arbitrary scene scales are not metric distances.
    pooled=np.concatenate(depth_values);lo,hi=np.quantile(pooled,[.02,.98]);settings['shared_depth_numeric_range']=[float(lo),float(hi)]
    import matplotlib
    cmap=matplotlib.colormaps['turbo']
    for source in SOURCES:
        images=[]
        for experiment in ('E1','E5','E6'):
            with np.load(args.render_root/experiment/'views'/f'crop_{source:04d}'/'raw_render_outputs.npz') as raw:
                depth=raw['depth_median'];alpha=raw['alpha'];colored=cmap(np.clip((depth-lo)/(hi-lo),0,1))[...,:3];colored[alpha<=.01]=0
                image=Image.fromarray(np.round(colored*255).astype(np.uint8));image.save(args.output/experiment/f'crop_{source:04d}'/'depth_median.png');images.append(image.resize((270,480),Image.Resampling.LANCZOS))
        canvas=Image.new('RGB',(810,480));[canvas.paste(image,(i*270,0)) for i,image in enumerate(images)];canvas.save(args.output/f'source_{source:04d}_E1_E5_E6_median_depth.png')
    settings['wall_seconds_full_evaluation_and_all_visualizations']=time.perf_counter()-begin
    settings['visualization_complete']=True
    (args.output/'common_metric_settings.json').write_text(json.dumps(settings,indent=2,allow_nan=False))
    print('COMPLETE common actual-checkpoint metrics',flush=True)


if __name__=='__main__':main()
