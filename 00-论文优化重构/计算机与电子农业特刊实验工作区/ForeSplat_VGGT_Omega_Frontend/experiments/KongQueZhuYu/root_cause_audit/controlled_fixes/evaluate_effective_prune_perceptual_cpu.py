#!/usr/bin/env python3
"""CPU-only SSIM/LPIPS supplement; never download, render, train, or touch baseline."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import numpy as np
import torch
import torchvision.models._api as vision_api
from skimage.metrics import structural_similarity
import lpips

AUDIT=Path(__file__).resolve().parents[1]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=AUDIT/'controlled_fixes/prune_effective_render/perceptual_cpu_supplement')
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    # Replace the exact torchvision weight-loader binding with a local-only
    # loader. A missing cache raises rather than attempting network access.
    cache=Path('/home/test/.cache/torch/hub/checkpoints')
    backbone=cache/'alexnet-owt-7be5be79.pth'
    calibration=Path(lpips.__file__).resolve().parent/'weights/v0.1/alex.pth'
    for p in [backbone,calibration]:
        if not p.is_file():raise FileNotFoundError(f'Offline cached weight missing: {p}')
    loads=[]
    def local_state_dict(url,*unused,**kwargs):
        p=cache/Path(urlparse(url).path).name
        if p!=backbone:raise RuntimeError(f'Unexpected backbone weight requested in offline audit: {url}')
        loads.append(str(p))
        return torch.load(p,map_location='cpu',weights_only=True)
    vision_api.load_state_dict_from_url=local_state_dict
    began=time.perf_counter()
    model=lpips.LPIPS(net='alex',model_path=str(calibration),verbose=False).cpu().eval()
    if any(v.device.type!='cpu' for v in model.parameters()):raise RuntimeError('CPU-only metric device invariant failed')
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    baseline_csv=AUDIT/'render_error_analysis/common_evaluation_v1/per_view_common_metrics.csv'
    baseline={(r['experiment'],int(r['source_id'])):r for r in csv.DictReader(baseline_csv.open())}
    rows=[]
    guarded={}
    for mode in ['E1','E5']:
        comparison=AUDIT/'controlled_fixes/prune_effective_render'/mode/'evaluation/comparison.json'
        guarded[str(comparison)]=sha256(comparison)
        for source in [0,95,189]:
            p=AUDIT/'controlled_fixes/prune_effective_render'/mode/'views'/f'crop_{source:04d}'/'raw_render_outputs.npz'
            with np.load(p) as raw:
                rgb=raw['render_rgb'];gt=raw['gt_rgb']
            if rgb.shape!=(1920,1080,3) or gt.shape!=rgb.shape:raise ValueError(f'Unexpected native shape: {p}')
            start=time.perf_counter()
            ssim=float(structural_similarity(gt,rgb,channel_axis=2,data_range=1.,win_size=7,
                gaussian_weights=False,use_sample_covariance=True))
            with torch.no_grad():
                image=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to('cpu')*2-1
                target=torch.from_numpy(gt.copy()).permute(2,0,1)[None].to('cpu')*2-1
                perceptual=float(model(image,target).item())
            old=baseline[(mode,source)]
            row=dict(experiment=mode,source_id=source,width=1080,height=1920,metric_device='cpu',
                ssim_baseline_reused=float(old['ssim']),ssim_effective_pruned=ssim,
                ssim_delta=ssim-float(old['ssim']),lpips_alex_baseline_reused=float(old['lpips_alex']),
                lpips_alex_effective_pruned=perceptual,lpips_alex_delta=perceptual-float(old['lpips_alex']),
                seconds_ssim_and_lpips=time.perf_counter()-start,
                raw_render_npz_sha256=sha256(p),GT_sha256=hashlib.sha256(gt.tobytes()).hexdigest())
            rows.append(row)
            print(f'{mode} source{source} CPU SSIM{ssim:.7f} LPIPS{perceptual:.7f} seconds{row["seconds_ssim_and_lpips"]:.3f}',flush=True)
    summaries=[]
    for mode in ['E1','E5']:
        subset=[r for r in rows if r['experiment']==mode];summary=dict(experiment=mode,views=3)
        for key in ['ssim_baseline_reused','ssim_effective_pruned','ssim_delta',
                    'lpips_alex_baseline_reused','lpips_alex_effective_pruned','lpips_alex_delta']:
            summary[key]=float(np.mean([r[key] for r in subset]))
        summaries.append(summary)
    with (args.output/'per_view_perceptual_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (args.output/'summary_perceptual_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    for p,digest in guarded.items():
        if sha256(p)!=digest:raise RuntimeError(f'Original comparison overwritten: {p}')
    report=dict(status='complete',scope='6 recorded effective-pruned float RGB views, no render or training',
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],metric_device='cpu',threads=args.threads,
        rendered_views=0,new_training=False,network_weight_downloads=False,local_backbone_loads=loads,
        backbone=dict(path=str(backbone),sha256=sha256(backbone)),
        LPIPS_calibration=dict(path=str(calibration),sha256=sha256(calibration)),
        versions={name:importlib.metadata.version(name)for name in ['torch','torchvision','lpips','scikit-image']},
        protocol=dict(RGB='saved float32 clamped native renderer; unchanged GT;1080x1920;no PNG quantization',
            SSIM='channel_axis2,data_range1,win7,gaussianFalse,sampleCovTrue, same scikit-image implementation as main12',
            LPIPS='same Alex backbone/calibration; full resolution [-1,1]; one CPU instance; no resizing',
            baseline='same source0/95/189 rows reused from original GPU main12 CSV; never recomputed',
            numeric_limit='LPIPS baseline ran GPU, supplement ran CPU; small device rounding differences are possible, so no bit-identical inference claim'),
        baseline_csv=dict(path=str(baseline_csv),sha256=sha256(baseline_csv)),
        original_comparison_hashes_unchanged=guarded,summaries=summaries,
        wall_seconds=time.perf_counter()-began)
    (args.output/'perceptual_supplement.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
