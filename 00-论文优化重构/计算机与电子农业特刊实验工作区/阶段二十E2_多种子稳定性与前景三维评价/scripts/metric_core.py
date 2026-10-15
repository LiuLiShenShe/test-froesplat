"""All metrics consume real floating tensors and one immutable R0 proxy ROI."""
from __future__ import annotations
import math
from pathlib import Path


def resize_reference(path,width,height):
    # Exact E1 channelwise Pillow default BICUBIC path, preserved intentionally.
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as image: array=np.array(image.convert('RGB'))
    channels=[np.array(Image.fromarray(array[...,channel]).resize((width,height))) for channel in range(3)]
    return torch.from_numpy(np.stack(channels)).float()/255.0


def load_roi(path,width,height):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as image:
        array=np.array(image.resize((width,height),Image.Resampling.NEAREST))
    assert array.ndim==2 and set(np.unique(array).tolist()) <= {0,255}
    return torch.from_numpy(array)>127


def foreground_metrics(rgb,reference,alpha,roi,ssim,perceptual):
    import torch
    assert roi.dtype==torch.bool and tuple(rgb.shape[1:])==tuple(roi.shape)
    count=int(roi.sum().item())
    if count==0:raise ValueError('Frozen foreground proxy ROI is empty; metric undefined')
    residual=(rgb-reference)[:,roi]
    mse=residual.square().mean().item()
    mask=roi.to(rgb.dtype).unsqueeze(0)
    coverage=(alpha[roi]>0.5).float().mean().item()
    return {'foreground_pixels':count,'foreground_channel_samples':3*count,
            'foreground_MSE':mse,'foreground_PSNR':-10*math.log10(mse) if mse>0 else float('inf'),
            'fixed_mask_composited_SSIM':ssim(rgb*mask,reference*mask).item(),
            'fixed_mask_composited_LPIPS':perceptual((rgb*mask)[None],(reference*mask)[None],normalize=True).item(),
            'foreground_absolute_color_error':residual.abs().mean().item(),
            'foreground_coverage_proxy':coverage,'missing_alpha_fraction':1.0-coverage,
            'ROI_is_independent_GT':False,'composited_metrics_are_proxy':True}


def background_geometry_metrics(rgb,alpha,roi,depth,normal,surface_normal):
    import cv2
    import numpy as np
    import torch
    outside=~roi; prediction=alpha>0.5
    mask=(outside&prediction).cpu().numpy().astype(np.uint8)
    components,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    sizes=stats[1:,cv2.CC_STAT_AREA] if components>1 else np.empty(0,dtype=np.int32)
    inside_union=(roi|prediction).sum().item()
    supported=roi&prediction&torch.isfinite(depth)&(depth>0)
    n_norm=torch.linalg.vector_norm(normal,dim=0); s_norm=torch.linalg.vector_norm(surface_normal,dim=0)
    normal_support=supported&torch.isfinite(normal).all(0)&torch.isfinite(surface_normal).all(0)&(n_norm>1e-6)&(s_norm>1e-6)
    cosine=((normal*surface_normal).sum(0)/(n_norm*s_norm).clamp_min(1e-12)).clamp(-1,1)
    depth_values=depth[supported]
    result={'outside_ROI_RGB_energy':rgb[:,outside].square().mean().item() if outside.any() else None,
            'outside_ROI_alpha_mean':alpha[outside].mean().item() if outside.any() else None,
            'outside_ROI_alpha_sum':alpha[outside].sum().item() if outside.any() else None,
            'missing_alpha_fraction':(~prediction)[roi].float().mean().item(),
            'proxy_silhouette_IoU':(roi&prediction).sum().item()/inside_union if inside_union else None,
            'outside_alpha_components_ge16px':int((sizes>=16).sum()),
            'outside_alpha_largest_component_pixels':int(sizes.max()) if len(sizes) else 0,
            'outside_alpha_pixels':int(mask.sum()),'depth_valid_ROI_pixels':int(supported.sum().item()),
            'depth_valid_ROI_fraction':supported.sum().item()/roi.sum().item(),
            'normal_valid_pixels':int(normal_support.sum().item()),
            'normal_consistency_cosine_proxy':cosine[normal_support].mean().item() if normal_support.any() else None,
            'normal_consistency_abs_cosine_proxy':cosine[normal_support].abs().mean().item() if normal_support.any() else None,
            'geometry_GT':'NONE','depth_units':'arbitrary inherited COLMAP units','components_are_confirmed_floaters':False}
    for label,q in [('p01',.01),('p05',.05),('p50',.5),('p95',.95),('p99',.99)]:
        result[f'depth_{label}']=torch.quantile(depth_values,q).item() if len(depth_values) else None
    return result
