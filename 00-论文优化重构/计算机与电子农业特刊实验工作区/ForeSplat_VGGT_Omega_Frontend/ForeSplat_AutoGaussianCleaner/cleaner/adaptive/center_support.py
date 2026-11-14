"""Strict foreground center-depth support, independent of a Gaussian's footprint."""
import numpy as np
import torch


def center_support(scene,model,config):
    xyz=model.get_xyz.detach();count=torch.zeros(len(xyz),device='cuda',dtype=torch.int16)
    for index,view in enumerate(scene.views):
        _,_,mask,depth,confidence=scene.get_view(index)
        if depth is None or confidence is None:continue
        valid=mask&np.isfinite(depth)&(depth>0)&np.isfinite(confidence)
        if not valid.any():continue
        unit=max(float(np.median(depth[valid])),1e-9)
        high=valid&(confidence>=np.quantile(confidence[valid],.2))
        rotation=torch.as_tensor(view['E'][:3,:3].T.copy(),dtype=xyz.dtype,device='cuda')
        translation=torch.as_tensor(view['E'][:3,3].copy(),dtype=xyz.dtype,device='cuda')
        intrinsic=torch.as_tensor(view['K'].T.copy(),dtype=xyz.dtype,device='cuda')
        position=xyz@rotation+translation;z=position[:,2];projection=position@intrinsic
        uv=torch.round(projection[:,:2]/z[:,None].clamp_min(1e-8)).long()
        inside=(z>0)&(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0])
        pixel=uv[:,1].clamp(0,mask.shape[0]-1)*mask.shape[1]+uv[:,0].clamp(0,mask.shape[1]-1)
        depth_tensor=torch.as_tensor(depth.ravel(),device='cuda')
        high_tensor=torch.as_tensor(high.ravel(),device='cuda')
        supported=inside&high_tensor[pixel]&((z-depth_tensor[pixel]).abs()/unit<=config.get('center_depth_relative_tolerance',.03))
        count+=supported.to(count.dtype)
        if (index+1)%32==0 or index+1==len(scene.views):print('center depth support',index+1,'/',len(scene.views),flush=True)
    return count.cpu().numpy()
