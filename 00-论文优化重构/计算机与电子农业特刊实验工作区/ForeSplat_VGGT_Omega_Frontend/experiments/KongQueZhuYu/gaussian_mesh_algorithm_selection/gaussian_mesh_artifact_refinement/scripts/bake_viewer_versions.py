from common import *
from plyfile import PlyData,PlyElement

def main():
    d=ROOT/'task0_gaussian_cleaning';versions=[('E10',SOURCE,d/'diagnosis/E10_gaussians_supersplat.ply'),('P2',P2,d/'diagnosis/P2_gaussians_supersplat.ply')]
    for name in ['round_01','round_02','round_03','additional_rounds/aggressive_delete','additional_rounds/round_04','additional_rounds/round_05']:
        folder=d/name;versions.append((name,folder/'gaussians.ply',folder/'gaussians_supersplat.ply'))
    versions.append(('final_removed',d/'best_cleaned/removed_gaussians.ply',d/'best_cleaned/removed_gaussians_supersplat.ply'))
    records=[]
    for name,source,out in versions:
        v=PlyData.read(str(source))['vertex'].data.copy();sigma=np.exp(np.column_stack([v[f'scale_{k}'] for k in range(3)]).astype(np.float64));filt=v['filter_3D'].astype(np.float64)[:,None];filtered=np.hypot(sigma,filt);op=1/(1+np.exp(-v['opacity'].astype(np.float64)));newop=np.clip(op*(sigma/filtered).prod(1),1e-12,1-1e-7)
        for k in range(3):v[f'scale_{k}']=np.log(filtered[:,k])
        v['opacity']=np.log(newop/(1-newop));v['filter_3D']=0
        if not out.exists():PlyData([PlyElement.describe(v,'vertex')],text=False,byte_order='<').write(str(out))
        effective=np.exp(np.column_stack([v[f'scale_{k}'] for k in range(3)]).astype(np.float64));diff=float(np.abs(effective-filtered).max());assert diff<1e-5
        records.append(dict(version=name,native_ply=str(source),viewer_ply=str(out),count=len(v),native_sha256=sha(source),viewer_sha256=sha(out),effective_scale_max_abs_rounding=diff,rotation_XYZ_SH_unchanged=True,formula='sigma_eff=sqrt(sigma^2+filter^2); alpha_eff=sigmoid(opacity)*prod(sigma/sigma_eff); zero filter after baking'))
        print('baked viewer',name,len(v),flush=True)
    write(d/'VIEWER_BAKING_MANIFEST.json',dict(records=records,status='Native parameter-equivalent standard viewer exports; final four-camera 8bit parity in best_cleaned/NATIVE_SUPERSPLAT_PARITY.json'))
if __name__=='__main__':main()
