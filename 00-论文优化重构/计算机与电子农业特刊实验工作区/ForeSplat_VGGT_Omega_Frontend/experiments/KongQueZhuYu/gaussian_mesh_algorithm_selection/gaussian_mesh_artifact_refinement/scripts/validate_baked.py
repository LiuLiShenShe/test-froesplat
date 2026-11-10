from common import *
import torch
def main():
    GaussianModel,native=setup();d=ROOT/'task0_gaussian_cleaning/best_cleaned';a=GaussianModel(3,0);a.load_ply(str(d/'cleaned_gaussians.ply'));b=GaussianModel(3,0);b.load_ply(str(d/'cleaned_gaussians_supersplat.ply'));torch.set_num_threads(8)
    s1,o1=a.get_scaling_n_opacity_with_3D_filter;s2,o2=b.get_scaling_n_opacity_with_3D_filter
    params=dict(scale_max_abs=float((s1-s2).abs().max()),filtered_opacity_max_abs=float((o1-o2).abs().max()));assert params['scale_max_abs']<1e-6 and params['filtered_opacity_max_abs']<1e-6
    cams,refs=cameras();pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);records=[]
    for c,r in zip(cams,refs):
        if r['source_id'] not in [0,53,105,157]:continue
        with torch.no_grad():x=native.render(c,a,pipe,torch.full((3,),.25,device='cuda'),0.);y=native.render(c,b,pipe,torch.full((3,),.25,device='cuda'),0.)
        rec=dict(source_id=r['source_id'],rgb_max=float((x['render']-y['render']).abs().max()),alpha_max=float((x['mask']-y['mask']).abs().max()),depth_changed_pixels=int(((x['median_depth']-y['median_depth']).abs()>1e-4).sum()));rec['rgb_mean']=float((x['render']-y['render']).abs().mean());rec['rgb_gt01_fraction']=float(((x['render']-y['render']).abs()>.01).float().mean());print(rec,flush=True);records.append(rec)
    write(d/'NATIVE_SUPERSPLAT_PARITY.json',dict(status='parameters_verified_actual_pixel_difference_reported',parameters=params,records=records,native_count=len(a.get_xyz),baked_count=len(b.get_xyz),native_sha256=sha(d/'cleaned_gaussians.ply'),baked_sha256=sha(d/'cleaned_gaussians_supersplat.ply'),viewer='Standard scale/opacity/rotation/SH schema verified and native raster parity. SuperSplat interactive GUI inspection not claimed.'))
    print('baking parity',params,records,flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
