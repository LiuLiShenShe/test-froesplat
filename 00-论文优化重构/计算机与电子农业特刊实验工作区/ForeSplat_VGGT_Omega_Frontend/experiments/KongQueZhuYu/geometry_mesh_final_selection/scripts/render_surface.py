"""Actual native checkpoint rendering for every camera; no Gaussian centres."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['MPLBACKEND']='Agg'
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from surface_protocol import ROOT,DUAL,write,save_npz_fast

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['E8','E10','E11'],required=True);a=p.parse_args();exp=a.experiment
    out=ROOT/'surface_inputs'/f'{exp.lower()}_{"2dgs" if exp=="E11" else "gggs"}_surface';out.mkdir(parents=True,exist_ok=True)
    if (out/'render_complete.json').exists():return
    ds=DUAL/('VGGT_Omega_Original_RGB' if exp=='E8' else 'Standard_VGGT_Original_RGB')/'COLMAP_Corrected'
    source=ROOT/'E11_2DGS_Corrected/checkpoints' if exp=='E11' else DUAL/('E8_Omega_Corrected_Interface' if exp=='E8' else 'E10_Standard_Corrected_Interface')/'checkpoints'
    runtime=out/'runtime_checkpoint';linked=runtime/'point_cloud/iteration_30000';linked.mkdir(parents=True,exist_ok=True)
    checkpoint=source/'point_cloud/iteration_30000/point_cloud.ply'
    if not (linked/'point_cloud.ply').exists():(linked/'point_cloud.ply').symlink_to(checkpoint)
    if exp=='E11':
        repo=ROOT/'camera_interface/2dgs_corrected';sys.path.insert(0,str(repo))
        from arguments import ModelParams,PipelineParams
        parser=argparse.ArgumentParser();mp=ModelParams(parser);pp=PipelineParams(parser)
        args=parser.parse_args(['-s',str(ds),'-m',str(runtime),'-r','2','--sh_degree','3','--depth_ratio','0.0']);ns=mp.extract(args);ns.data_device='cpu';pipe=pp.extract(args)
        from scene import GaussianModel,Scene
        import gaussian_renderer as native
        model=GaussianModel(3)
    else:
        repo=DUAL/'camera_interface/gggs_corrected';sys.path.insert(0,str(repo))
        from utils.cuda_preload import preload_cuda_ext_deps
        preload_cuda_ext_deps()
        from scene import GaussianModel,Scene
        import gaussian_renderer as native
        ns=argparse.Namespace(sh_degree=3,sg_degree=0,source_path=str(ds),model_path=str(runtime),images='images',dataset='',resolution=2,white_background=False,data_device='cpu',eval=False,use_decoupled_appearance=0,disable_filter3D=False,kernel_size=0.,multi_view_num=8,multi_view_max_angle=30,multi_view_min_dis=.01,multi_view_max_dis=1.5)
        pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);model=GaussianModel(3,0)
    torch.set_num_threads(8);start=time.time();scene=Scene(ns,model,load_iteration=30000,shuffle=False);cams=scene.getTrainCameras();assert len(cams)==208
    refs=json.loads((DUAL/'render_comparison'/('E10' if exp=='E11' else exp)/'camera_runtime.json').read_text())['records'];refs={r['name']:r for r in refs}
    old={int(r['source_id']):r for r in csv.DictReader((DUAL/'render_comparison'/('E10' if exp=='E11' else exp)/'per_view_metrics.csv').open())}
    bg=torch.zeros(3,device='cuda');records=[];manifest=[];torch.cuda.reset_peak_memory_stats()
    for c in cams:
        sid=int(c.image_name.split('_')[-1]);r=refs[c.image_name];rec=dict(name=c.image_name,source_id=sid,width=c.image_width,height=c.image_height,Fx=c.Fx,Fy=c.Fy,Cx=c.Cx,Cy=c.Cy)
        assert (c.image_width,c.image_height)==(1080,1920)
        for k in ['Fx','Fy','Cx','Cy']:assert abs(rec[k]-r[k])<1e-8
        for k in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
            rec[k]=getattr(c,k).cpu().tolist();assert np.max(np.abs(np.asarray(rec[k])-r[k]))<1e-7
        records.append(rec);dest=out/'rendered'/f'{c.image_name}.npz';dest.parent.mkdir(exist_ok=True)
        gt=c.original_image[:3].cpu().numpy().transpose(1,2,0).astype(np.float32)
        assert hashlib.sha256(gt.tobytes()).hexdigest()==old[sid]['gt_sha256']
        fg=np.asarray(Image.open(ds/'masks'/f'{c.image_name}.png').convert('L').resize((c.image_width,c.image_height),Image.Resampling.NEAREST))>127
        assert hashlib.sha256(fg.tobytes()).hexdigest()==old[sid]['mask_sha256']
        if not dest.exists():
            with torch.no_grad():pkg=native.render(c,model,pipe,bg) if exp=='E11' else native.render(c,model,pipe,bg,0.)
            raw=pkg['render'].cpu().numpy().transpose(1,2,0).astype(np.float32)
            if exp=='E11':
                depth=pkg['surf_depth'].cpu().numpy().squeeze();alpha=pkg['rend_alpha'].cpu().numpy().squeeze();normal=pkg['rend_normal'].cpu().numpy().transpose(1,2,0)
                extra=dict(depth_median=pkg['depth_median'].cpu().numpy().squeeze(),depth_expected=pkg['depth_expected'].cpu().numpy().squeeze(),depth_normal_world=pkg['surf_normal'].cpu().numpy().transpose(1,2,0))
            else:
                depth=pkg['median_depth'].cpu().numpy().squeeze();alpha=pkg['mask'].cpu().numpy().squeeze();normal=pkg['normal'].cpu().numpy().transpose(1,2,0);extra={}
            save_npz_fast(dest,render_raw=raw,gt_rgb=gt,foreground_mask=fg,alpha=alpha,depth_surface=depth,native_normal=normal,**extra)
            del pkg
        manifest.append(dict(source_id=sid,path=str(dest),camera=rec,gt_sha256=old[sid]['gt_sha256'],mask_sha256=old[sid]['mask_sha256']))
        print(exp,sid,'native full-resolution surface outputs saved',flush=True)
    write(out/'camera_runtime.json',dict(records=records));write(out/'render_index.json',manifest)
    write(out/'render_complete.json',dict(status='complete',experiment=exp,views=208,checkpoint=str(checkpoint),dataset=str(ds),repo=str(repo),normal_coordinates='world' if exp=='E11' else 'camera',
        surface_depth='native expected depth with original depth_ratio=0' if exp=='E11' else 'native rendered median depth',model_count=len(model.get_xyz),wall_seconds=time.time()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,physical_gpu=1,
        renderer_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),all208_GT_and_Mask_hashes_match_E10_or_E8=True))
    if exp=='E11':
        e=ROOT/'E11_2DGS_Corrected'
        for name in ['renders','depth','normals','alpha']:
            (e/name).mkdir(exist_ok=True);write(e/name/'INDEX.json',dict(source=str(out/'render_index.json'),storage='All208 native float outputs in rendered NPZ files; no repeated large-file copies'))

if __name__=='__main__':main()
