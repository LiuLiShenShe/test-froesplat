"""Same held-out cameras and foreground masks for every newly trained model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import io
import json
import sys
import time
from pathlib import Path
from runtime import DATA, OUT, SOURCES, Meter, install_io, write
import numpy as np
import torch
from PIL import Image


def load_native(method,override_ply=None):
    sys.path.insert(0,str(SOURCES[method]))
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams
    parser=argparse.ArgumentParser();lp,pp=ModelParams(parser),PipelineParams(parser)
    config=json.loads((OUT/'gaussian'/method/'training_configuration.json').read_text())
    # Optimization flags belong to the official OptimizationParams group.
    from arguments import OptimizationParams
    OptimizationParams(parser)
    args=parser.parse_args(config['cli'])
    model=GaussianModel(3,0) if method=='GGGS' else GaussianModel(3)
    scene=Scene(lp.extract(args),model,load_iteration=30000,shuffle=False)
    if override_ply:model.load_ply(str(override_ply))
    pipe=pp.extract(args);background=torch.zeros(3,device='cuda')
    def call(camera):
        result=render(camera,model,pipe,background,args.kernel_size) if method=='GGGS' else render(camera,model,pipe,background)
        if method=='GGGS':alpha,depth,normal=result['mask'],result['median_depth'],result['normal']
        elif method=='2DGS':alpha,depth,normal=result['rend_alpha'],result['surf_depth'],result['rend_normal']
        else:
            ones=torch.ones_like(model.get_xyz)
            alpha=render(camera,model,pipe,background,override_color=ones)['render'][0:1]
            matrix=camera.world_view_transform
            z=model.get_xyz@matrix[:3,2]+matrix[3,2]
            depth=render(camera,model,pipe,background,override_color=z[:,None].expand(-1,3).contiguous())['render'][0:1]/alpha.clamp_min(1e-8)
            normal=None
        return result['render'],alpha,depth,normal
    call.rgb_only=lambda camera:(render(camera,model,pipe,background,args.kernel_size,require_depth=False) if method=='GGGS' else render(camera,model,pipe,background))['render']
    return scene,call


def load_gsplat(cleaned=False):
    sys.path.insert(0,str(SOURCES['3DGS']))
    from scene import Scene,GaussianModel
    from arguments import ModelParams
    p=argparse.ArgumentParser();lp=ModelParams(p)
    args=p.parse_args(['-s',str(DATA),'-m',str(OUT/'gaussian/gsplat'),'-r','1','--eval'])
    # Camera construction only; an existing output avoids duplicate initial PLY.
    scene=Scene(lp.extract(args),GaussianModel(3),shuffle=False)
    from gsplat import rasterization
    saved=torch.load(OUT/'gaussian/gsplat/ckpts/ckpt_29999_rank0.pt',map_location='cuda',weights_only=False)['splats']
    if cleaned:
        decision=json.loads((OUT/'cleaning/gsplat/complete.json').read_text())
        assert len(saved['means'])==decision['input_gaussians'],'Checkpoint/PLY row identity required'
        indices=torch.from_numpy(np.load(OUT/'cleaning/gsplat/retained_indices.npy')).cuda()
        saved={key:value[indices] for key,value in saved.items()}
    K=torch.tensor(json.loads((OUT/'common_input/colmap_source.json').read_text())['K'],device='cuda',dtype=torch.float32)
    def rasterize(camera,mode):
        color,alpha,_=rasterization(means=saved['means'],quats=saved['quats'],scales=saved['scales'].exp(),
            opacities=saved['opacities'].sigmoid(),colors=torch.cat([saved['sh0'],saved['shN']],1),
            viewmats=camera.world_view_transform.T[None],Ks=K[None],width=camera.image_width,height=camera.image_height,
            sh_degree=3,render_mode=mode,packed=False)
        return color,alpha
    def call(camera):
        color,alpha=rasterize(camera,'RGB+ED')
        return color[0,...,:3].permute(2,0,1),alpha[0].permute(2,0,1),color[0,...,3:4].permute(2,0,1),None
    call.rgb_only=lambda camera:rasterize(camera,'RGB')[0]
    return scene,call


def save_image(path,array,meter):
    buffer=io.BytesIO();Image.fromarray((np.clip(array,0,1)*255).astype(np.uint8)).save(buffer,format='PNG')
    meter.save(path,buffer.getbuffer())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('method');parser.add_argument('--surface',action='store_true')
    parser.add_argument('--cleaned',action='store_true')
    args=parser.parse_args();method=args.method
    assert (OUT/'common_input/rap_masks_complete.json').exists(),'RAP-FSAM3 evaluation masks are not ready; no SAM fallback'
    base=OUT/('cleaning' if args.cleaned else 'gaussian')/method
    output=base/'evaluation';output.mkdir(exist_ok=True)
    meter=Meter();install_io(meter);torch.set_num_threads(8)
    override=OUT/'cleaning'/method/'cleaned_gaussians.ply' if args.cleaned else None
    scene,render=load_gsplat(args.cleaned) if method=='gsplat' else load_native(method,override)
    # Every method is scored with the exact same baseline implementation,
    # independent of native training loss modules and their tensor signatures.
    import importlib.util
    loss_spec=importlib.util.spec_from_file_location('common_baseline_loss',SOURCES['3DGS']/'utils/loss_utils.py')
    loss_module=importlib.util.module_from_spec(loss_spec);loss_spec.loader.exec_module(loss_module)
    ssim=loss_module.ssim
    import lpips
    metric=lpips.LPIPS(net='alex').cuda().eval()
    references=OUT.parents[2]/'experiments/KongQueZhuYu/geometry_mesh_final_selection/scripts'
    sys.path.insert(0,str(references))
    from surface_protocol import normals_from_grid
    from evaluate_meshes import boundary_scores
    from scipy import ndimage
    K=np.asarray(json.loads((OUT/'common_input/colmap_source.json').read_text())['K'])
    representative={'0000','0048','0096','0144','0192','0208'}
    training_names={Path(n).stem for n in (OUT/'common_input/train_views.txt').read_text().splitlines()}
    views=scene.getTestCameras()
    if args.surface:views=sorted(scene.getTrainCameras()+views,key=lambda c:c.image_name)
    records=[];scale_samples=[];compute_s=0
    surface_key=method+('_Cleaned' if args.cleaned else '')
    for camera in views:
        name=Path(camera.image_name).stem
        mask=np.array(Image.open(DATA/'rap_masks'/f'{name}.png').convert('L'))>127
        with torch.no_grad():render.rgb_only(camera)
        torch.cuda.synchronize();rgb_start=time.perf_counter()
        with torch.no_grad():render.rgb_only(camera)
        torch.cuda.synchronize();rgb_elapsed=time.perf_counter()-rgb_start
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():rgb,alpha,depth,normal=render(camera)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;compute_s+=elapsed
        image=rgb.clamp(0,1);target=camera.original_image.cuda()[:3]
        a=alpha.detach().cpu().numpy().squeeze();z=depth.detach().cpu().numpy().squeeze()
        valid=np.isfinite(z)&(z>0)&(a>=.5)
        yy,xx=np.mgrid[:z.shape[0],:z.shape[1]]
        camera_xyz=np.stack([(xx+.5-K[0,2])*z/K[0,0],(yy+.5-K[1,2])*z/K[1,1],z],-1)
        c2w=np.linalg.inv(camera.world_view_transform.detach().cpu().numpy().T)
        xyz=camera_xyz@c2w[:3,:3].T+c2w[:3,3]
        derived,normal_valid=normals_from_grid(xyz,z,valid&mask,c2w[:3,3])
        array=image.detach().cpu().numpy().transpose(1,2,0)
        if args.surface:
            selected=valid&mask&normal_valid
            values=xyz[selected]
            if len(values) and name in training_names:scale_samples.append(values[np.linspace(0,len(values)-1,min(len(values),4096),dtype=int)].astype(np.float32))
            frames=OUT/'mesh'/f'{surface_key}_surface/frames';frames.mkdir(parents=True,exist_ok=True)
            buffer=io.BytesIO();np.savez_compressed(buffer,depth=np.where(valid&mask,z,0).astype(np.float32),
                rgb=(array*255).astype(np.uint8),alpha=a.astype(np.float32),K=K,c2w=c2w)
            meter.save(frames/f'{name}.npz',buffer.getbuffer())
        if camera in scene.getTestCameras():
            mse=(image-target).square();foreground=torch.from_numpy(mask).cuda()
            fg_mse=float(mse[:,foreground].mean());all_mse=float(mse.mean())
            with torch.no_grad():perceptual=float(metric(image[None]*2-1,target[None]*2-1).squeeze())
            record=dict(name=name,psnr=-10*np.log10(max(all_mse,1e-12)),foreground_psnr=-10*np.log10(max(fg_mse,1e-12)),
                ssim=float(ssim(image,target)),lpips=perceptual,alpha_iou=float(((a>=.5)&mask).sum()/max(1,((a>=.5)|mask).sum())),
                background_alpha=float(a[~mask].mean()),foreground_alpha_recall=float(((a>=.5)&mask).sum()/max(1,mask.sum())),
                dark_background_alpha_fraction=float(((a>=.5)&(~mask)&(array.max(-1)<.06)).sum()/max(1,(~mask).sum())),
                valid_foreground_depth_fraction=float((valid&mask).sum()/max(1,mask.sum())),render_s=elapsed,rgb_render_s=rgb_elapsed)
            record.update(boundary_scores(a>=.5,mask))
            reference_rgb=target.detach().cpu().numpy().transpose(1,2,0)
            dark_mismatch=(a>=.5)&(array.max(-1)<.06)&(reference_rgb.max(-1)>.15)
            record['darkened_background_fraction']=float((dark_mismatch&~mask).sum()/max(1,(~mask).sum()))
            record['darkened_foreground_fraction']=float((dark_mismatch&mask).sum()/max(1,mask.sum()))
            thin=mask&(ndimage.distance_transform_edt(mask)<=2)
            record['thin_mask_band_alpha_recall']=float(((a>=.5)&thin).sum()/max(1,thin.sum()))
            neighbour=(valid&mask)[:,:-1]&(valid&mask)[:,1:]
            delta=np.abs(z[:,1:]-z[:,:-1])/np.maximum((z[:,1:]+z[:,:-1])/2,1e-6)
            record['foreground_relative_depth_jump_fraction_5pct']=float((delta[neighbour]>.05).mean()) if neighbour.any() else float('nan')
            nvalid=normal_valid[:,:-1]&normal_valid[:,1:]
            cosine=np.sum(derived[:,:-1]*derived[:,1:],axis=-1)
            record['adjacent_normal_cosine']=float(cosine[nvalid].mean()) if nvalid.any() else float('nan')
            records.append(record)
            if name in representative:
                save_image(output/f'{name}_RGB.png',array,meter);save_image(output/f'{name}_Alpha.png',a,meter)
                save_image(output/f'{name}_Original.png',target.detach().cpu().numpy().transpose(1,2,0),meter)
                save_image(output/f'{name}_Mask.png',mask,meter)
                finite=z[valid&mask]
                lo,hi=np.quantile(finite,[.02,.98]) if len(finite) else (0,1)
                import matplotlib
                matplotlib.use('Agg')
                from matplotlib import colormaps
                dcolor=colormaps['viridis'](np.clip((z-lo)/max(hi-lo,1e-6),0,1))[...,:3];dcolor[~valid]=0
                save_image(output/f'{name}_Depth.png',dcolor,meter)
                save_image(output/f'{name}_Normal_derived.png',np.where(normal_valid[...,None],(derived+1)/2,0),meter)
                if normal is not None:
                    n=normal.detach().cpu().numpy().transpose(1,2,0)
                    save_image(output/f'{name}_Normal_native.png',(n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-6)+1)/2,meter)
            buffer=io.BytesIO();np.savez_compressed(buffer,depth_camera_z=z,alpha=a,valid=valid,K=K,c2w=c2w)
            meter.save(output/f'{name}_geometry.npz',buffer.getbuffer())
        print('RENDER',method,name,flush=True)
    write(output/'per_view.json',records)
    write(output/'metrics.json',dict(method=method,views=len(records),metrics={key:float(np.mean([r[key] for r in records])) for key in records[0] if key!='name'},
        geometry='All exported depth is camera-Z. 3DGS/MCMC use alpha-normalized native compositor weights over Gaussian centre camera-Z; 2DGS native expected surfel depth; GGGS native median depth; gsplat expected depth.',
        depth_comparison_limit='Depth estimators differ even after unit/frame normalization; no uncalibrated cross-method depth-error ranking.',
        normal='Normal_derived uses the same world-point finite differences and view orientation for every method; native maps retained separately.',
        render_scope='render_s: RGB plus alpha/depth attribute passes, synchronized per camera; rgb_render_s: one warmed native RGB call, GGGS require_depth=False and gsplat RGB mode; excludes serialization and LPIPS',io=meter.stats()))
    if args.surface:
        from runtime import CLEANER
        sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import StreamingObservations
        low,high=np.quantile(np.concatenate(scale_samples),[.005,.995],axis=0);diagonal=float(np.linalg.norm(high-low))
        scale=dict(diagonal=diagonal,origin=((low+high)/2).tolist(),sampling_voxel=diagonal/1024,nksr_voxel=diagonal/512,
            robust_bbox_min=low.tolist(),robust_bbox_max=high.tolist())
        observations=StreamingObservations(scale)
        surface=OUT/'mesh'/f'{surface_key}_surface'
        # Second pass retains every valid pixel before the existing exact voxel
        # and normal-direction sampler; no image stride can erase thin stems.
        for frame in sorted((surface/'frames').glob('*.npz')):
            if frame.stem not in training_names:continue
            with np.load(io.BytesIO(meter.read(frame))) as d:z,c2w,array=d['depth'],d['c2w'],d['rgb']
            yy,xx=np.mgrid[:z.shape[0],:z.shape[1]]
            camera_xyz=np.stack([(xx+.5-K[0,2])*z/K[0,0],(yy+.5-K[1,2])*z/K[1,1],z],-1)
            xyz=camera_xyz@c2w[:3,:3].T+c2w[:3,3]
            normal,valid=normals_from_grid(xyz,z,z>0,c2w[:3,3])
            selected=(z>0)&valid
            observations.add(dict(xyz=xyz[selected].astype(np.float32),normal=normal[selected],
                sensor=np.broadcast_to(c2w[:3,3],(selected.sum(),3)).astype(np.float32).copy(),rgb=array[selected]))
            print('SURFACE',surface_key,frame.stem,observations.count,flush=True)
        arrays=observations.finish()
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
        surface.mkdir(exist_ok=True)
        meter.save(surface/'nksr_input.npz',buffer.getbuffer());write(surface/'spatial_scale.json',scale)
        write(surface/'complete.json',dict(method=method,cleaned=args.cleaned,points=len(arrays['xyz']),source='Native rendered camera-Z, RAP masks, alpha>=.5, shared depth normals; all pixels then existing voxel-direction sampler',scale=scale,
            input_view_names=sorted(training_names),input_views=len(training_names),cached_views=len(views),
            held_out='27 test frames cached for evaluation only, excluded from oriented points, scale estimation and TSDF integration',
            scale_estimation='0.5/99.5 percentiles from deterministic up-to-4096 evenly spaced valid pixels per training frame',io=meter.stats()))
    write(output/'evaluation_complete.json',dict(status='complete',method=method,cleaned=args.cleaned,views=len(records),
        geometry_views=[Path(c.image_name).stem for c in scene.getTestCameras()],metrics_fields=list(records[0]),surface=args.surface))
    write(output/'evaluation_v2_complete.json',dict(status='complete',method=method,views=len(records),
        dark_mismatch='Render maxRGB<0.06 while original maxRGB>0.15 and alpha>=0.5; mismatch proxy, not an automatic semantic black-cloud label'))


if __name__=='__main__':main()
