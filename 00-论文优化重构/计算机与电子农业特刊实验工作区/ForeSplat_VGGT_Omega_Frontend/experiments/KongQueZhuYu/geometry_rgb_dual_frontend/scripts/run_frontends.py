"""Full 208-view original-background experiment, isolated from E1--E6."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('MPLBACKEND','Agg')
import argparse, csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OMEGA=ROOT.parents[2]
AUDIT=ROOT.parent/'root_cause_audit'
STANDARD=OMEGA.parent/'ForeSplat_Standard_VGGT_Frontend'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
common=module('existing_control_driver',AUDIT/'controlled_fixes/view_count_control_v1/run_view_count_control.py')
rgb_adapter=module('existing_geometry_rgb_adapter',AUDIT/'controlled_fixes/geometry_rgb_adapter.py')
write=common.write_json
NAMES={'Omega':'VGGT_Omega_Original_RGB','Standard':'Standard_VGGT_Original_RGB'}

def complete_export_record(model,out):
    metadata=json.loads((out/'metadata.json').read_text())
    exported=json.loads((out/'export_complete.json').read_text())
    metadata['preprocessing']['rgb_background']='Original 02-FFT RGB outside the unchanged RAP Mask; exact RAP pixels inside'
    metadata['preprocessing']['padding_background']='black square padding only' if model=='Standard' else 'none in this balanced resize'
    metadata['scale_semantics']='Native learned world gauge; no metric calibration or scan scale fitting. Existing frontend export fusion/sampling and rigid display convention are preserved.'
    metadata.update(derived_world_points_saved=True,all_208_new_geometry_inference_executed=True,
        point_output_note='Full depth-unprojected world point maps and mask/confidence-filtered points persisted; original learned point head remains unexecuted.',
        export_record=str(out/'export_complete.json'))
    write(out/'metadata.json',metadata)
    (out/'RUN_REPORT.md').write_text(f'# {model}: completed 208-view geometry inference and export\n\n'
        f'Actual new inference on the exact 208 historical frame IDs and order, with restored original background. '
        f'Existing audit geometry_rgb_adapter preserves RAP foreground pixels inside the unchanged mask. '
        f'Training RGB remains the historical RAP files. Physical GPU1, uv, seed0.\n\n'
        f'Original preprocessing: {metadata["preprocessing"]["name"]}. Official adapter call '
        f'{metadata["official_inference_call_seconds"]:.6f}s; peak Torch allocation {metadata["peak_vram_gib"]:.6f}GiB.\n\n'
        f'Camera/K/Depth/Confidence and full depth-unprojected Point Maps are saved. '
        f'Unfiltered PLY: {exported["unfiltered_points"]:,}; foreground/confidence-filtered PLY: '
        f'{exported["foreground_points"]:,}. Original and corrected COLMAP datasets share the same initialization PLY '
        f'and RGB/Mask files. Existing frontend fusion/sampling is preserved; no learned point head was introduced.\n\n'
        f'See metadata.json, export_complete.json, cameras.json, pixel_transforms.json and shared_manifest.csv.\n')

def prepare():
    ready=ROOT/'configs/prepared_sources.json'
    if ready.exists():return json.loads(ready.read_text())
    rows=common.source_rows()
    assert len(rows)==208 and [r['source_index'] for r in rows]==[i for i in range(210) if i not in (44,206)]
    for d in ['configs','geometry_images','training_images','masks','original_images','camera_interface','geometry_comparison','render_comparison']:
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    result=[];checks=[]
    for row in rows:
        sid=row['source_index'];name=row['image_name']
        raw=Path('/data/fj/F2DMAS/02-FFT/KongQueZhuYu')/f'{sid:04d}.jpg'
        rap=Path('/data/fj/F2DMAS/03-SAM/KongQueZhuYu')/name
        mask=rap.parent/f'mask_{sid:04d}.png'
        restored=ROOT/'geometry_images'/name
        check=rgb_adapter.restore_geometry_rgb(rap,mask,raw,appearance_rgb_path=rap,output_path=restored if not restored.exists() else None)
        # Existing adapter preserves RAP pixels inside Mask and restores every
        # outside pixel from the exact raw image, matching the root-cause control.
        if restored.exists():
            assert np.array_equal(np.asarray(Image.open(restored)),check.geometry_rgb)
        assert common.sha(rap)==row['rgb_sha256_historical']
        assert common.sha(mask)==row['mask_sha256_historical']
        for directory,source in [('training_images',rap),('masks',mask),('original_images',raw)]:
            target=ROOT/directory/(name if directory!='original_images' else raw.name)
            if not target.exists():target.symlink_to(source)
        w,h=check.metadata['original_resolution_wh'];assert (w,h)==(row['original_width'],row['original_height'])
        ao=np.array([[384/w,0,(384/w-1)/2],[0,688/h,(688/h-1)/2],[0,0,1]])
        ss=518/max(w,h);ast=np.array([[ss,0,((max(w,h)-w)//2)*ss],[0,ss,((max(w,h)-h)//2)*ss],[0,0,1.]])
        ast_true=ast.copy();ast_true[:2,2]+=(ss-1)/2
        result.append(dict(row,image_path=str(restored),mask_path=str(mask),geometry_rgb_path=str(restored),raw_rgb_path=str(raw),training_rgb_path=str(rap),
            condition='original background restored, exact RAP pixels inside original Mask',
            omega_model_width=384,omega_model_height=688,standard_model_width=518,standard_model_height=518,
            omega_pixel_transform=ao.tolist(),standard_pixel_transform_legacy=ast.tolist(),standard_pixel_transform=ast_true.tolist()))
        checks.append(check.metadata)
        print('validated source',sid,'foreground changes',check.metadata['foreground_changed_pixels'],flush=True)
    write(ROOT/'configs/input_invariants.json',dict(frames=checks,source_ids=[r['source_index'] for r in rows],excluded_historical_ids=[44,206],
        intervention='Existing geometry_rgb_adapter: exact original background outside RAP Mask; unchanged RAP foreground pixels. No additional blending/resize.',
        untouched_originals=str(ROOT/'original_images'),training_rgb_unchanged=True,mask_unchanged=True))
    with (ROOT/'shared_manifest.csv').open('w',newline='') as f:
        fields=list(result[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in result:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    write(ready,result)
    return result

def infer(model):
    rows=prepare();out=ROOT/NAMES[model]
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('status')=='complete':return
    info=common.model_metadata(model)
    info.update(control_factor='geometry RGB source on full historical208',all_frames=208,training_rgb_source=str(ROOT/'training_images'),mask_source=str(ROOT/'masks'))
    common.manifest(out,rows);write(out/'run_parameters.json',info)
    common.run(model,info,{208:(out,rows)})

def export(model):
    import torch
    sys.path[:0]=[str(OMEGA),str(STANDARD),str(STANDARD/'vggt')]
    out=ROOT/NAMES[model]
    if (out/'export_complete.json').exists():return
    cameras=json.loads((out/'cameras.json').read_text())['frames'];rows=prepare();n=len(rows);assert n==208
    from foresplat_adapter.inference import unproject_depth_map_to_point_map
    from foresplat_adapter.ply import write_point_cloud_ply
    depth=np.stack([np.load(out/'per_frame'/f'depth_{Path(r["image_name"]).stem}.npy') for r in rows])
    conf=np.stack([np.load(out/'per_frame'/f'confidence_{Path(r["image_name"]).stem}.npy') for r in rows])
    masks=np.stack([np.load(out/'per_frame'/f'model_mask_{Path(r["image_name"]).stem}.npy') for r in rows]).astype(bool)
    e=np.array([r['extrinsic_model_w2c'] for r in cameras]);k=np.array([r['intrinsic_model'] for r in cameras])
    world=unproject_depth_map_to_point_map(depth[...,None],e[:,:3],k).astype(np.float32)
    h,w=depth.shape[1:]
    # Independently replay the ORIGINAL appearance preprocessing, never replace
    # the saved model RGB or the GGGS training images with restored-background RGB.
    paths=[r['training_rgb_path'] for r in rows];mp=[r['mask_path'] for r in rows]
    if model=='Omega':
        from vggt_omega.utils.load_fn import load_and_preprocess_images_with_masks
        rgb,appearance_masks,transforms=load_and_preprocess_images_with_masks(paths,mp,mode='balanced',image_resolution=512,patch_size=16,background='black')
    else:
        from vggt_foreground.preprocess import load_and_preprocess_images_square_with_masks
        rgb,appearance_masks,affines,sizes=load_and_preprocess_images_square_with_masks(paths,mp)
        rgb=torch.nn.functional.interpolate(rgb.to('cuda'),size=(518,518),mode='bilinear',align_corners=False).cpu()
        transforms=json.loads((out/'pixel_transforms.json').read_text())
    assert np.array_equal(masks,appearance_masks.numpy().astype(bool))
    rgb=rgb.numpy();colors=(rgb.transpose(0,2,3,1).clip(0,1)*255).round().astype(np.uint8)
    finite=np.isfinite(world).all(-1)&np.isfinite(depth)&(depth>0)&np.isfinite(conf)
    threshold=float(np.percentile(conf[masks&finite],20));valid=masks&finite&(conf>=threshold)&(conf>1e-5)
    for i,row in enumerate(rows):
        stem=Path(row['image_name']).stem
        np.save(out/'per_frame'/f'world_points_{stem}.npy',world[i])
        np.save(out/'per_frame'/f'foreground_world_points_{stem}.npy',world[i][valid[i]])
        np.save(out/'per_frame'/f'valid_foreground_{stem}.npy',valid[i])
        np.save(out/'per_frame'/f'appearance_rgb_{stem}.npy',colors[i])
    # Complete unfiltered and mask/conf-filtered PLY, no display subsampling.
    geometry_rgb=np.stack([np.load(out/'per_frame'/f'model_rgb_float32_{Path(r["image_name"]).stem}.npy').transpose(1,2,0) for r in rows])
    gc=(geometry_rgb.clip(0,1)*255).round().astype(np.uint8)
    write_point_cloud_ply(str(out/'unfiltered.ply'),world[finite],gc[finite],comments=['Full depth-unprojected point map, before foreground filtering; not learned point head'])
    write_point_cloud_ply(str(out/'foreground.ply'),world[valid],colors[valid],comments=['Full original RAP mask + original p20 confidence, no fusion'])
    del geometry_rgb,gc
    dataset=out/'COLMAP_Original';dataset.mkdir(exist_ok=True)
    if model=='Omega':
        from foresplat_adapter.full_sequence import build_fusion_views
        from foresplat_adapter.fusion import fuse_views,FusionConfig
        from foresplat_adapter.export import export_gggs_dataset
        from foresplat_adapter.dataset import FramePaths
        from visual_util import get_scene_alignment_transform
        predictions=dict(depth=depth[...,None],depth_conf=conf,world_points_from_depth=world,images=rgb,extrinsic=e[:,:3],intrinsic=k)
        views,t=build_fusion_views(predictions,masks,e[:,:3]);assert abs(t-threshold)<1e-6
        fusion=fuse_views(views,FusionConfig());fused=fusion['fused'];align=get_scene_alignment_transform(e[:,:3])
        frames=[FramePaths(index=i,image_path=r['training_rgb_path'],mask_path=r['mask_path'],image_name=r['image_name']) for i,r in enumerate(rows)]
        summary=export_gggs_dataset(predictions,transforms,masks,frames,str(dataset),checkpoint_path=str(OMEGA/'weights/vggt_omega_1b_512.pt'),link_sources=True,
            fused_points=dict(xyz=fused['xyz']@align[:3,:3].T+align[:3,3],colors=fused['colors'],normals=fused['normals'],is_edge=fused['is_edge'],observation_count=fused['observation_count'],confidence=fused['confidence']),observations=fusion['observations'])
        write(out/'fusion_metadata.json',dict(config=vars(FusionConfig()),stats=fusion['stats'],scene_alignment_matrix=align))
    else:
        from run_standard_vggt_frontend import stratified_allocate
        from scipy.spatial.transform import Rotation
        sparse=dataset/'sparse/0';sparse.mkdir(parents=True,exist_ok=True)
        allocation=stratified_allocate(valid.reshape(n,-1).sum(-1),1252383);rng=np.random.default_rng(0);xyz=[];cs=[]
        with (sparse/'cameras.txt').open('w') as cf,(sparse/'images.txt').open('w') as ef,(sparse/'points3D.txt').open('w') as pf:
            pid=1
            for i,row in enumerate(rows):
                a=np.array(cameras[i]['pixel_transform']);ki=np.linalg.inv(a)@k[i];ww,hh=row['original_width'],row['original_height']
                cf.write(f'{i+1} PINHOLE {ww} {hh} '+ ' '.join(map(str,[ki[0,0],ki[1,1],ki[0,2],ki[1,2]]))+'\n')
                q=Rotation.from_matrix(e[i,:3,:3]).as_quat();ef.write(' '.join(map(str,[i+1,q[3],*q[:3],*e[i,:3,3],i+1,row['image_name']]))+'\n')
                idx=np.flatnonzero(valid[i].ravel());kk=int(allocation[i]);idx=rng.choice(idx,kk,replace=False) if kk<len(idx) else idx
                idx=np.sort(idx);xx=world[i].reshape(-1,3)[idx];cc=colors[i].reshape(-1,3)[idx];xyz.append(xx);cs.append(cc)
                uv=np.c_[idx%w,idx//w,np.ones(len(idx))]@np.linalg.inv(a).T;tokens=[]
                for j in range(len(idx)):
                    tokens.extend([str(uv[j,0]),str(uv[j,1]),str(pid)])
                    pf.write(' '.join(map(str,[pid,*xx[j],*cc[j],0,i+1,j]))+'\n');pid+=1
                ef.write(' '.join(tokens)+'\n')
        write_point_cloud_ply(str(sparse/'points3D.ply'),np.concatenate(xyz),np.concatenate(cs),normals=np.zeros((sum(map(len,xyz)),3)))
        for directory in ['images','masks']:
            (dataset/directory).symlink_to(ROOT/('training_images' if directory=='images' else 'masks'))
        summary=dict(num_fused_points_in_ply=sum(map(len,xyz)),points3d_ply_is_fused=False)
    # Corrected dataset differs exclusively in K/2D pixel transforms. PLY and
    # appearance files are the SAME files. Camera poses are unchanged.
    import shutil
    corrected=out/'COLMAP_Corrected';shutil.copytree(dataset,corrected,symlinks=True)
    for directory in ['images','masks']:
        target=corrected/directory
        if target.is_symlink():target.unlink()
        else:shutil.rmtree(target)
        target.symlink_to(dataset/directory)
    import pycolmap
    recon=pycolmap.Reconstruction(str(corrected/'sparse/0'))
    for image in recon.images.values():
        i=next(j for j,r in enumerate(rows) if r['image_name']==image.name)
        old=np.array(cameras[i]['pixel_transform']);new=np.array(cameras[i]['pixel_transform_pixel_center']);kk=np.linalg.inv(new)@k[i]
        recon.cameras[image.camera_id].params=np.array([kk[0,0],kk[1,1],kk[0,2],kk[1,2]])
        if model=='Standard':
            change=np.linalg.inv(new)@old
            for p in image.points2D:p.xy=(change@np.r_[p.xy,1])[:2]
    recon.write(str(corrected/'sparse/0'));recon.write_text(str(corrected/'sparse/0'))
    if model=='Omega':
        # Omega A was already pixel-center correct. Preserve its pose and
        # observation bytes, avoiding pycolmap's quaternion renormalization.
        for name in ['images.bin','images.txt','points3D.bin','points3D.txt']:
            if (dataset/'sparse/0'/name).exists():shutil.copy2(dataset/'sparse/0'/name,corrected/'sparse/0'/name)
    # Replace duplicated point initialization by a symlink, preserving identity.
    (corrected/'sparse/0/points3D.ply').unlink();(corrected/'sparse/0/points3D.ply').symlink_to(dataset/'sparse/0/points3D.ply')
    write(out/'export_complete.json',dict(model=model,frames=n,world_point_channel='depth unprojection using unchanged original official camera/depth heads; not learned point head',
        confidence_percentile=20,confidence_threshold=threshold,unfiltered_points=int(finite.sum()),foreground_points=int(valid.sum()),summary=summary,
        original_dataset=str(dataset),corrected_dataset=str(corrected),same_init_ply=True))
    complete_export_record(model,out)
    print('EXPORT COMPLETE',model,summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','infer','export']);p.add_argument('--model',choices=list(NAMES));a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='infer':infer(a.model)
    else:export(a.model)
