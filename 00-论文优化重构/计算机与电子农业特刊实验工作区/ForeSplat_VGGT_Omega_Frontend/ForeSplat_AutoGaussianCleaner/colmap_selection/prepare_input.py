"""Prepare one true SfM dataset; never mutate historical source/model files."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, json, struct, time, importlib.util, hashlib, resource
from pathlib import Path
import numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parents[1]
ROOT=Path('/data/fj/F2DMAS')
SOURCE=ROOT/'04-COLMAP-rerun-original/KongQueZhuYu/sequential'
OUT=BASE/'experiments/COLMAP_Gaussian_Mesh_Selection/common_input'
sys.path.insert(0,str(BASE))
from nksr_pipeline.common import Meter,PacedFile,hash_file,stamp,write
E0=BASE.parent.parent/'阶段二十E0_三维全视图与输入冻结/scripts'
sys.path.insert(0,str(E0))
from build_datasets import inverse_maps, remap_nearest, remap_bilinear, native_camera
spec=importlib.util.spec_from_file_location('source_colmap',ROOT/'2d-gaussian-splatting-main/scene/colmap_loader.py')
io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io)


def load_image(path,meter,mode='RGB'):
    with PacedFile(path,'rb',meter) as stream:
        with Image.open(stream) as image:return np.array(image.convert(mode))


def main():
    if (OUT/'complete.json').exists():
        print('Common input already complete');return
    wall=time.perf_counter();cpu=time.process_time();meter=Meter(20)
    dataset=OUT/'dataset';sparse=dataset/'sparse/0'
    for path in [sparse,dataset/'images',dataset/'masks']:path.mkdir(parents=True,exist_ok=True)
    evidence={}
    for source in [SOURCE/'sparse/0'/n for n in ['cameras.bin','images.bin','points3D.bin']]+[SOURCE/'distorted/sparse/1/cameras.bin']+[SOURCE/'logs'/n for n in ['feature_extractor.log','sequential_matcher.log','mapper.log','image_undistorter.log']]:
        evidence[str(source)]=dict(**stamp(source),sha256=hash_file(source,meter))
    # Read only small source model after paced copies. The dataset images.bin
    # observations are transformed consistently with its scaled intrinsics.
    originals=OUT/'source_sparse';originals.mkdir(exist_ok=True)
    for name in ['cameras.bin','images.bin','points3D.bin']:
        with PacedFile(SOURCE/'sparse/0'/name,'rb',meter) as src,PacedFile(originals/name,'wb',meter) as dst:
            while block:=src.read(1024*1024):dst.write(block)
    cameras=io.read_intrinsics_binary(originals/'cameras.bin')
    views=io.read_extrinsics_binary(originals/'images.bin')
    points,rgb,errors=io.read_points3D_binary(originals/'points3D.bin')
    assert len(cameras)==1 and len(views)==210 and len(points)==177918
    source_camera=next(iter(io.read_intrinsics_binary(SOURCE/'distorted/sparse/1/cameras.bin').values()))
    camera=next(iter(cameras.values()));assert camera.model=='PINHOLE'
    width,height=530,948;sx,sy=width/camera.width,height/camera.height
    params=camera.params*np.array([sx,sy,sx,sy])
    with PacedFile(sparse/'cameras.bin','wb',meter) as f:
        f.write(struct.pack('<QiiQQdddd',1,camera.id,1,width,height,*params))
    with PacedFile(sparse/'images.bin','wb',meter) as f:
        f.write(struct.pack('<Q',len(views)))
        for image in views.values():
            f.write(struct.pack('<i7di',image.id,*image.qvec,*image.tvec,image.camera_id))
            f.write(image.name.encode()+b'\0');f.write(struct.pack('<Q',len(image.xys)))
            rows=np.empty(len(image.xys),dtype=[('x','<f8'),('y','<f8'),('id','<i8')])
            rows['x']=image.xys[:,0]*sx;rows['y']=image.xys[:,1]*sy;rows['id']=image.point3D_ids
            f.write(rows.tobytes())
    with PacedFile(originals/'points3D.bin','rb',meter) as src,PacedFile(sparse/'points3D.bin','wb',meter) as dst:
        while block:=src.read(1024*1024):dst.write(block)
    # Native GS readers expect existing PLY. XYZ/order/RGB are copied from the
    # same SfM points; no Gaussian centres or frontend-predicted points involved.
    rows=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('nx','<f4'),('ny','<f4'),('nz','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for axis,key in enumerate(['x','y','z']):rows[key]=points[:,axis]
    for key in ['nx','ny','nz']:rows[key]=0
    for axis,key in enumerate(['red','green','blue']):rows[key]=rgb[:,axis]
    header='ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(points))+'\n'+''.join('property float '+k+'\n' for k in ['x','y','z','nx','ny','nz'])+''.join('property uchar '+k+'\n' for k in ['red','green','blue'])+'end_header\n'
    with PacedFile(sparse/'points3D.ply','wb',meter) as f:f.write(header.encode());f.write(rows.tobytes())
    source_native,target_native=native_camera(source_camera),native_camera(camera)
    mx,my,valid=inverse_maps(source_native,target_native)
    names=sorted(image.name for image in views.values());train=[n for i,n in enumerate(names) if i%8];test=names[::8]
    (OUT/'train_views.txt').write_text('\n'.join(train)+'\n');(OUT/'test_views.txt').write_text('\n'.join(test)+'\n')
    manifest=[];checks=[];trajectory=[]
    for order,name in enumerate(names):
        image_path=SOURCE/'images'/name
        image=load_image(image_path,meter)
        assert image.shape[:2]==(camera.height,camera.width)
        mask_source=ROOT/'03-SAM/KongQueZhuYu'/('mask_'+Path(name).stem+'.png')
        mask=load_image(mask_source,meter,'L')>127
        mapped=remap_nearest(mask,mx,my,valid)
        small=Image.fromarray(image).resize((width,height),Image.Resampling.LANCZOS)
        with PacedFile(dataset/'images'/name,'wb',meter) as f:small.save(f,format='PNG')
        smallmask=Image.fromarray(mapped.astype(np.uint8)*255).resize((width,height),Image.Resampling.NEAREST)
        mask_name=Path(name).stem+'.png'
        with PacedFile(dataset/'masks'/mask_name,'wb',meter) as f:smallmask.save(f,format='PNG')
        manifest.append(dict(name=name,source=stamp(image_path),source_sha256=hash_file(image_path,meter),
            image_sha256=hash_file(dataset/'images'/name,meter),mask_source=stamp(mask_source),mask_source_sha256=hash_file(mask_source,meter),mask_sha256=hash_file(dataset/'masks'/mask_name,meter),
            split='test' if order%8==0 else 'train'))
        if order in [0,50,100,150,200]:
            raw_path=ROOT/'02-FFT/KongQueZhuYu'/name
            raw=load_image(raw_path,meter);expected=remap_bilinear(raw,mx,my,valid)
            difference=np.abs(expected.astype(np.int16)-image.astype(np.int16))
            mean=float(difference[valid].mean());assert mean<5,('Historical RGB differs from raw undistortion',name,mean)
            checks.append(dict(name=name,original_rgb=str(raw_path),historical_undistorted_rgb_mae_0_255=mean,
                black_pixel_fraction=float((image.max(-1)<5).mean()),mask_background_fraction=float((~mapped).mean())))
        if order%20==0:print('prepare common RGB/masks',order+1,'/210',flush=True)
    for image in sorted(views.values(),key=lambda im:im.name):
        r=image.qvec2rotmat();trajectory.append(dict(name=image.name,qvec=image.qvec.tolist(),tvec=image.tvec.tolist(),camera_center=(-r.T@image.tvec).tolist()))
    write(OUT/'camera_trajectory.json',trajectory);write(OUT/'image_manifest.json',manifest)
    report=dict(source=str(SOURCE),source_alias=str(ROOT/'00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked/KongQueZhuYu'),
        provenance='Historical actual CPU SIFT, sequential matching, triangulation and bundle adjustment; distorted model1; native COLMAP image undistorter',
        input_images=210,registered_images=210,registration_rate=1.,sparse_points=len(points),mean_point_reprojection_error_px=float(errors.mean()),median_point_reprojection_error_px=float(np.median(errors)),final_ba_cost_px=.754325,
        source_evidence=evidence,width=width,height=height,K=[[params[0],0,params[2]],[0,params[1],params[3]],[0,0,1]],
        pixel_convention='Native COLMAP edge-centre coordinates; K and sparse xys scaled independently by width/height ratio',
        source_camera=dict(model=source_camera.model,width=source_camera.width,height=source_camera.height,params=source_camera.params.tolist()),
        undistorted_camera=dict(model=camera.model,width=camera.width,height=camera.height,params=camera.params.tolist()),
        train_count=len(train),test_count=len(test),split='Sorted names, every eighth image held out starting index0',
        seed=0,sh_degree=3,iterations=30000,mask_loss=False,
        rgb='Existing native COLMAP undistorted raw RGB; resize LANCZOS; lossless PNG payload retains original registered filename .jpg for all loaders',
        mask_provenance='Existing 03-SAM masks, not automatically certified as later RAP production; source identities recorded',
        mask_mapping='Reused stage20E0 inverse_maps and remap_nearest, target halfpixel -> original OPENCV -> array halfpixel; nearest downsample',
        raw_rgb_consistency_checks=checks,dataset=str(dataset))
    write(OUT/'colmap_source.json',report)
    write(OUT/'resources.json',dict(wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        logical_read_bytes=meter.read_bytes,logical_write_bytes=meter.write_bytes,observed_peak_one_second_mib_s=meter.peak_window_bytes/2**20,gpu=None))
    write(OUT/'complete.json',dict(status='complete',dataset=str(dataset),views=len(names),train=len(train),test=len(test)))
    print('COMMON INPUT COMPLETE',flush=True)

if __name__=='__main__':main()
