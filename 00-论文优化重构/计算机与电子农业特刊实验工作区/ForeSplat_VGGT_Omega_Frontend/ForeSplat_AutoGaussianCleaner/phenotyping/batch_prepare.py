"""Verify genuine feature-track SfM and prepare raw RGB in its camera domain."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from batch_common import *
import struct,subprocess
from types import SimpleNamespace

def locate(ctx):
 io_col=colmap_io();candidates=[]
 roots=[ctx.out/'colmap',ROOT/'04-COLMAP-rerun-original'/ctx.plant,ROOT/'04-COLMAP'/ctx.plant]
 for base in roots:
  for p in base.glob('**/sparse/0/cameras.bin'):
   if 'distorted' in p.parts:continue
   directory=p.parents[2]
   try:
    cams=io_col.read_intrinsics_binary(p);views=io_col.read_extrinsics_binary(p.with_name('images.bin'));obs=sum(int((v.point3D_ids>=0).sum()) for v in views.values())
    if len(cams)!=1 or obs<300:continue
    raw=None
    for q in directory.glob('distorted/sparse/*/images.bin'):
     original=io_col.read_extrinsics_binary(q)
     if not original:continue
     item=next(iter(views.values()));other=original.get(item.id)
     if other is not None and other.name==item.name and np.allclose(other.tvec,item.tvec,atol=1e-6) and np.allclose(other.qvec,item.qvec,atol=1e-6):raw=q.parent;break
    if raw is None:continue
    candidates.append((len(views),obs,directory,raw))
   except (FileNotFoundError,ValueError,EOFError):continue
 if not candidates:return None
 local=[c for c in candidates if c[2]==ctx.out/'colmap']
 if local:candidates=local
 _,_,directory,raw=max(candidates,key=lambda r:(r[0],r[1]));return directory,raw

def run_sfm(ctx):
 work=ctx.out/'colmap';inputdir=work/'input';inputdir.mkdir(parents=True,exist_ok=True)
 for p in sorted(ctx.raw.glob('*.jpg')):
  if not(inputdir/p.name).exists():link(inputdir/p.name,p)
 (work/'distorted/sparse').mkdir(parents=True,exist_ok=True);(work/'logs').mkdir(exist_ok=True)
 commands=[['feature_extractor','--database_path',str(work/'distorted/database.db'),'--image_path',str(inputdir),'--ImageReader.camera_model','OPENCV','--ImageReader.single_camera','1','--SiftExtraction.use_gpu','0','--SiftExtraction.num_threads','8','--SiftExtraction.max_image_size','2000'],['sequential_matcher','--database_path',str(work/'distorted/database.db'),'--SiftMatching.use_gpu','0','--SiftMatching.num_threads','8','--SequentialMatching.overlap','10'],['mapper','--database_path',str(work/'distorted/database.db'),'--image_path',str(inputdir),'--output_path',str(work/'distorted/sparse'),'--Mapper.num_threads','8']]
 for command in commands:
  marker=work/'logs'/f'{command[0]}.complete.json'
  if marker.exists():continue
  with (work/'logs'/f'{command[0]}.log').open('w') as log:subprocess.run(['colmap',*command],env=dict(os.environ,QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
  write(marker,dict(command=['colmap',*command],status='complete'))
 io_col=colmap_io();models=list((work/'distorted/sparse').glob('*/images.bin'));best=max(models,key=lambda p:len(io_col.read_extrinsics_binary(p))).parent
 with (work/'logs/image_undistorter.log').open('w') as log:subprocess.run(['colmap','image_undistorter','--image_path',str(inputdir),'--input_path',str(best),'--output_path',str(work),'--output_type','COLMAP','--max_image_size','2000'],stdout=log,stderr=subprocess.STDOUT,check=True)
 # Native image_undistorter produces sparse/{cameras,images,points3D}.bin.
 sparse=work/'sparse/0';sparse.mkdir(exist_ok=True)
 for n in ['cameras.bin','images.bin','points3D.bin']:
  if (work/'sparse'/n).exists():(work/'sparse'/n).replace(sparse/n)
 return work,best

def reuse_native_distorted(ctx):
 """A valid native distorted SfM needs undistortion, not a new pose solve."""
 loader=colmap_io();options=[]
 for base in [ROOT/'04-COLMAP-rerun-original'/ctx.plant,ROOT/'04-COLMAP'/ctx.plant]:
  for file in base.glob('**/distorted/sparse/*/images.bin'):
   try:
    views=loader.read_extrinsics_binary(file);obs=sum(int((v.point3D_ids>=0).sum()) for v in views.values());cams=loader.read_intrinsics_binary(file.with_name('cameras.bin'));directory=file.parents[3]
    if len(views)>10 and obs>300 and len(cams)==1 and (directory/'input').is_dir():options.append((len(views),obs,directory,file.parent))
   except (FileNotFoundError,ValueError,EOFError):continue
 if not options:return None
 _,_,directory,raw=max(options,key=lambda row:(row[0],row[1]));work=ctx.out/'colmap';work.mkdir(exist_ok=True);(work/'logs').mkdir(exist_ok=True)
 if not(work/'input').exists():link(work/'input',directory/'input')
 if not(work/'distorted/sparse/0').exists():link(work/'distorted/sparse/0',raw)
 command=['colmap','image_undistorter','--image_path',str(work/'input'),'--input_path',str(raw),'--output_path',str(work),'--output_type','COLMAP','--max_image_size','2000']
 with (work/'logs/reuse_native_undistorter.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,QT_QPA_PLATFORM='offscreen',CUDA_VISIBLE_DEVICES='1'),check=True)
 (work/'sparse/0').mkdir(exist_ok=True)
 for name in ['cameras.bin','images.bin','points3D.bin']:(work/'sparse'/name).replace(work/'sparse/0'/name)
 write(work/'reused_native_sfm.json',dict(source=str(raw),input=str(directory/'input'),action='Reuse genuine native feature-track SfM; undistort its original camera/images without changing poses',command=command))
 return work,raw

def mask_source(ctx,name):
 stem=Path(name).stem;archive=ROOT/'00-论文优化重构/数据管理/03-分割Mask/05-RAP-FSAM3掩膜'
 choices=[archive/'E4b_downstream'/ctx.plant/'最终掩膜'/f'mask_{stem}.png']
 choices.extend(p/'最终掩膜'/f'mask_{stem}.png' for p in sorted(archive.glob(f'E1_{ctx.plant}_A6A7*')))
 choices.append(Path('/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1')/ctx.plant/'R0'/f'mask_{stem}.png')
 choices.append(ROOT/'03-SAM'/ctx.plant/f'mask_{stem}.png')
 for p in choices:
  if not p.exists():continue
  try:
   with Image.open(p) as im:im.verify()
   return p
  except Exception:continue
 return None

def bind_registered_rgb(ctx,directory,views):
 files=sorted(ctx.raw.glob('*.jpg'));fft_hash={digest(file,ctx.meter):file.stem for file in files};identities=[];pending=[]
 for image in sorted(views.values(),key=lambda x:x.name):
  registered=Path(image.name).stem.removeprefix('crop_');source=directory/'input'/image.name
  if not source.exists():source=ROOT/'01-FFmepg'/ctx.plant/f'{registered}.jpg'
  if not source.exists():source=ctx.raw/f'{registered}.jpg'
  if not source.exists():continue
  stamp=digest(source,ctx.meter);name=fft_hash.get(stamp)
  if name is not None:identities.append((image,name,ctx.raw/f'{name}.jpg',registered,stamp,'ExactCOLMAP input byte SHA256 matched currentFFT RGB; image renamed jointly with sparse observations'));continue
  pending.append((image,registered,source))
 if pending:
  # Native COLMAP can have used a black-masked PNG of an original RGB.
  # Sampled pixels identify a candidate only; ALL retained RGB pixels must
  # match byte-for-byte before the camera is accepted. No image similarity.
  samples=[]
  for file in files:
   try:im=Image.open(io.BytesIO(ctx.meter.read(file))).convert('RGB');samples.append(np.array(im.resize((64,114),Image.Resampling.NEAREST)))
   except Exception:samples.append(np.zeros((114,64,3),np.uint8))
  samples=np.stack(samples)
  for image,registered,source in pending:
   try:original=np.array(Image.open(io.BytesIO(ctx.meter.read(source))).convert('RGB'))
   except Exception:continue
   small=np.array(Image.fromarray(original).resize((64,114),Image.Resampling.NEAREST));visible=small.any(2)
   if visible.sum()<100:continue
   matches=np.flatnonzero(np.all(samples[:,visible]==small[visible],axis=(1,2)))
   if len(matches)!=1:continue
   file=files[int(matches[0])];raw=np.array(Image.open(io.BytesIO(ctx.meter.read(file))).convert('RGB'));retained=original.any(2)
   if original.shape!=raw.shape or retained.sum()<10000 or not np.array_equal(original[retained],raw[retained]):continue
   sha=digest(file,ctx.meter);proof='Exact ALL retained nonblack originalCOLMAP PNG pixels verified against rawFFT RGB; raw SHA256 recorded; source-image pixels outside mask are black. Cameras renamed jointly, no guessed filename correspondence.';identities.append((image,file.stem,file,registered,sha,proof))
 return sorted(identities,key=lambda row:row[0].name)

def prepare(ctx, *, source=None):
 marker=ctx.common/'complete.json'
 if marker.exists():return read(marker)
 start=time.time();ctx.status('colmap','running');io_col=colmap_io()
 if ctx.plant=='KongQueZhuYu':
  original=read(PRE/'common_input/colmap_source.json');trajectory=read(PRE/'common_input/camera_trajectory.json')
  for directory,target in [('images',PRE/'common_input/dataset/images'),('rap_masks',PRE/'common_input/dataset/rap_masks'),('sparse',PRE/'common_input/dataset/sparse')]:link(ctx.common/directory,target)
  views=[]
  for r in trajectory:
   q=r['qvec'];R=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix();views.append(dict(name=Path(r['name']).stem,qvec=q,R=R.tolist(),tvec=r['tvec'],center=(-R.T@r['tvec']).tolist(),registered_filename=r['name']))
  data=dict(K=original['K'],width=original['width'],height=original['height'],views=views,coordinate_frame='ActualCOLMAP original model units',source_camera=original['source_camera'],undistorted_camera=original['undistorted_camera'],registered_images=len(views),sparse_points=original['sparse_points'],provenance=original['provenance'],source=original['source'])
  write(ctx.common/'cameras.json',data)
  for split in ['train','test']:link(ctx.common/f'{split}_views.txt',PRE/'common_input'/f'{split}_views.txt')
  result=dict(status='complete',reuse=True,source=original['source'],registered_images=len(views),sparse_points=original['sparse_points']);write(marker,result);ctx.status('colmap','complete',**{k:v for k,v in result.items() if k!='status'});return result
 explicit_source=source is not None
 if not explicit_source:source=locate(ctx)
 if source is None:
  try:source=reuse_native_distorted(ctx)
  except subprocess.CalledProcessError as failure:
   # Some verified historical inputs contain corrupt PNG files. Preserve the
   # failed native undistortion and run genuine SfM on the current raw RGB.
   work=ctx.out/'colmap'
   if work.exists():
    archive=ctx.out/'colmap_attempts'/f'reuse_undistortion_failed_{time.time_ns()}';archive.parent.mkdir(exist_ok=True);work.rename(archive)
    write(archive/'failure.json',dict(command=failure.cmd,returncode=failure.returncode,next_action='Run genuine COLMAP SfM on current raw RGB; historical source remains untouched'))
 if source is None:source=run_sfm(ctx)
 directory,raw_model=source;sparse=directory/'sparse/0';cams=io_col.read_intrinsics_binary(sparse/'cameras.bin');views=io_col.read_extrinsics_binary(sparse/'images.bin');cam=next(iter(cams.values()));rawcam=next(iter(io_col.read_intrinsics_binary(raw_model/'cameras.bin').values()));xyz,color,error=io_col.read_points3D_binary(sparse/'points3D.bin')
 preparation_key=dict(source=str(directory.resolve()),raw_model=str(raw_model.resolve()),camera=dict(model=cam.model,width=cam.width,height=cam.height,params=cam.params.tolist()),raw_camera=dict(model=rawcam.model,width=rawcam.width,height=rawcam.height,params=rawcam.params.tolist()),target_height=948)
 previous=ctx.common/'prepare_signature.json'
 if previous.exists() and read(previous)!=preparation_key:
  archive=ctx.out/'input_attempts'/str(time.time_ns());archive.parent.mkdir(exist_ok=True);ctx.common.rename(archive)
 write(ctx.common/'prepare_signature.json',preparation_key)
 observations=sum(int((v.point3D_ids>=0).sum()) for v in views.values());assert len(xyz)>100 and observations>len(xyz),'Actual triangulated observations required'
 height=948;width=round(cam.width/cam.height*height);sx,sy=width/cam.width,height/cam.height;params=cam.params*np.array([sx,sy,sx,sy]);assert cam.model=='PINHOLE'
 sys.path.insert(0,str(FRONT.parent/'阶段二十E0_三维全视图与输入冻结/scripts'));from build_datasets import inverse_maps,remap_nearest,remap_bilinear,native_camera
 target=SimpleNamespace(model='PINHOLE',width=width,height=height,params=params);mx,my,valid=inverse_maps(native_camera(rawcam),native_camera(target))
 for sub in ['images','rap_masks','sparse/0']: (ctx.common/sub).mkdir(parents=True,exist_ok=True)
 with (ctx.common/'sparse/0/cameras.bin').open('wb') as stream:stream.write(struct.pack('<QiiQQdddd',1,cam.id,1,width,height,*params))
 # Some historical FFT exports renumbered filenames after frame removal.
 # Exact byte hashes bind each registered camera to the actual RGB and mask.
 identity=bind_registered_rgb(ctx,directory,views)
 if len(identity)<max(10,len(views)*.9):
  write(ctx.out/'colmap/rejected_historical_input.json',dict(source=str(directory),mapped_images=len(identity),registered_images=len(views),reason='Original registered RGB SHA256 cannot be matched to current FFT input; nativeSfM must be rerun, no guessed filename correspondence'))
  if explicit_source:raise ValueError('Explicit COLMAP input mapping incomplete')
  if directory==ctx.out/'colmap':
   if (directory/'reused_native_sfm.json').exists():
    archive=ctx.out/'colmap_attempts'/f'reused_unmatched_{time.time_ns()}';archive.parent.mkdir(exist_ok=True);directory.rename(archive)
   else:raise ValueError('Fresh COLMAP input mapping incomplete')
  run_sfm(ctx);return prepare(ctx)
 ordered=[r[0] for r in identity];trajectory=[];mapping=[];missing_masks=[]
 buffer=io.BytesIO();buffer.write(struct.pack('<Q',len(ordered)))
 for i,(image,name,rgbpath,registered,source_hash,identity_proof) in enumerate(identity):
  buffer.write(struct.pack('<i7di',image.id,*image.qvec,*image.tvec,image.camera_id));buffer.write((name+'.jpg').encode()+b'\0');buffer.write(struct.pack('<Q',len(image.xys)));r=np.empty(len(image.xys),dtype=[('x','<f8'),('y','<f8'),('id','<i8')]);r['x']=image.xys[:,0]*sx;r['y']=image.xys[:,1]*sy;r['id']=image.point3D_ids;buffer.write(r.tobytes())
  if not(ctx.common/'images'/f'{name}.jpg').exists() or name!=registered:
   raw=np.array(Image.open(io.BytesIO(ctx.meter.read(rgbpath))).convert('RGB'));assert raw.shape[:2]==(rawcam.height,rawcam.width),'Raw RGB/camera domain mismatch';undistorted=remap_bilinear(raw,mx,my,valid);png(ctx.common/'images'/f'{name}.jpg',undistorted,ctx.meter)
  maskpath=mask_source(ctx,name)
  if maskpath is None:
   missing_masks.append(name);mapped=valid.copy()
  else:
   mask=np.array(Image.open(io.BytesIO(ctx.meter.read(maskpath))).convert('L'))>0;assert mask.shape==(rawcam.height,rawcam.width);mapped=remap_nearest(mask,mx,my,valid)
  png(ctx.common/'rap_masks'/f'{name}.png',mapped.astype('uint8')*255,ctx.meter)
  R=image.qvec2rotmat();trajectory.append(dict(name=name,registered_filename=name+'.jpg',original_registered_name=image.name,qvec=image.qvec.tolist(),R=R.tolist(),tvec=image.tvec.tolist(),center=(-R.T@image.tvec).tolist()));mapping.append(dict(view=name,original_registered_name=image.name,raw=str(rgbpath),raw_sha256=source_hash,mask_source=str(maskpath),name_match_verified=identity_proof,image_domain='Raw RGB undistorted with actual SfM distortion camera; no black mask in training RGB'))
  if i%20==0:print('PREPARE',ctx.plant,i+1,len(ordered),flush=True)
 ctx.meter.save(ctx.common/'sparse/0/images.bin',buffer.getbuffer());ctx.meter.save(ctx.common/'sparse/0/points3D.bin',ctx.meter.read(sparse/'points3D.bin'))
 from plyfile import PlyData,PlyElement
 a=np.empty(len(xyz),dtype=[(n,'f4') for n in ['x','y','z','nx','ny','nz']]+[(n,'u1') for n in ['red','green','blue']])
 for axis,n in enumerate(['x','y','z']):a[n]=xyz[:,axis]
 for n in ['nx','ny','nz']:a[n]=0
 for axis,n in enumerate(['red','green','blue']):a[n]=color[:,axis]
 b=io.BytesIO();PlyData([PlyElement.describe(a,'vertex')]).write(b);ctx.meter.save(ctx.common/'sparse/0/points3D.ply',b.getbuffer())
 names=[r['name']+'.jpg' for r in trajectory];(ctx.common/'train_views.txt').write_text('\n'.join(n for i,n in enumerate(names) if i%8)+'\n');(ctx.common/'test_views.txt').write_text('\n'.join(names[::8])+'\n')
 data=dict(K=[[params[0],0,params[2]],[0,params[1],params[3]],[0,0,1]],width=width,height=height,views=trajectory,coordinate_frame='ActualCOLMAP original model units',source_camera=dict(model=rawcam.model,width=rawcam.width,height=rawcam.height,params=rawcam.params.tolist()),undistorted_camera=dict(model=cam.model,width=cam.width,height=cam.height,params=cam.params.tolist()),source=str(directory),registered_images=len(ordered),sparse_points=len(xyz),valid_feature_observations=observations,mean_sparse_reprojection_px=float(error.mean()),provenance='Historical native feature-track sparse reconstruction with paired distorted/undistorted camera models; not manifest Omega imported poses')
 if directory==ctx.out/'colmap' and not(directory/'reused_native_sfm.json').exists():data['provenance']='Fresh genuine COLMAP feature extraction, matching and incremental SfM on current raw RGB, followed by paired native undistortion; not Omega/VGGT imported poses'
 write(ctx.common/'cameras.json',data);write(ctx.common/'image_manifest.json',mapping)
 write(ctx.common/'missing_mask_views.json',dict(views=missing_masks,policy='Corrupt or absent archived masks excluded. Temporary valid camera envelope must be replaced by actualSAM3 potted-plant detection before any mesh integration.'))
 result=dict(status='complete',source=str(directory),raw_model=str(raw_model),registered_images=len(ordered),raw_images=len(list(ctx.raw.glob('*.jpg'))),sparse_points=len(xyz),valid_feature_observations=observations,raw_camera=data['source_camera'],mean_sparse_error_px=float(error.mean()),source_hashes={n:digest(sparse/n,ctx.meter) for n in ['cameras.bin','images.bin','points3D.bin']},wall_s=time.time()-start,io=ctx.meter.stats());write(marker,result);ctx.status('colmap','complete',**{k:v for k,v in result.items() if k!='status'});return result

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('plant');args=parser.parse_args();prepare(Context(args.plant))
