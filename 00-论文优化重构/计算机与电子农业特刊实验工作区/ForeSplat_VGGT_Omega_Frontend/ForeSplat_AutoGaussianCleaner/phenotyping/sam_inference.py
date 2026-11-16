"""Existing SAM3 image/text/geometric adapter; all prompts logged explicitly."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,sys,time,io,resource,hashlib,fcntl
from common import *
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--views',default='pilot');p.add_argument('--leaves',action='store_true');p.add_argument('--correct',action='store_true');a=p.parse_args()
 lock=(PRE/'summary/gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 meter=Meter();start=time.perf_counter();cpu=time.process_time()
 sys.path.insert(0,str(ROOT/'第三方源码/SAM3-latest'));sys.path.insert(0,str(RAP))
 from sam3.model_builder import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 from sam3_prompt_adapter import append_point_prompts
 # The 3.3GiB existing weight is loaded once via a paced in-memory stream.
 original_load=torch.load;weight=ROOT/'sam3/sam3.pt';digest={}
 def load(file,*args,**kw):
  if hasattr(file,'name') and str(weight) in str(file.name):
   payload=meter.read(weight);digest['sha256']=hashlib.sha256(payload).hexdigest();return original_load(io.BytesIO(payload),*args,**kw)
  return original_load(file,*args,**kw)
 torch.load=load;torch.set_num_threads(8);torch.manual_seed(0)
 model=build_sam3_image_model(checkpoint_path=str(weight),load_from_HF=False,device='cuda',enable_inst_interactivity=True)
 torch.load=original_load
 processor=Sam3Processor(model,confidence_threshold=.3,device='cuda')
 views=[f'{n:04}' for n in range(0,210,14)] if a.views=='pilot' else ([f'{n:04}' for n in range(210)] if a.views=='all' else ([r['view'] for r in read_json(OUT/'annotations/semantic_guidance.json')['views']] if a.views=='guided' else a.views.split(',')))
 ledger=[]
 for name in views:
  folder=OUT/'semantic_masks'/name;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'corrected_complete.json').exists() and a.correct:continue
  if (folder/'complete.json').exists() and not a.leaves and not a.correct:continue
  im=rgb(name,meter)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=processor.set_image(im)
   for label,prompt in [('plant','plant leaves and stems'),('pot','flowerpot and tray'),('blue_cube','blue calibration cube')]+([('leaf','leaf')] if a.leaves else []):
    if label!='leaf' and (folder/f'{label}.png').exists():continue
    processor.reset_all_prompts(state);output=processor.set_text_prompt(prompt,state)
    masks=output['masks'][:,0].cpu().numpy();scores=output['scores'].float().cpu().numpy()
    combined=np.any(masks,axis=0) if len(masks) else np.zeros((im.height,im.width),bool)
    if label=='leaf':
     dest=OUT/'leaf_instances/candidates'/name;dest.mkdir(parents=True,exist_ok=True)
     meta=[]
     for i,(mask,score) in enumerate(zip(masks,scores)):
      if mask.sum()<30:continue
      png(dest/f'candidate_{i:03}.png',mask.astype('uint8')*255,meter)
      ys,xs=np.where(mask);meta.append(dict(candidate_id=i,score=float(score),pixels=int(mask.sum()),box=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]))
     write(dest/'candidates.json',dict(view=name,prompt=prompt,prompt_type='SAM3 text proposals; identity not assigned by text',instances=meta))
    else:png(folder/f'{label}.png',combined.astype('uint8')*255,meter)
    ledger.append(dict(view=name,semantic=label,prompt=prompt,prompt_type='text',instances=len(masks),scores=scores.tolist(),foreground_pixels=int(combined.sum()),manual_corrections=0))
    print('SAM3',name,label,len(masks),int(combined.sum()),flush=True)
   if a.correct:
    from scipy.ndimage import distance_transform_edt,binary_erosion
    rap=np.array(Image.open(PRE/'common_input/dataset/rap_masks'/f'{name}.png'))>0
    guide=np.array(Image.open(OUT/'annotations/pot_projection'/f'{name}.png'))>0
    pot=np.zeros_like(guide);prompt_record=[]
    if guide.any():
     ys,xs=np.where(guide);box=np.array([max(0,xs.min()-8),max(0,ys.min()-8),min(529,xs.max()+8),min(947,ys.max()+8)],dtype='float32')
     dist=distance_transform_edt(guide);y,x=np.unravel_index(dist.argmax(),dist.shape);points=[[float(x),float(y)]];labels=[1]
     plant_negative=rap&~guide;plant_negative[:int(box[1])+10]=False;plant_negative[int(box[3])-10:]=False
     # Negative points come from actual first-pass leaf proposals above/along
     # the rim, never guessed as semantic pot purely from text.
     firstplant=np.array(Image.open(folder/'plant.png'))>0
     candidates=firstplant&~guide;yc,xc=np.where(candidates)
     if len(yc):
      for idx in np.linspace(0,len(yc)-1,4,dtype=int):points.append([float(xc[idx]),float(yc[idx])]);labels.append(0)
     masks_i,scores_i,_=model.predict_inst(state,point_coords=np.array(points),point_labels=np.array(labels),box=box,multimask_output=True)
     masks_i=masks_i>0
     overlap=[(mask&guide).sum()/max(1,(mask|guide).sum()) for mask in masks_i]
     chosen=int(np.argmax(overlap));pot=masks_i[chosen].astype(bool)
     # Keep SAM's actual mask. A failed prompt remains flagged for review.
     prompt_record.append(dict(semantic='pot',type='SAM3 interactive box+positive/negative points',box_xyxy=box.tolist(),points_xy=points,point_labels=labels,generated_by='Historical5view pot geometry -> visible projection, automatic prompt',score=float(scores_i[chosen]),guide_iou=float(overlap[chosen]),requires_review=bool(overlap[chosen]<.5)))
    cubeinfo=read_json(OUT/'calibration/cube_corners.json');xy,z=project(np.array(cubeinfo['joint_cube_corners']),name);cube=np.zeros_like(guide)
    if (z>0).all() and xy[:,0].max()>0 and xy[:,0].min()<530 and xy[:,1].max()>0 and xy[:,1].min()<948:
     lo=np.maximum([0,0],np.floor(xy.min(0)-10).astype(int));hi=np.minimum([530,948],np.ceil(xy.max(0)+10).astype(int));crop=im.crop((*lo,*hi))
     if crop.width>4 and crop.height>4:
      cstate=processor.set_image(crop);processor.set_text_prompt('blue calibration cube',cstate)
      masks_c=cstate['masks'][:,0].cpu().numpy()
      if len(masks_c):
       center=np.clip(np.rint(xy.mean(0)-lo).astype(int),[0,0],[crop.width-1,crop.height-1])
       options=[(int(mask[center[1],center[0]]),float(cstate['scores'][i])) for i,mask in enumerate(masks_c)];ci=max(range(len(options)),key=options.__getitem__)
       cube[lo[1]:hi[1],lo[0]:hi[0]]=masks_c[ci]
      prompt_record.append(dict(semantic='blue_cube',type='SAM3 text on geometry-proposed ROI',crop_box=[*lo.tolist(),*hi.tolist()],instances=len(masks_c)))
      del cstate
    for label,mask in [('plant',(np.array(Image.open(folder/'plant.png'))>0)&rap&~pot&~cube),('pot',pot&~cube),('blue_cube',cube)]:
     old=folder/f'{label}.png';archive=folder/f'{label}_first_text.png'
     if not archive.exists():archive.write_bytes(old.read_bytes())
     png(old,mask.astype('uint8')*255,meter)
    background=~((np.array(Image.open(folder/'plant.png'))>0)|pot|cube);png(folder/'background.png',background.astype('uint8')*255,meter)
    write(folder/'corrected_complete.json',dict(status='predicted_with_geometric_prompts',view=name,prompts=prompt_record,plant_operation='Independent text plant intersect archivedRAP target envelope, explicitly subtract separate interactivepot/cube; no without-pot text assumption',new_user_clicks=0,automatic_correction_calls=len(prompt_record)))
    print('CORRECTED',name,'pot',int(pot.sum()),'cube',int(cube.sum()),flush=True)
   del state
  write(folder/'complete.json',dict(status='predicted',view=name,role='IndependentSAM3 semantic candidates; reviewed fusion still required',prompts=[r for r in ledger if r['view']==name]))
  write(OUT/'summary/sam_progress.json',dict(status='running',last_view=name,ledger=ledger))
 write(OUT/'summary'/f'sam_{a.views.replace(",","_")[:30]}_resources.json',dict(status='complete',wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30,io=meter.stats(),weight=str(weight),weight_digest=digest,local_source=str(ROOT/'第三方源码/SAM3-latest'),physical_gpu=1,ledger=ledger))

if __name__=='__main__':main()
