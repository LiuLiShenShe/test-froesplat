"""Same SAM3 prompts for all species; independent cube masks before calibration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from batch_common import *
import fcntl,resource
from scipy.ndimage import distance_transform_edt

def historical(ctx,name,label):
 path=ctx.annotation/f'{name}.json'
 if not path.exists():return None
 from PIL import ImageDraw
 record=read(path);polygons=[s['points'] for s in record.get('shapes',[]) if s['label']==label]
 if not polygons:return None
 annotated_image=ctx.annotation/Path(record.get('imagePath',f'{name}.jpg')).name
 current_image=ctx.raw/f'{name}.jpg'
 if not annotated_image.exists() or not current_image.exists():return None
 if digest(annotated_image,ctx.meter)!=digest(current_image,ctx.meter):return None
 data,_=ctx.cameras();raw=data['source_camera'];w,h=raw['width'],raw['height'];im=Image.new('L',(w,h));draw=ImageDraw.Draw(im)
 for points in polygons:draw.polygon([tuple(p) for p in points],fill=255)
 sys.path.insert(0,str(FRONT.parent/'阶段二十E0_三维全视图与输入冻结/scripts'));from build_datasets import inverse_maps,remap_nearest
 import pycolmap
 target=pycolmap.Camera(model='PINHOLE',width=data['width'],height=data['height'],params=[data['K'][0][0],data['K'][1][1],data['K'][0][2],data['K'][1][2]])
 source=pycolmap.Camera(**raw);mx,my,valid=inverse_maps(source,target);return remap_nearest(np.array(im)>0,mx,my,valid)

def main(ctx):
 marker=ctx.out/'semantic_masks/complete.json'
 if marker.exists():return
 data,cameras=ctx.cameras();names=sorted(cameras);keyviews=[names[i] for i in np.unique(np.linspace(0,len(names)-1,16,dtype=int))];missing=read(ctx.common/'missing_mask_views.json')['views'] if (ctx.common/'missing_mask_views.json').exists() else [];selected=sorted(set(names[::3]+keyviews+missing+[p.stem for p in ctx.annotation.glob('*.json') if p.stem in cameras]));ctx.status('sam3','running',selected_views=len(selected))
 lock=(PRE/'summary/gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);wall=time.time();cpu=time.process_time()
 import torch
 torch.set_num_threads(8);torch.manual_seed(0);sys.path.insert(0,str(ROOT/'第三方源码/SAM3-latest'));from sam3.model_builder import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 original=torch.load;weight=ROOT/'sam3/sam3.pt'
 def load(path,*args,**kwargs):
  if hasattr(path,'name') and str(weight) in str(path.name):return original(io.BytesIO(ctx.meter.read(weight)),*args,**kwargs)
  return original(path,*args,**kwargs)
 torch.load=load
 try:model=build_sam3_image_model(checkpoint_path=str(weight),load_from_HF=False,device='cuda',enable_inst_interactivity=True)
 finally:torch.load=original
 processor=Sam3Processor(model,device='cuda',confidence_threshold=.3);allproposals=[];ledger=[]
 for name in selected:
  folder=ctx.out/'semantic_masks'/name;done=folder/'complete.json'
  if done.exists():
   old=read(done);ledger.extend(old['prompts']);allproposals.extend(old.get('leaf_proposals',[]));continue
  image=ctx.image(name);envelope=ctx.mask(name);masks={};prompts=[]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=processor.set_image(image)
   def text(prompt):
    processor.reset_all_prompts(state);res=processor.set_text_prompt(prompt=prompt,state=state);mm=res['masks'].squeeze(1).cpu().numpy()>0 if len(res['masks']) else np.empty((0,data['height'],data['width']),bool);ss=res['scores'].float().cpu().numpy();return mm,ss
   if name in missing:
    mm,ss=text('potted plant');envelope=mm[int(np.argmax(ss))] if len(mm) else np.zeros(envelope.shape,bool);png(ctx.common/'rap_masks'/f'{name}.png',envelope.astype('uint8')*255,ctx.meter);prompts.append(dict(view=name,label='target_envelope',type='SAM3 replacement for corrupt/absent archived mask',historical_assisted=False,foreground_pixels=int(envelope.sum()),requires_review=bool(not envelope.any())))
   for label,prompt in [('plant','plant leaves and stems'),('pot','flowerpot and tray'),('blue_cube','blue calibration cube')]:
    mm,ss=text(prompt);masks[label]=(mm.any(0) if len(mm) else np.zeros(envelope.shape,bool));prompts.append(dict(view=name,label=label,type='SAM3 independent text',text=prompt,instances=len(mm),scores=ss.tolist(),historical_assisted=False))
    if label=='pot':pot_candidates=mm;pot_scores=ss
   # Box+point is an actual independent SAM3 prediction. A historical mask,
   # where available, is a guide rather than the exported segmentation.
   guide=historical(ctx,name,'pot');assisted=guide is not None
   if guide is None:
    ys,xs=np.where(envelope);guide=np.zeros(envelope.shape,bool)
    if len(ys):
     bottom=ys.min()+.62*(ys.max()-ys.min());guide=envelope&(np.indices(envelope.shape)[0]>=bottom)
    eligible=[]
    for j,mask in enumerate(pot_candidates):
     yy,xx=np.where(mask&envelope)
     if len(yy) and mask.sum()<envelope.sum()*.65:eligible.append((float(pot_scores[j])+(mask&guide).sum()/max(1,mask.sum()),j))
    if eligible:guide=pot_candidates[max(eligible)[1]]&envelope
   if guide.any():
    yy,xx=np.where(guide);box=np.array([max(0,xx.min()-3),max(0,yy.min()-3),min(data['width']-1,xx.max()+3),min(data['height']-1,yy.max()+3)],float);dist=distance_transform_edt(guide);y,x=np.unravel_index(dist.argmax(),dist.shape);points=[[float(x),float(y)]];labels=[1]
    neg=masks['plant']&~guide&envelope;yy,xx=np.where(neg)
    for k in np.linspace(0,max(0,len(yy)-1),min(4,len(yy)),dtype=int):points.append([float(xx[k]),float(yy[k])]);labels.append(0)
    native,scores,_=model.predict_inst(state,point_coords=np.array(points),point_labels=np.array(labels),box=box,multimask_output=True);native=native>0;ious=np.array([(m&guide).sum()/max(1,(m|guide).sum()) for m in native]);best=int(ious.argmax());masks['pot']=native[best]&envelope;prompts.append(dict(view=name,label='pot',type='SAM3 box+positive/negative points',box=box.tolist(),points=points,point_labels=labels,historical_assisted=assisted,guide_iou=float(ious[best]),requires_review=bool(ious[best]<.5),human_new_clicks=0))
   cube_assisted=False
   # Missing small-object detection: historical ROI is optional and declared.
   if masks['blue_cube'].sum()<12:
    cube_guide=historical(ctx,name,'blue_cube')
    if cube_guide is not None and cube_guide.any():
     yy,xx=np.where(cube_guide);bounds=[max(0,xx.min()-12),max(0,yy.min()-12),min(data['width'],xx.max()+13),min(data['height'],yy.max()+13)];cropstate=processor.set_image(image.crop(bounds));processor.reset_all_prompts(cropstate);res=processor.set_text_prompt(prompt='blue calibration cube',state=cropstate)
     if len(res['masks']):masks['blue_cube'][bounds[1]:bounds[3],bounds[0]:bounds[2]]=res['masks'].squeeze(1).any(0).cpu().numpy()>0
     cube_assisted=True;prompts.append(dict(view=name,label='blue_cube',type='SAM3 independent ROI detection with historical boundary ROI',historical_assisted=True,crop=bounds))
   if masks['blue_cube'].sum()<12:
    # A generic blue-colour proposal makes small cubes visible to SAM3.
    # Colour locates an ROI only; the exported mask is an actual SAM3 result.
    import cv2
    from scipy.ndimage import label as components
    hsv=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2HSV);blue=(hsv[:,:,0]>=90)&(hsv[:,:,0]<=140)&(hsv[:,:,1]>=80)&(hsv[:,:,2]>=35);cc,num=components(blue);sizes=np.bincount(cc.ravel());options=[]
    for ci in np.argsort(-sizes[1:])[:8]+1:
     yy,xx=np.where(cc==ci)
     if len(xx)<8 or len(xx)>blue.size*.035:continue
     bw,bh=xx.max()-xx.min()+1,yy.max()-yy.min()+1
     if not .25<bw/bh<4:continue
     pad=max(24,2*max(bw,bh));bounds=[max(0,int(xx.min()-pad)),max(0,int(yy.min()-pad)),min(data['width'],int(xx.max()+pad+1)),min(data['height'],int(yy.max()+pad+1))];cs=processor.set_image(image.crop(bounds));processor.reset_all_prompts(cs);out=processor.set_text_prompt(prompt='blue cube',state=cs)
     for cmask,cscore in zip(out['masks'],out['scores']):
      cmask=cmask.squeeze().cpu().numpy()>0;full=np.zeros_like(blue);full[bounds[1]:bounds[3],bounds[0]:bounds[2]]=cmask;pixels=int(full.sum());overlap=float((full&blue).sum()/max(1,pixels))
      if 12<=pixels<=blue.size*.035 and overlap>.2:options.append((float(cscore)+overlap,full,bounds,float(cscore),overlap))
    if options:
     score,mask,bounds,confidence,overlap=max(options,key=lambda o:o[0]);masks['blue_cube']=mask;cube_assisted=False;prompts.append(dict(view=name,label='blue_cube',type='SAM3 text on automatic blue-colour ROI',text='blue cube',crop=bounds,score=confidence,blue_fraction=overlap,historical_assisted=False,colour_used_for_scale=False))
   masks['plant']&=envelope&~masks['pot']&~masks['blue_cube'];masks['pot']&=~masks['blue_cube'];masks['background']=~(masks['plant']|masks['pot']|masks['blue_cube'])
   proposals=[]
   if name in keyviews:
    mm,ss=text('leaf');candidates=[]
    for j in np.argsort(-ss):
     mask=mm[j];area=int(mask.sum());fraction=float((mask&envelope).sum()/max(1,area));plantoverlap=float((mask&masks['plant']).sum()/max(1,area))
     if ss[j]<.4 or area<max(35,envelope.sum()*.0005) or area>envelope.sum()*.5 or fraction<.75 or plantoverlap<.55:continue
     if any((mask&old).sum()/max(1,min(area,old.sum()))>.8 for old in candidates):continue
     candidates.append(mask)
     # Native geometric prompting refines each candidate; neighboring text
     # instances supply negative points without a species-specific rule.
     yy,xx=np.where(mask);box=np.array([max(0,xx.min()-2),max(0,yy.min()-2),min(data['width']-1,xx.max()+2),min(data['height']-1,yy.max()+2)],float);d=distance_transform_edt(mask);y,x=np.unravel_index(d.argmax(),d.shape);points=[[float(x),float(y)]];labels=[1]
     for other in mm:
      neg=other&~mask;yy,xx=np.where(neg&(np.indices(neg.shape)[1]>=box[0])&(np.indices(neg.shape)[1]<=box[2])&(np.indices(neg.shape)[0]>=box[1])&(np.indices(neg.shape)[0]<=box[3]))
      if len(yy):k=len(yy)//2;points.append([float(xx[k]),float(yy[k])]);labels.append(0)
      if len(points)>=4:break
     native,scores,_=model.predict_inst(state,point_coords=np.array(points),point_labels=np.array(labels),box=box,multimask_output=True);native=native>0;ious=[(m&mask).sum()/max(1,(m|mask).sum()) for m in native];best=int(np.argmax(ious));refined=native[best] if ious[best]>=.5 else mask
     path=ctx.out/'leaf_instances/candidates'/name/f'candidate_{len(proposals):03}.png';png(path,refined.astype('uint8')*255,ctx.meter);proposals.append(dict(view=name,candidate_id=len(proposals),mask=str(path.relative_to(ctx.out)),score=float(ss[j]),box=box.tolist(),points=points,point_labels=labels,prompt_type='SAM3 leaf text proposal plus native box/positive/negative prompts',human_new_clicks=0,agent_visual_selection=False,refinement_iou=float(ious[best])))
   for label,mask in masks.items():png(folder/f'{label}.png',mask.astype('uint8')*255,ctx.meter)
  record=dict(status='complete',view=name,prompts=prompts,leaf_proposals=proposals,cube_historical_assisted=cube_assisted);write(done,record);ledger.extend(prompts);allproposals.extend(proposals);print('SAM3_BATCH',ctx.plant,name,'leaf_candidates',len(proposals),flush=True)
 write(ctx.out/'leaf_instances/candidates.json',dict(proposals=allproposals,key_views=keyviews,no_fixed_leaf_limit=True,human_clicks=0,semantic_guidance='Historical annotation where available; automatically proposed pot geometry otherwise'))
 result=dict(status='complete',views=selected,key_views=keyviews,prompts=ledger,leaf_candidates=len(allproposals),checkpoint=str(weight),checkpoint_sha256=read(OLD/'summary/sam_pilot_resources.json')['weight_digest']['sha256'],model_source=str(ROOT/'第三方源码/SAM3-latest'),physical_gpu=1,wall_s=time.time()-wall,cpu_s=time.process_time()-cpu,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,io=ctx.meter.stats());write(marker,result);ctx.status('sam3','complete',views=len(selected),leaf_candidates=len(allproposals))

if __name__=='__main__':main(Context(sys.argv[1]))
