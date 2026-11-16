"""Prompt refinement plus actual SAM3 short-range video propagation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
from common import *
import torch,fcntl,resource,argparse
from scipy.ndimage import distance_transform_edt

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--video-only',action='store_true');parser.add_argument('--repair-nested',action='store_true');args=parser.parse_args()
 meter=Meter();start=time.perf_counter();cpu=time.process_time();lock=(PRE/'summary/gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 sys.path.insert(0,str(ROOT/'第三方源码/SAM3-latest'))
 from sam3.model_builder import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 original=torch.load;weight=ROOT/'sam3/sam3.pt'
 def load(file,*a,**kw):
  if hasattr(file,'name') and str(weight) in str(file.name):return original(io.BytesIO(meter.read(weight)),*a,**kw)
  return original(file,*a,**kw)
 torch.load=load;torch.set_num_threads(8);torch.manual_seed(0)
 model=build_sam3_image_model(checkpoint_path=str(weight),load_from_HF=False,device='cuda',enable_inst_interactivity=True);torch.load=original
 processor=Sam3Processor(model,device='cuda',confidence_threshold=.3)
 association=read_json(OUT/'leaf_instances/association_candidates.json');proposals=association['proposals'];selected=association['groups'][:12];ledger=[]
 tasks={}
 for g in selected:
  leaf_id=f"Leaf_{g['group_id']+1:03}"
  folder=OUT/'leaf_instances'/leaf_id;folder.mkdir(exist_ok=True);(folder/'masks').mkdir(exist_ok=True)
  for pi in g['proposal_ids']:tasks.setdefault(proposals[pi]['view'],[]).append((leaf_id,proposals[pi]))
 context_tasks=tasks
 if args.repair_nested:
  reduced={}
  for name,group in tasks.items():
   for leaf_id in {leaf for leaf,p in group}:
    candidates=[p for leaf,p in group if leaf==leaf_id]
    if len(candidates)>1:reduced.setdefault(name,[]).append((leaf_id,max(candidates,key=lambda p:p['pixels'])))
  tasks=reduced
 elif not args.video_only:
  tasks={name:list({leaf_id:(leaf_id,max([q for key,q in group if key==leaf_id],key=lambda q:q['pixels'])) for leaf_id,p in group}.values()) for name,group in tasks.items()}
 if args.video_only:tasks={}
 for name,group in tasks.items():
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=processor.set_image(rgb(name,meter))
   for leaf_id,p in group:
    mask=np.array(Image.open(p['mask_path']))>0;ys,xs=np.where(mask);box=np.array([xs.min()-2,ys.min()-2,xs.max()+2,ys.max()+2],dtype='float32')
    d=distance_transform_edt(mask);y,x=np.unravel_index(d.argmax(),d.shape);points=[[float(x),float(y)]];labels=[1]
    # Adjacent instance negative point from actual SAM3 proposal, if it lies
    # in the target box and outside this leaf mask.
    for other_id,q in context_tasks[name]:
     if other_id==leaf_id:continue
     neg=(np.array(Image.open(q['mask_path']))>0)&~mask
     yy,xx=np.where(neg&(np.indices(neg.shape)[1]>=box[0])&(np.indices(neg.shape)[1]<=box[2])&(np.indices(neg.shape)[0]>=box[1])&(np.indices(neg.shape)[0]<=box[3]))
     if len(yy):idx=len(yy)//2;points.append([float(xx[idx]),float(yy[idx])]);labels.append(0)
     if len(points)>=4:break
    masks,scores,_=model.predict_inst(state,point_coords=np.array(points),point_labels=np.array(labels),box=box,multimask_output=True);masks=masks>0
    ious=[(m&mask).sum()/max(1,(m|mask).sum()) for m in masks];best=int(np.argmax(ious))
    valid=bool(ious[best]>=.5);result=masks[best] if valid else mask
    png(OUT/'leaf_instances'/leaf_id/'masks'/f'{name}.png',result.astype('uint8')*255,meter)
    record=dict(leaf_id=leaf_id,view=name,box_xyxy=box.tolist(),points_xy=points,point_labels=labels,prompt_type='SAM3 native interactive box + positive point + adjacent-leaf negative points',prompt_author='automatic from SAM3 proposals after agent visual group selection',new_human_clicks=0,agent_selected_instance=True,proposal_iou=float(ious[best]),predicted_score=float(scores[best]),refinement_accepted=valid,mask_source='SAM3 prompt-refined' if valid else 'OriginalSAM3 proposal retained; refinement rejected')
    ledger.append(record);print('LEAF_REFINE',leaf_id,name,record['proposal_iou'],flush=True)
   del state
 if args.repair_nested:
  previous=read_json(OUT/'annotations/leaf_prompt_ledger.json');archive=OUT/'annotations/leaf_prompt_ledger_initial85.json'
  if not archive.exists():write(archive,previous)
  old={(r['leaf_id'],r['view']):r for r in previous['prompts']}
  for r in ledger:old[(r['leaf_id'],r['view'])]=r
  previous.update(prompts=list(old.values()),initial_prompt_calls=previous.get('initial_prompt_calls',previous['calls']),calls=previous['calls']+len(ledger),automatic_prompt_calls=previous['automatic_prompt_calls']+len(ledger),active_unique_observations=len(old),nested_proposal_repair_calls=previous.get('nested_proposal_repair_calls',0)+len(ledger),repair_rule='Same-view proposals are nested (overlap/min>=.89); largest originalSAM3 proposal re-prompted once, no duplicate view voting')
  write(OUT/'annotations/leaf_prompt_ledger.json',previous)
  write(OUT/'summary/leaf_nested_repair_resources.json',dict(wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,io=meter.stats(),physical_gpu=1,calls=len(ledger)))
  return
 # Existing image model's instance module is the official SAM3 tracker with
 # the SAME checkpoint. Share the already loaded VL backbone, as official
 # sam3_for_sam2_video_task_example.ipynb does with detector.backbone.
 tracker=model.inst_interactive_predictor.model;tracker.backbone=model.backbone
 video_dir=OUT/'annotations/short_video';video_dir.mkdir(exist_ok=True);frames=[f'{i:04}' for i in range(84,91)]
 for index,name in enumerate(frames):
  p=video_dir/f'{index:05}.jpg'
  if not p.exists():p.symlink_to(PRE/'common_input/dataset/images'/f'{name}.jpg')
 track_records=[]
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   inference=tracker.init_state(video_path=str(video_dir),offload_video_to_cpu=True,offload_state_to_cpu=True)
   for obj,leaf_id in enumerate(['Leaf_003','Leaf_008','Leaf_009'],1):
    mask=np.array(Image.open(OUT/'leaf_instances'/leaf_id/'masks/0084.png'))>0
    tracker.add_new_mask(inference_state=inference,frame_idx=0,obj_id=obj,mask=torch.as_tensor(mask,device=tracker.device))
   for frame_index,obj_ids,low,full,score in tracker.propagate_in_video(inference,start_frame_idx=0,max_frame_num_to_track=7,reverse=False,propagate_preflight=True):
    for k,obj in enumerate(obj_ids):
     leaf_id=['Leaf_003','Leaf_008','Leaf_009'][int(obj)-1];name=frames[frame_index];mask=(full[k,0]>0).cpu().numpy()
     png(OUT/'leaf_instances'/leaf_id/'video_masks'/f'{name}.png',mask.astype('uint8')*255,meter)
     track_records.append(dict(leaf_id=leaf_id,view=name,tracker_obj_id=int(obj),pixels=int(mask.sum()),identity_status='Pending3D overlap verification; trackerID alone not sufficient'))
    print('SAM3_VIDEO',frame_index,flush=True)
  del inference
  video_status='completed'
 except Exception as e:
  import traceback;traceback.print_exc();video_status='failed:'+repr(e)
 if ledger:write(OUT/'annotations/leaf_prompt_ledger.json',dict(status='completed',instances=12,key_views=len(tasks),calls=len(ledger),new_user_clicks=0,agent_visual_selections=12,automatic_prompt_calls=len(ledger),human_manual_mask_edits=0,prompts=ledger))
 write(OUT/'leaf_instances/video_propagation.json',dict(status=video_status,source='OfficialSAM3 tracker from checkpoint instance predictor, existing backbone reused',frames=frames,records=track_records,identity_rule='Validate in trueCOLMAP3D before adding to fused instance; short7frame local propagation only'))
 write(OUT/'summary'/('leaf_video_resources.json' if args.video_only else 'leaf_sam_resources.json'),dict(wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,io=meter.stats(),physical_gpu=1))

if __name__=='__main__':main()
