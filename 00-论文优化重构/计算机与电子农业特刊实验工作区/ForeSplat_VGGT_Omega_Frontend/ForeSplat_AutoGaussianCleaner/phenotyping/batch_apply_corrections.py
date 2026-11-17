"""Apply explicit human masks using fixed Leaf IDs and visible source triangles."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from batch_common import *
import shutil,subprocess

def main(ctx):
 status=read(ctx.out/'status.json')
 if any(v.get('status')=='running' for v in status['stages'].values()):raise RuntimeError('This plant is being processed; apply after its active stages finish')
 edits=read(ctx.out/'annotations/user_edits.json');confirmed=[r for r in edits['records'].values() if r.get('human_confirmed')];catalog=read(ctx.out/'leaf_instances/leaf_instances.json');byid={i['leaf_id']:i for i in catalog['instances']};unknown={r['leaf_id'] for r in confirmed}-byid.keys()
 if unknown:raise ValueError('Corrections must name an existing stable Leaf ID: '+str(unknown))
 revision=ctx.out/'annotations/revisions'/str(time.time_ns());revision.mkdir(parents=True);write(revision/'applied_edits.json',edits);mesh=ctx.mesh();metric=catalog['metric'];f=np.asarray(mesh.triangles);sem=np.load(ctx.out/'plant_only/triangle_semantics.npy');votes=np.zeros((len(f),len(byid)),np.uint16);cache=ctx.out/'plant_only/projection_cache';scene=None;ids_by_view={};leaf_ids=list(byid);overrides={(r['leaf_id'],r['view']):r for r in confirmed if r.get('mask')}
 for k,leaf in enumerate(leaf_ids):
  inst=byid[leaf];sources={s['view']:s['mask'] for s in inst['pixel_sources']}
  sources.update({view:r['mask'] for (leafid,view),r in overrides.items() if leafid==leaf})
  for view,path in sources.items():
   if view not in ids_by_view:
    if (cache/f'{view}.npz').exists():
     with np.load(cache/f'{view}.npz') as d:ids_by_view[view]=d['triangle_id']
    else:
     ids,z,uv,scene=cast(ctx,mesh,view,metric,scene);ids_by_view[view]=ids;npz(cache/f'{view}.npz',ctx.meter,triangle_id=ids,uv=uv.astype('float16'),depth=z.astype('float32'))
   ids=ids_by_view[view];mask=np.array(Image.open(ctx.out/path))>0;fid=np.unique(ids[mask&(ids>=0)]);fid=fid[sem[fid]==1];votes[fid,k]+=1
 labels=np.zeros(len(f),np.uint16)
 if len(byid):
  best=votes.max(1);total=votes.sum(1);good=(best>=2)&(best/np.maximum(total,1)>=.6);lookup=np.array([int(leaf.split('_')[1]) for leaf in leaf_ids]);labels[good]=lookup[votes[good].argmax(1)]
 exports=[]
 for leaf,inst in byid.items():
  folder=ctx.out/'leaf_instances'/leaf;fid=np.flatnonzero(labels==int(leaf.split('_')[1]));saved=revision/'leaf_instances'/leaf;saved.mkdir(parents=True)
  for name in ['leaf_cm.ply','leaf_units.ply','source_indices.npz','instance.json','landmarks.json','viewer.html']:
   p=folder/name
   if p.exists():p.rename(saved/name)
  if len(fid)<3:
   write(folder/'review_status.json',dict(status='INSUFFICIENT_MULTIVIEW_SURFACE_AFTER_HUMAN_MASKS',previous_model=str(saved),fixed_leaf_id=leaf));continue
  sub,vi=subset(mesh,fid);ctx.save_mesh(folder/f'leaf_{"cm" if metric else "units"}.ply',sub);npz(folder/'source_indices.npz',ctx.meter,original_face_ids=fid,original_vertex_ids=vi);components,counts,areas=sub.cluster_connected_triangles()
  for (leafid,view),record in overrides.items():
   if leafid==leaf:
    dest=folder/'masks'/f'{view}.png'
    if dest.is_symlink():dest.unlink()
    elif dest.exists():dest.rename(saved/f'{view}_previous_mask.png')
    link(dest,ctx.out/record['mask'])
  sources={s['view']:s for s in inst['pixel_sources']}
  sources.update({view:dict(view=view,mask=str((folder/'masks'/f'{view}.png').relative_to(ctx.out)),prompt_type='Explicit human confirmed SAM3 prompt',human_confirmed=True) for (leafid,view),record in overrides.items() if leafid==leaf})
  inst.update(faces=len(fid),vertices=len(vi),views=sorted(sources),pixel_sources=list(sources.values()),components=len(counts),largest_component_fraction=float(max(counts)/len(fid)),human_revision=str(revision),new_human_clicks=sum(len(r.get('points',[])) for r in confirmed if r['leaf_id']==leaf));write(folder/'instance.json',inst);exports.append(inst)
 npz(revision/'previous_triangle_leaf_ids.npz',ctx.meter,labels=np.load(ctx.out/'leaf_instances/triangle_leaf_ids.npy'));payload=io.BytesIO();np.save(payload,labels);ctx.meter.save(ctx.out/'leaf_instances/triangle_leaf_ids.npy',payload.getbuffer());catalog.update(instances=exports,stable_leaf_count=len(exports),human_revision=str(revision));write(ctx.out/'leaf_instances/leaf_instances.json',catalog);write(ctx.out/'leaf_instances/complete.json',catalog)
 for p in [ctx.out/'phenotype/complete.json',ctx.out/'summary/validation.json',ctx.out/'summary/report_complete.json']:
  if p.exists():p.rename(revision/(p.parent.name+'_'+p.name))
 for p in (ctx.out/'phenotype').glob('*.csv'):
  if p.name!='manual_reference.csv':(revision/'phenotype').mkdir(exist_ok=True);p.rename(revision/'phenotype'/p.name)
 for script in ['batch_measure.py','batch_manual.py','batch_report.py','batch_validate.py']:subprocess.run([str(UV),'run','--no-project','--python',str(PYTHON),str(CODE/script),ctx.plant],check=True,cwd=ROOT)
 print('Applied explicit human corrections; old derived exports preserved at',revision)

if __name__=='__main__':main(Context(sys.argv[1]))
