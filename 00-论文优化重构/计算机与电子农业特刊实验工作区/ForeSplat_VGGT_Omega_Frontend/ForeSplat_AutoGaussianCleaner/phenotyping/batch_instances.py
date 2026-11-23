"""Visible semantic votes and all stable multi-view leaf groups, no species rules."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from batch_common import *
from scipy.ndimage import binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import ImageDraw

def main(ctx):
 marker=ctx.out/'leaf_instances/complete.json'
 if marker.exists():return
 wall=time.time();ctx.status('instances','running');mesh=ctx.mesh();v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);n=len(f);mesh_info=read(ctx.out/'mesh/complete.json');metric=mesh_info['metric'];data,cameras=ctx.cameras(metric);sam=read(ctx.out/'semantic_masks/complete.json');views=sam['views'];votes=np.zeros((n,4),np.uint16);visible=np.zeros(n,np.uint16);scene=None;cachefolder=ctx.out/'plant_only/projection_cache';cachefolder.mkdir(parents=True,exist_ok=True)
 for name in views:
  cache=cachefolder/f'{name}.npz'
  if cache.exists():
   with np.load(io.BytesIO(ctx.meter.read(cache))) as d:ids=d['triangle_id'];uv=d['uv'];z=d['depth']
  else:
   ids,z,uv,scene=cast(ctx,mesh,name,metric,scene);npz(cache,ctx.meter,triangle_id=ids,uv=uv.astype('float16'),depth=z.astype('float32'))
  valid=ids>=0
  with np.load(io.BytesIO(ctx.meter.read(ctx.out/'mesh/surface/frames'/f'{name}.npz'))) as d:native=d['depth']
  tolerance=max(mesh_info['voxel_length']*2,.001);valid&=(native>0)&(np.abs(z-native)<=np.maximum(tolerance,.03*native));counts=np.bincount(ids[valid],minlength=n);visible+=counts>0;layer=np.zeros((n,4),np.uint32)
  for k,label in enumerate(['background','plant','pot','blue_cube']):
   mask=np.array(Image.open(ctx.out/'semantic_masks'/name/f'{label}.png'))>0;layer[:,k]=np.bincount(ids[valid&mask],minlength=n)
  confident=(counts>0)&(layer.max(1)/np.maximum(counts,1)>=.6);votes[np.flatnonzero(confident),layer[confident].argmax(1)]+=1
  print('SEMANTIC_VOTE',ctx.plant,name,flush=True)
 total=votes.sum(1);labels=np.full(n,4,np.uint8);best=votes.argmax(1);good=(total>=3)&(votes.max(1)/np.maximum(total,1)>=.6);labels[good]=best[good]
 centers=v[f].mean(1);inframe=np.zeros(n,np.uint16);inside=np.zeros(n,np.uint16)
 for name in views:
  xy,z=ctx.xyz_project(centers,name,metric);ij=np.rint(xy-.5).astype(int);good=(z>0)&(ij[:,0]>=0)&(ij[:,0]<data['width'])&(ij[:,1]>=0)&(ij[:,1]<data['height']);inframe+=good;mask=binary_dilation(ctx.mask(name),iterations=3);hit=np.zeros(n,bool);hit[good]=mask[ij[good,1],ij[good,0]];inside+=hit
 bad=((labels==1)|(labels==2))&((inframe<(len(views)+1)//2)|(inside/np.maximum(inframe,1)<.75));labels[bad]=4
 payload=io.BytesIO();np.save(payload,labels);ctx.meter.save(ctx.out/'plant_only/triangle_semantics.npy',payload.getbuffer());npz(ctx.out/'plant_only/semantic_votes.npz',ctx.meter,view_votes=votes,visible_views=visible,silhouette_inframe=inframe,silhouette_inside=inside)
 for k,name in [(1,'plant_only'),(2,'pot')]:
  faceids=np.flatnonzero(labels==k)
  if len(faceids):
   sub,vi=subset(mesh,faceids);ctx.save_mesh(ctx.out/'plant_only'/f'{name}_{"cm" if metric else "units"}.ply',sub);npz(ctx.out/'plant_only'/f'{name}_source_indices.npz',ctx.meter,original_face_ids=faceids,original_vertex_ids=vi)
 write(ctx.out/'plant_only/complete.json',dict(status='complete',metric=metric,unit='cm' if metric else 'model_units',plant_faces=int((labels==1).sum()),pot_faces=int((labels==2).sum()),unknown_faces=int((labels==4).sum()),original_mesh=mesh_info['mesh'],visibility='First ray intersection, consistent nativeGGGS depth, >=3view semantic majority; >=half cameras inframe and target silhouette support>=.75',largest_component_filter=False))
 return associate(ctx, mesh, labels, wall=wall)

def associate(ctx,mesh,labels,*,organ_dir='leaf_instances',id_prefix='Leaf',wall=None):
 wall=time.time() if wall is None else wall;folder_base=ctx.out/organ_dir;marker=folder_base/'complete.json';mesh_info=read(ctx.out/'mesh/complete.json');metric=mesh_info['metric'];cachefolder=ctx.out/'plant_only/projection_cache';v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);n=len(f);centers=v[f].mean(1)
 proposals=read(folder_base/'candidates.json')['proposals'];supported=[];areas=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
 for p in proposals:
  with np.load(io.BytesIO(ctx.meter.read(cachefolder/f"{p['view']}.npz"))) as d:ids=d['triangle_id']
  mask=np.array(Image.open(ctx.out/p['mask']))>0;fid=np.unique(ids[mask&(ids>=0)]);fid=fid[labels[fid]==1]
  if len(fid)<3:continue
  center=np.average(centers[fid],axis=0,weights=np.maximum(areas[fid],1e-12));spread=float(np.quantile(np.linalg.norm(centers[fid]-center,axis=1),.9));supported.append(dict(**p,faces=fid,center=center,spread=spread,area=float(areas[fid].sum())))
 # Candidate graph accepts shared *visible original triangles*. Same-view
 # disjoint proposals can never merge through a chain of occluded views.
 parent=list(range(len(supported)));members={i:[i] for i in parent}
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 links=[]
 for i,p in enumerate(supported):
  for j in range(i):
   q=supported[j]
   if p['view']==q['view']:continue
   if np.linalg.norm(p['center']-q['center'])>max(p['spread'],q['spread'])*.8:continue
   common=np.intersect1d(p['faces'],q['faces'],assume_unique=True);overlap=float(areas[common].sum()/max(1e-12,min(p['area'],q['area'])))
   if overlap>=.35:links.append((overlap,i,j))
 ambiguous=[]
 for overlap,i,j in sorted(links,reverse=True):
  a,b=root(i),root(j)
  if a==b:continue
  conflict=False
  for pi in members[a]:
   for qi in members[b]:
    p,q=supported[pi],supported[qi]
    if p['view']==q['view']:
     common=np.intersect1d(p['faces'],q['faces'],assume_unique=True);ratio=areas[common].sum()/max(1e-12,min(p['area'],q['area']))
     if ratio<.5:conflict=True
  if conflict:ambiguous.append(dict(candidate_a=i,candidate_b=j,shared_fraction=overlap,reason='Would merge separate same-view leaf candidates'));continue
  parent[b]=a;members[a]+=members.pop(b)
 groups=[group for group in members.values() if len({supported[i]['view'] for i in group})>=3];groups.sort(key=lambda g:(-len({supported[i]['view'] for i in g}),-sum(supported[i]['area'] for i in g),min(g)))
 leafvotes=np.zeros((n,len(groups)),np.uint16)
 for leaf,g in enumerate(groups):
  perview={}
  for i in g:perview.setdefault(supported[i]['view'],[]).append(i)
  for name,indices in perview.items():leafvotes[np.unique(np.concatenate([supported[i]['faces'] for i in indices])),leaf]+=1
 leaflabels=np.zeros(n,np.uint16)
 if len(groups):
  top=leafvotes.max(1);total=leafvotes.sum(1);good=(labels==1)&(top>=2)&(top/np.maximum(total,1)>=.6);leaflabels[good]=leafvotes[good].argmax(1)+1
 exports=[]
 for index,g in enumerate(groups):
  leaf=index+1;fid=np.flatnonzero(leaflabels==leaf)
  if len(fid)<3:continue
  leafid=f'{id_prefix}_{leaf:03}';folder=folder_base/leafid;folder.mkdir(exist_ok=True);sub,vi=subset(mesh,fid);ctx.save_mesh(folder/f'leaf_{"cm" if metric else "units"}.ply',sub);npz(folder/'source_indices.npz',ctx.meter,original_face_ids=fid,original_vertex_ids=vi);component,counts,aa=sub.cluster_connected_triangles();perview={}
  for pi in g:
   p=supported[pi];name=p['view']
   if name not in perview or p['area']>supported[perview[name]]['area']:perview[name]=pi
  sources=[]
  for name,pi in perview.items():
   original=ctx.out/supported[pi]['mask'];dest=folder/'masks'/f'{name}.png';link(dest,original);sources.append(dict(view=name,mask=str(dest.relative_to(ctx.out)),candidate=pi,prompt_type=supported[pi]['prompt_type']))
  reference=max(perview,key=lambda name:supported[perview[name]]['area']);image=ctx.image(reference);mask=np.array(Image.open(folder/'masks'/f'{reference}.png'))>0;arr=np.array(image);png(folder/'reference_rgb.png',arr,ctx.meter);arr[mask]=(.65*arr[mask]+.35*np.array([50,210,100])).astype('uint8');png(folder/'reference.png',arr,ctx.meter)
  record=dict(leaf_id=leafid,plant=ctx.plant,metric=metric,unit='cm' if metric else 'model_units',faces=len(fid),vertices=len(vi),views=sorted(perview),reference_view=reference,components=len(counts),largest_component_fraction=float(max(counts)/len(fid)),pixel_sources=sources,source_mesh=mesh_info['mesh'],source_indices='source_indices.npz',source_coordinates_and_normals_preserved=True,stability_rule='At least3views candidate association, assigned triangles >=2distinctview support and >=.6instance agreement',geometry_status='OBSERVED_PARTIAL_SURFACE_UNCONFIRMED_ANATOMICAL_LANDMARKS',new_human_clicks=0,agent_visual_selections=0);write(folder/'instance.json',record);exports.append(record);print('LEAF_EXPORT',ctx.plant,leafid,len(fid),flush=True)
 payload=io.BytesIO();np.save(payload,leaflabels);ctx.meter.save(folder_base/'triangle_leaf_ids.npy',payload.getbuffer());npz(folder_base/'leaf_votes.npz',ctx.meter,view_votes=leafvotes);write(folder_base/'ambiguities.json',dict(ambiguous_links=ambiguous,groups_requiring_confirmation=True))
 result=dict(status='complete',plant=ctx.plant,metric=metric,instances=exports,stable_leaf_count=len(exports),proposals=len(proposals),supported_proposals=len(supported),unassigned_plant_faces=int(((labels==1)&(leaflabels==0)).sum()),identity_not_all_biological_leaves=True,no_fixed_count_limit=True,human_manual_annotation=False,wall_s=time.time()-wall,io=ctx.meter.stats());write(folder_base/'leaf_instances.json',result);write(marker,result);ctx.status('instances' if organ_dir=='leaf_instances' else 'compound','complete',stable_leaf_count=len(exports),plant_faces=int((labels==1).sum()),pot_faces=int((labels==2).sum()));return result

if __name__=='__main__':main(Context(sys.argv[1]))
