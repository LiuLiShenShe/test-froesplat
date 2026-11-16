"""Fuse reviewedSAM3 leaf masks, retain source indices, export12 actual leaves."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from common import *
from PIL import ImageDraw
from scipy.ndimage import binary_erosion

PALETTE=np.array([[225,55,75],[40,180,90],[40,130,230],[230,160,30],[170,80,210],[40,195,200],[230,80,190],[130,170,40],[100,100,230],[240,130,80],[70,170,150],[170,120,75]],np.uint8)

def main():
 meter=Meter();mesh=read_mesh(meter);f=np.asarray(mesh.triangles);n=len(f);sem=np.load(io.BytesIO(meter.read(OUT/'plant_only/triangle_semantics.npy')))
 ledger=read_json(OUT/'annotations/leaf_prompt_ledger.json');overrides={}
 # Multiple proposal refinements may write the same leaf/view mask. That
 # final saved mask is one observation, never multiple independent votes.
 ledger['prompts']=list({(r['leaf_id'],r['view']):r for r in ledger['prompts']}.values())
 editfile=OUT/'annotations/user_edits.json'
 if editfile.exists():
  for key,edit in read_json(editfile)['records'].items():
   if edit.get('human_confirmed') and edit.get('mask'):
    path=(OUT/edit['mask']).resolve()
    if not path.is_relative_to((OUT/'annotations/user_masks').resolve()):raise ValueError('Invalid correction mask path')
    overrides[(edit['leaf_id'],edit['view'])]=path
    ledger['prompts']=[r for r in ledger['prompts'] if (r['leaf_id'],r['view'])!=(edit['leaf_id'],edit['view'])]
    ledger['prompts'].append(dict(leaf_id=edit['leaf_id'],view=edit['view'],prompt_type='Human-confirmed UI SAM3 box/point mask',prompt_author='Human via local UI'))
 def maskpath(leaf_id,name):return overrides.get((leaf_id,name),OUT/'leaf_instances'/leaf_id/'masks'/f'{name}.png')
 scene=None
 for name in {r['view'] for r in ledger['prompts']}:
  cache=OUT/'plant_only/projection_cache'/f'{name}.npz'
  if not cache.exists():
   if scene is None:scene=cast_scene(mesh)
   ids,z,uv=cast(scene,name);npz(cache,meter,triangle_id=ids.astype('int32'),barycentric_uv=uv.astype('float16'))
 votes=np.zeros((n,12),np.uint16);coverage={i:[] for i in range(12)};origins={i:[] for i in range(12)}
 for row in ledger['prompts']:
  name=row['view'];leaf=int(row['leaf_id'].split('_')[1])-1;path=maskpath(row['leaf_id'],name)
  mask=np.array(Image.open(path))>0
  with np.load(io.BytesIO(meter.read(OUT/'plant_only/projection_cache'/f'{name}.npz'))) as d:ids=d['triangle_id']
  keep=(ids>=0)&mask;face,ct=np.unique(ids[keep],return_counts=True);votes[face[ct>=1],leaf]+=1
  coverage[leaf].append(dict(view=name,mask_pixels=int(mask.sum()),source_mesh_hit_fraction=float(keep.sum()/max(mask.sum(),1))))
  origins[leaf].append(dict(view=name,mask=str(path.relative_to(OUT)),prompt_type=row['prompt_type'],prompt_author=row['prompt_author']))
 labels=np.zeros(n,np.uint16);best=votes.argmax(1);top=votes.max(1);total=votes.sum(1)
 keep=(sem==1)&(top>=1)&(top/np.maximum(total,1)>=.6);labels[keep]=best[keep]+1
 uncertain=(sem==1)&(total>0)&~keep
 # Only unanimous mesh neighbours can fill small unsupported labelled gaps.
 from fuse_semantics import adjacency,render
 a,b=adjacency(f)
 for _ in range(2):
  counts=np.zeros((n,12),np.uint8)
  for leaf in range(12):
   np.add.at(counts[:,leaf],a,labels[b]==leaf+1);np.add.at(counts[:,leaf],b,labels[a]==leaf+1)
  num=counts.sum(1);good=(sem==1)&(labels==0)&(num>=2)&(counts.max(1)==num)
  labels[good]=counts[good].argmax(1)+1
 bfr=io.BytesIO();np.save(bfr,labels);meter.save(OUT/'leaf_instances/triangle_leaf_ids.npy',bfr.getbuffer())
 npz(OUT/'leaf_instances/triangle_leaf_support.npz',meter,view_votes=votes,ambiguous=uncertain)
 # Coverage is measured against the assigned leaf's visible surface, rather
 # than any source-mesh hit (which may belong to another occluding organ).
 for name in sorted({r['view'] for r in ledger['prompts']}):
  with np.load(io.BytesIO(meter.read(OUT/'plant_only/projection_cache'/f'{name}.npz'))) as d:ids=d['triangle_id']
  pixel_labels=np.zeros(ids.shape,np.uint16);valid=ids>=0;pixel_labels[valid]=labels[ids[valid]]
  for leaf in range(12):
   record=next((r for r in coverage[leaf] if r['view']==name),None)
   if record is None:continue
   mask=np.array(Image.open(maskpath(f'Leaf_{leaf+1:03}',name)))>0
   record['fused_instance_visible_coverage']=float(((pixel_labels==leaf+1)&mask).sum()/max(mask.sum(),1))
 catalog=[]
 for leaf in range(12):
  leaf_id=f'Leaf_{leaf+1:03}';folder=OUT/'leaf_instances'/leaf_id;faceids=np.flatnonzero(labels==leaf+1)
  leafmesh,vertices=subset_mesh(mesh,faceids);save_mesh(folder/'leaf.ply',leafmesh,meter)
  npz(folder/'source_indices.npz',meter,original_face_ids=faceids,original_vertex_ids=vertices)
  cl,cn,ca=leafmesh.cluster_connected_triangles();record=dict(leaf_id=leaf_id,status='SAM3+multi-view geometry, agent reviewed candidate group; botanical landmarks pending',faces=len(faceids),vertices=len(vertices),views=len(origins[leaf]),view_names=sorted({x['view'] for x in origins[leaf]}),components=len(cn),largest_component_face_fraction=float(max(cn)/max(1,len(faceids))) if len(cn) else 0.,leaf_surface_coverage=float(np.median([r['source_mesh_hit_fraction'] for r in coverage[leaf]])),coverage_definition='Median2D SAM3 instance pixels hit by sourceMesh; not anatomical full-surface groundtruth',coverage_by_view=coverage[leaf],pixel_sources=origins[leaf],source_vertices_and_normals_preserved=True,source_face_index_archive='source_indices.npz',source_faces_boundary_policy='Whole original triangles assigned by visible pixel votes; no unreported filling or synthetic surface',color=PALETTE[leaf].tolist())
  record['source_mesh_hit_fraction']=record['leaf_surface_coverage']
  record['leaf_surface_coverage']=float(np.median([r['fused_instance_visible_coverage'] for r in coverage[leaf]]))
  record['coverage_definition']='Median fraction of SAM3 leaf-mask pixels explained by first-visible assigned leaf triangles; visible2D coverage, not complete anatomical surface coverage'
  record['pixel_provenance_definition']='For each pixel_sources view: intersect that mask with plant_only/projection_cache/<view>.npz triangle_id; original face ID plus barycentric_uv identifies the exact source surface point'
  write(folder/'instance.json',record);catalog.append(record)
  print('LEAF_MESH',leaf_id,len(faceids),len(cn),record['leaf_surface_coverage'],flush=True)
 write(OUT/'leaf_instances/leaf_instances.json',dict(status='12candidate_instances_extracted',plant_id='KongQueZhuYu',species='孔雀竹芋',instances=catalog,all_faces_reference='OriginalTSDFPost triangle indices',unassigned_plant_faces=int(((sem==1)&(labels==0)).sum()),ambiguous_faces=int(uncertain.sum()),human_manual_masks=bool(overrides),human_confirmed_mask_overrides=len(overrides),agent_visual_candidate_selection=12))
 canvas=Image.new('RGB',(4*530,2*975),'white');draw=ImageDraw.Draw(canvas)
 for i,name in enumerate(['0000','0084','0112','0196']):
  with np.load(io.BytesIO(meter.read(OUT/'plant_only/projection_cache'/f'{name}.npz'))) as d:ids=d['triangle_id'];uv=d['barycentric_uv']
  rgbimage=np.array(rgb(name));overlay=rgbimage.copy();colored=np.zeros_like(rgbimage);valid=ids>=0;pixelids=np.zeros(ids.shape,np.uint16);pixelids[valid]=labels[ids[valid]]
  colored[valid&(pixelids==0)&(sem[np.maximum(ids,0)]==1)]=[110,110,110]
  for leaf in range(12):
   m=pixelids==leaf+1;colored[m]=PALETTE[leaf];overlay[m]=(.5*overlay[m]+.5*PALETTE[leaf]).astype('uint8')
  im1=Image.fromarray(overlay);d1=ImageDraw.Draw(im1);im2=Image.fromarray(colored);d2=ImageDraw.Draw(im2)
  for leaf in range(12):
   ys,xs=np.where(pixelids==leaf+1)
   if len(ys)>40:
    label=f'Leaf_{leaf+1:03}';point=(int(np.median(xs)),int(np.median(ys)))
    for dr in [d1,d2]:dr.text(point,label,fill='white',stroke_width=1,stroke_fill='black')
  png(OUT/'leaf_instances'/f'{name}_RGB_ID.png',np.array(im1),meter);png(OUT/'leaf_instances'/f'{name}_Mesh_ID.png',np.array(im2),meter)
  canvas.paste(im1,(i*530,22));canvas.paste(im2,(i*530,997));draw.text((i*530+4,3),name+' RGB identities',fill='black');draw.text((i*530+4,978),name+' actual triangles',fill='black')
 canvas.save(OUT/'leaf_instances/leaf_instances_overview.png')
 # Offline actual colouredMesh, without changing the complete exported PLY.
 import plotly.graph_objects as go
 vv=np.asarray(mesh.vertices);faceids=np.flatnonzero(labels>0);ff=f[faceids]
 fig=go.Figure(go.Mesh3d(x=vv[:,0],y=vv[:,1],z=vv[:,2],i=ff[:,0],j=ff[:,1],k=ff[:,2],facecolor=['rgb('+','.join(map(str,PALETTE[x-1]))+')' for x in labels[faceids]],flatshading=True))
 fig.update_layout(scene_aspectmode='data',title='Actual originalTSDF triangles: twelve SAM3 leaf IDs; full mesh, no invented surfaces')
 html=fig.to_html(include_plotlyjs=True);meter.save(OUT/'leaf_instances/leaf_instances_3d.html',html.encode())

if __name__=='__main__':main()
