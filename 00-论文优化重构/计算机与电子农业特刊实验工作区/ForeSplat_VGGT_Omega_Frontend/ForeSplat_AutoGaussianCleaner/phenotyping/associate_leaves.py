"""SAM3 proposals associated by first-hit mesh triangles, with ambiguity records."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from common import *
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from PIL import ImageDraw

def main():
 meter=Meter();mesh=read_mesh(meter);v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);centroids=v[f].mean(1);area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
 proposals=[];keyviews=['0000','0028','0056','0084','0112','0140','0168','0196']
 for name in keyviews:
  folder=OUT/'leaf_instances/candidates'/name
  with np.load(io.BytesIO(meter.read(OUT/'plant_only/projection_cache'/f'{name}.npz'))) as d:ids=d['triangle_id']
  rap=np.array(Image.open(PRE/'common_input/dataset/rap_masks'/f'{name}.png'))>0
  records=read_json(folder/'candidates.json')['instances'];accepted=[]
  for row in sorted(records,key=lambda x:-x['score']):
   original=np.array(Image.open(folder/f"candidate_{row['candidate_id']:03}.png"))>0
   mask=original&rap
   if mask.sum()<300 or mask.sum()>rap.sum()*.42 or mask.sum()/max(1,original.sum())<.7:continue
   if any((mask&m).sum()/max(1,(mask|m).sum())>.65 for m in accepted):continue
   accepted.append(mask);keep=(ids>=0)&binary_erosion(mask,iterations=1);faces,counts=np.unique(ids[keep],return_counts=True);faces=faces[counts>=1]
   if len(faces)<40:continue
   center=np.average(centroids[faces],axis=0,weights=area[faces]);spread=np.quantile(np.linalg.norm(centroids[faces]-center,axis=1),.9)
   proposals.append(dict(view=name,candidate_id=row['candidate_id'],score=row['score'],mask_path=str(folder/f"candidate_{row['candidate_id']:03}.png"),pixels=int(mask.sum()),faces=faces,center=center,spread=float(spread)))
 n=len(proposals);links=[]
 for i in range(n):
  for j in range(i):
   a,b=proposals[i],proposals[j]
   if a['view']==b['view']:continue
   if np.linalg.norm(a['center']-b['center'])>max(a['spread'],b['spread'])*.8:continue
   common=np.intersect1d(a['faces'],b['faces'],assume_unique=True);overlap=float(area[common].sum());small=min(float(area[a['faces']].sum()),float(area[b['faces']].sum()))
   fraction=overlap/max(small,1e-12)
   if fraction>.35:links.append((fraction,i,j))
 parent=list(range(n));members={i:{i} for i in range(n)};ambiguous=[]
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for score,i,j in sorted(links,reverse=True):
  a,b=root(i),root(j)
  if a==b:continue
  # Two disjoint leaves in one keyframe cannot be collapsed by a broad mask
  # in another camera. This prevents common transitive merging failures.
  conflict=False
  for ai in members[a]:
   for bi in members[b]:
    if proposals[ai]['view']==proposals[bi]['view']:
     common=np.intersect1d(proposals[ai]['faces'],proposals[bi]['faces']);den=min(len(proposals[ai]['faces']),len(proposals[bi]['faces']))
     if len(common)/max(den,1)<.5:conflict=True
  if conflict:ambiguous.append(dict(proposal_pair=[i,j],reason='same-view distinct masks would merge',overlap_score=score));continue
  parent[b]=a;members[a]|=members.pop(b)
 groups=[]
 for ids in members.values():
  support={}
  for i in ids:
   for face in proposals[i]['faces']:support[int(face)]=support.get(int(face),0)+1
  fs=np.array(list(support));center=np.average(centroids[fs],axis=0,weights=area[fs])
  groups.append(dict(proposal_ids=sorted(ids),views=sorted({proposals[i]['view'] for i in ids}),surface_area=float(area[fs].sum()),center=center,faces=fs,support=np.array([support[int(face)] for face in fs])))
 groups.sort(key=lambda x:(-len(x['views']),-x['surface_area']))
 archive={}
 for i,p in enumerate(proposals):archive[f'proposal_{i}_faces']=p['faces']
 for i,g in enumerate(groups):archive[f'group_{i}_faces']=g['faces'];archive[f'group_{i}_support']=g['support']
 npz(OUT/'leaf_instances/association_support.npz',meter,**archive)
 plain=lambda d:{k:(value.tolist() if isinstance(value,np.ndarray) else value) for k,value in d.items() if k not in ['faces','support']}
 write(OUT/'leaf_instances/association_candidates.json',dict(status='geometry_associated_candidates_need_review',key_views=keyviews,proposals=[plain(p) for p in proposals],groups=[dict(group_id=i,**plain(g)) for i,g in enumerate(groups)],ambiguous_links=ambiguous,identity_rule='Triangle first-hit co-visibility and overlap; no2D trackingID or colour alone establishes identity',new_user_clicks=0,agent_geometric_selection_count=0))
 canvas=Image.new('RGB',(4*530,2*990),'white');draw=ImageDraw.Draw(canvas)
 palette=np.array([[int((i*97+80)%255),int((i*157+80)%255),int((i*53+120)%255)] for i in range(len(groups))])
 for k,name in enumerate(keyviews):
  im=np.array(rgb(name)).astype(float);overlay=im.copy();dtext=[]
  for gid,g in enumerate(groups[:25]):
   for pi in g['proposal_ids']:
    p=proposals[pi]
    if p['view']!=name:continue
    m=np.array(Image.open(p['mask_path']))>0;overlay[m]=.55*overlay[m]+.45*palette[gid];y,x=np.where(m);dtext.append((int(np.median(x)),int(np.median(y)),f'G{gid}'))
  tile=Image.fromarray(overlay.astype('uint8'));td=ImageDraw.Draw(tile)
  for x,y,label in dtext:td.text((x,y),label,fill='white',stroke_width=1,stroke_fill='black')
  x=(k%4)*530;y=(k//4)*990;canvas.paste(tile,(x,y+28));draw.text((x+4,y+5),name,fill='black')
 canvas.save(OUT/'review/leaf_association_candidates.png')
 print('LEAF_ASSOC',n,'proposals',len(groups),'groups',[(len(g['views']),round(g['surface_area'],3)) for g in groups[:20]],flush=True)

if __name__=='__main__':main()
