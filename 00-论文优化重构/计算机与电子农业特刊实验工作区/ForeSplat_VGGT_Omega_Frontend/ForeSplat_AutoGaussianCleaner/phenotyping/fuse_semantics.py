"""Occlusion-aware2D semantic votes on immutable source triangle IDs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
from common import *
from PIL import ImageDraw
from scipy.ndimage import binary_dilation

def adjacency(f):
 edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);owners=np.tile(np.arange(len(f)),3);edges.sort(axis=1)
 order=np.lexsort((edges[:,1],edges[:,0]));e=edges[order];same=np.all(e[1:]==e[:-1],axis=1);a,b=owners[order[:-1][same]],owners[order[1:][same]]
 return a,b

def render(mesh,name,ids,uv):
 f=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors);image=np.zeros((*ids.shape,3),np.uint8);keep=ids>=0
 ff=f[ids[keep]];u=uv[keep];w=np.c_[1-u.sum(1),u]
 image[keep]=np.clip((colors[ff]*w[:,:,None]).sum(1)*255,0,255).astype('uint8')
 return image

def main():
 meter=Meter();mesh=read_mesh(meter);faces=np.asarray(mesh.triangles);n=len(faces);counts=np.zeros((n,4),np.uint16);pixels=np.zeros((n,4),np.uint32);visible=np.zeros(n,np.uint16);viewlog=[]
 views=[r['view'] for r in read_json(OUT/'annotations/semantic_guidance.json')['views']]
 pending=[name for name in views if not (OUT/'semantic_masks'/name/'corrected_complete.json').exists()]
 if pending:raise RuntimeError(f'Semantics not complete: {len(pending)} views pending, e.g. {pending[:5]}')
 for name in views:
  with np.load(io.BytesIO(meter.read(OUT/'plant_only/projection_cache'/f'{name}.npz'))) as d:ids=d['triangle_id'];uv=d['barycentric_uv']
  valid=ids>=0;allpixels=np.bincount(ids[valid],minlength=n);vis=allpixels>0;visible+=vis.astype('uint16')
  # Reject remote TSDF fragments that merely project into the semantic mask.
  # Their sourceGaussian camera-Z must agree with this actual first-hit point.
  with np.load(io.BytesIO(meter.read(PRE/'mesh/GGGS_surface/frames'/f'{name}.npz'))) as depth_frame:native_z=depth_frame['depth']
  xy=uv[valid].astype(float);ff=faces[ids[valid]];weights=np.c_[1-xy.sum(1),xy]
  world=(np.asarray(mesh.vertices)[ff]*weights[:,:,None]).sum(1)
  mesh_z=world@CAMERAS[name]['R'][2]+CAMERAS[name]['t'][2]
  reference=native_z[valid];agreement=(reference>0)&(np.abs(mesh_z-reference)<=np.maximum(.05,reference*.03))
  depth_valid=np.zeros_like(valid);depth_valid[valid]=agreement;valid&=depth_valid
  allpixels=np.bincount(ids[valid],minlength=n);vis=allpixels>0
  masks=[np.array(Image.open(OUT/'semantic_masks'/name/f'{key}.png'))>0 for key in ['background','plant','pot','blue_cube']]
  layer=np.zeros((n,4),np.uint32)
  for label,mask in enumerate(masks):layer[:,label]=np.bincount(ids[valid&mask],minlength=n)
  winner=layer.argmax(1);confidence=layer.max(1)/np.maximum(allpixels,1);keep=vis&(confidence>=.6);counts[np.flatnonzero(keep),winner[keep]]+=1;pixels+=layer
  viewlog.append(dict(view=name,visible_faces=int(vis.sum()),depth_consistent_pixels=int(valid.sum()),depth_rejected_pixels=int((ids>=0).sum()-valid.sum()),confident_semantic_faces=int(keep.sum()),pixel_conflicts=int(np.any(np.stack(masks[1:]),axis=0).sum()-sum(m.sum() for m in masks[1:]))))
  print('SEMANTIC_FUSE',name,flush=True)
 support=counts.sum(1);winner=counts.argmax(1);confidence=counts.max(1)/np.maximum(support,1)
 labels=np.full(n,4,np.uint8);labels[(support>=3)&(confidence>=.6)]=winner[(support>=3)&(confidence>=.6)]
 # A single supporting view is retained as uncertain, except multi-pixel
 # faces corroborated by independent historical semantic visibility.
 historical=np.load(io.BytesIO(meter.read(OUT/'annotations/geometry_guidance.npz')))
 for label,key in [(1,'plant_support'),(2,'pot_support')]:
  old=historical[key];keep=(labels==4)&(old>=2)&(counts[:,label]>=2);labels[keep]=label
 a,b=adjacency(faces);propagated=0
 for _ in range(2):
  neigh=np.zeros((n,4),np.uint8)
  for label in range(4):
   np.add.at(neigh[:,label],a,labels[b]==label);np.add.at(neigh[:,label],b,labels[a]==label)
  top=neigh.argmax(1);total=neigh.sum(1);keep=(labels==4)&(total>=2)&(neigh.max(1)==total)&(visible>=1)
  propagated+=int(keep.sum());labels[keep]=top[keep]
 # Independent target-silhouette space-carving test. It does not assign the
 # front surface label to hidden triangles; it flags a remote fragment that
 # lies outside the target's outer silhouette in most other cameras.
 centers=np.asarray(mesh.vertices)[faces].mean(1);inframe=np.zeros(n,np.uint16);inside=np.zeros(n,np.uint16)
 for name in views:
  xy,z=project(centers,name);pixels_xy=np.rint(xy-.5).astype(int);good=(z>0)&(pixels_xy[:,0]>=0)&(pixels_xy[:,0]<530)&(pixels_xy[:,1]>=0)&(pixels_xy[:,1]<948);inframe+=good
  envelope=np.array(Image.open(PRE/'common_input/dataset/rap_masks'/f'{name}.png'))>0;envelope=binary_dilation(envelope,iterations=3)
  hits=np.zeros(n,bool);hits[good]=envelope[pixels_xy[good,1],pixels_xy[good,0]];inside+=hits
 hull_agreement=inside/np.maximum(inframe,1);minimum_inframe=(len(views)+1)//2
 rejected=((labels==1)|(labels==2))&((inframe<minimum_inframe)|(hull_agreement<.75));labels[rejected]=4
 output=OUT/'plant_only';output.mkdir(exist_ok=True)
 bfr=io.BytesIO();np.save(bfr,labels);meter.save(output/'triangle_semantics.npy',bfr.getbuffer())
 npz(output/'semantic_votes.npz',meter,view_support=counts,pixel_support=pixels,visible_views=visible,consistency=confidence.astype('float32'),silhouette_inframe_views=inframe,silhouette_supported_views=inside,visual_hull_agreement=hull_agreement.astype('float32'))
 outputs={}
 for label,name in [(1,'plant_only'),(2,'pot'),(3,'reference_cube')]:
  face_ids=np.flatnonzero(labels==label)
  if not len(face_ids):outputs[name]=dict(status='No supported source triangles; no synthetic mesh exported');continue
  extracted,vertices=subset_mesh(mesh,face_ids);save_mesh(output/f'{name}.ply',extracted,meter)
  npz(output/f'{name}_source_indices.npz',meter,original_face_ids=face_ids,original_vertex_ids=vertices)
  outputs[name]=dict(status='exported',faces=len(face_ids),vertices=len(vertices),path=str(output/f'{name}.ply'))
 write(output/'triangle_semantics.json',dict(status='completed',label_map={'0':'background','1':'plant','2':'pot','3':'blue_cube','4':'uncertain'},original_mesh=str(mesh_path()),original_faces=n,counts={str(i):int((labels==i).sum()) for i in range(5)},views=views,view_votes=viewlog,visibility='Only first ray intersection receives pixel label; occluded geometry cannot receive the front pixel label',fusion='At least3 confidentperview votes, >=.6 agreement; sourceGaussian depth within max(.05u,3%Z); historical2view fallback;2 rounds unanimous adjacency propagation for unknowns only',visual_hull='Target RAP envelope dilated3pixels; centroid in frame in at least half of selected cameras and silhouette support>=75%; remote uncertain fragments retained in source, excluded from traits',minimum_inframe_views=minimum_inframe,visual_hull_rejected_faces=int(rejected.sum()),adjacency_filled_faces=propagated,outputs=outputs,source_geometry_unchanged=True,largest_component_filter=False,io=meter.stats()))
 canvas=Image.new('RGB',(4*530,2*980),'white');draw=ImageDraw.Draw(canvas)
 plant=read_mesh(meter) # still the immutable mesh, for colour interpolation
 plantmesh,_=subset_mesh(mesh,np.flatnonzero(labels==1));scene=cast_scene(plantmesh)
 for i,name in enumerate(['0000','0048','0096','0144']):
  with np.load(io.BytesIO(meter.read(output/'projection_cache'/f'{name}.npz'))) as d:ids=d['triangle_id'];uv=d['barycentric_uv']
  original=render(mesh,name,ids,uv);pids,pz,puv=cast(scene,name);only=render(plantmesh,name,pids,puv)
  png(output/f'{name}_original_mesh.png',original,meter);png(output/f'{name}_plant_only.png',only,meter)
  rgbim=np.array(rgb(name));pred=ids>=0;colors=np.array([[90,90,90],[30,220,70],[240,70,45],[30,90,250],[250,210,30]])
  overlay=rgbim.copy();overlay[pred]=(.55*rgbim[pred]+.45*colors[labels[ids[pred]]]).astype('uint8');png(output/f'{name}_semantics.png',overlay,meter)
  canvas.paste(Image.fromarray(original),(i*530,25));canvas.paste(Image.fromarray(only),(i*530,1005));draw.text((i*530+4,5),name+' original Mesh',fill='black');draw.text((i*530+4,985),name+' Plant-only',fill='black')
 canvas.save(output/'plant_only_overview.png')
 print('SEMANTIC_COMPLETE',outputs,flush=True)

if __name__=='__main__':main()
