"""Surface graph distances, centre paths and single-side observed area."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components,dijkstra
from scipy.spatial import cKDTree

def surface_graph(vertices,faces):
 edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);edges.sort(1);edges=np.unique(edges,axis=0)
 lengths=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
 return csr_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices)))

def surface_path(vertices,graph,anchors):
 anchors=list(dict.fromkeys(map(int,anchors)));path=[]
 for source,target in zip(anchors[:-1],anchors[1:]):
  distances,previous=dijkstra(graph,indices=source,return_predecessors=True)
  if not np.isfinite(distances[target]):return None,None
  ids=[target]
  while ids[-1]!=source:
   index=int(previous[ids[-1]])
   if index<0:return None,None
   ids.append(index)
  ids=ids[::-1];path.extend(ids if not path else ids[1:])
 if len(path)<2:return None,None
 return np.asarray(path,int),float(np.linalg.norm(np.diff(vertices[path],axis=0),axis=1).sum())

def one_side_area(vertices,faces,reference_camera):
 center=vertices.mean(0);_,_,axes=np.linalg.svd(vertices-center,full_matrices=False);normal=axes[2];normal*=np.sign(normal@(reference_camera-center)) or 1.
 cross=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
 area=np.linalg.norm(cross,axis=1)/2;chosen=cross@normal>0
 # Report the better observed side. Selecting the camera-facing side alone
 # can switch to a tiny reverse-side fragment when the preferred RGB view
 # changes, even though the actual source mesh is identical.
 if area[~chosen].sum()>area[chosen].sum():normal=-normal;chosen=~chosen
 return float(area[chosen].sum()),float(area.sum()),chosen,normal

def centerline_anchors(vertices,faces):
 """Centre of successive principal-axis slabs, snapped to actual vertices."""
 center=vertices.mean(0);_,_,axes=np.linalg.svd(vertices-center,full_matrices=False);coords=(vertices-center)@axes.T
 lo,hi=np.quantile(coords[:,0],[.005,.995]);step=(hi-lo)/24
 anchors=[];width_pairs=[]
 for value in np.linspace(lo,hi,25):
  selected=np.flatnonzero(np.abs(coords[:,0]-value)<=max(step*.65,1e-8))
  if len(selected)<2:continue
  # Station centroid along both width and thickness, then an actual surface
  # vertex. No straight3D shortcut across empty space is introduced.
  target=np.median(coords[selected],axis=0);target[0]=value
  distance=np.linalg.norm((coords[selected]-target)*[1,1,1],axis=1)
  anchors.append(int(selected[np.argmin(distance)]))
  left=int(selected[np.argmin(coords[selected,1])]);right=int(selected[np.argmax(coords[selected,1])]);width_pairs.append((left,right))
 return np.array(anchors),width_pairs,axes
