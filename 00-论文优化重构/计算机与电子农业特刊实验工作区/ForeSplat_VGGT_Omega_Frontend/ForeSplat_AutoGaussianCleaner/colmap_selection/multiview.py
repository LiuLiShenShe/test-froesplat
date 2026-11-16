"""Camera-Z reprojection consistency; a self-consistency measure, not 3D truth."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from runtime import OUT,Meter,write


def main():
 p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--cleaned',action='store_true');a=p.parse_args()
 folder=OUT/('cleaning' if a.cleaned else 'gaussian')/a.method/'evaluation';meter=Meter()
 names=[Path(n).stem for n in (OUT/'common_input/test_views.txt').read_text().splitlines()]
 data={}
 for name in names:
  with np.load(io.BytesIO(meter.read(folder/f'{name}_geometry.npz'))) as d:record={k:d[k] for k in d.files}
  record['mask']=np.array(Image.open(OUT/'common_input/dataset/rap_masks'/f'{name}.png').convert('L'))>127
  data[name]=record
 rows=[]
 for name in names:
  d=data[name];z=d['depth_camera_z'];K=d['K'];c2w=d['c2w'];valid=d['valid']&d['mask']
  y,x=np.nonzero(valid)
  if len(x)>50000:
   idx=np.linspace(0,len(x)-1,50000,dtype=int);x,y=x[idx],y[idx]
  xyz=np.column_stack([(x+.5-K[0,2])*z[y,x]/K[0,0],(y+.5-K[1,2])*z[y,x]/K[1,1],z[y,x]])@c2w[:3,:3].T+c2w[:3,3]
  # Fixed camera proximity determines neighbours independently of the model.
  candidates=[]
  for other,e in data.items():
   if other==name:continue
   cosine=float(np.dot(c2w[:3,2],e['c2w'][:3,2]))
   if cosine>.8:candidates.append((float(np.linalg.norm(c2w[:3,3]-e['c2w'][:3,3])),other))
  for _,other in sorted(candidates)[:3]:
   e=data[other];w2c=np.linalg.inv(e['c2w']);points=xyz@w2c[:3,:3].T+w2c[:3,3]
   uv=points[:,:2]/np.maximum(points[:,2,None],1e-8)*[K[0,0],K[1,1]]+[K[0,2]-.5,K[1,2]-.5]
   ij=np.rint(uv).astype(int);h,w=z.shape
   inside=(points[:,2]>0)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
   ids=np.flatnonzero(inside);ix,iy=ij[ids,0],ij[ids,1]
   supported=e['valid'][iy,ix]&e['mask'][iy,ix];ids=ids[supported];ix,iy=ij[ids,0],ij[ids,1]
   expected=points[ids,2];observed=e['depth_camera_z'][iy,ix]
   # Source points behind the other visible surface can be genuinely occluded.
   visible=expected<=observed*1.02
   relative=np.abs(expected[visible]-observed[visible])/np.maximum(observed[visible],1e-8)
   rows.append(dict(source=name,target=other,sampled_source_points=len(xyz),projected_common_points=len(expected),occlusion_eligible_points=int(visible.sum()),occlusion_eligible_fraction=float(visible.mean()) if len(visible) else 0.,relative_depth_median=float(np.median(relative)) if len(relative) else None,relative_depth_p95=float(np.quantile(relative,.95)) if len(relative) else None,within_2pct_fraction=float((relative<=.02).mean()) if len(relative) else None))
 values=[r for r in rows if r['relative_depth_median'] is not None]
 write(folder/'multiview_consistency.json',dict(method=a.method,cleaned=a.cleaned,pairs=rows,mean={k:float(np.mean([r[k] for r in values])) for k in ['occlusion_eligible_fraction','relative_depth_median','relative_depth_p95','within_2pct_fraction']},protocol='Up to50k deterministic RAP+alpha valid pixels; nearest3 camera centres whose optical-axis cosine>.8; same camera-Z convention and half-pixel intrinsics; reject source points >2% behind target surface. Counts and eligible fractions are retained to expose visibility censoring.',limitation='Self-consistency only. Incorrect smooth surfaces can agree across views; combines with RGB, thin-mask coverage and visual inspection. Different native depth estimators remain.',io=meter.stats()))
 print(a.method,'multiview pairs',len(rows),flush=True)


if __name__=='__main__':main()
