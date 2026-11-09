import json
import numpy as np
from scipy import ndimage
from surface_protocol import ROOT,write
records=[]
for sid in [0,26,53,79,105,131,157,183]:
 with np.load(ROOT/'surface_inputs/e10_gggs_surface/rendered'/f'crop_{sid:04d}.npz') as z:fg=z['foreground_mask']
 labels,n=ndimage.label(fg);counts=np.bincount(labels.ravel());counts[0]=0
 silhouette=ndimage.binary_fill_holes(labels==counts.argmax())
 yy,xx=np.nonzero(silhouette);lo=np.array([xx.min(),yy.min()]);hi=np.array([xx.max(),yy.max()])
 border=silhouette^ndimage.binary_erosion(silhouette)
 by,bx=np.nonzero(border);h,w=fg.shape
 valid=(bx>=96)&(bx<w-96)&(by>=96)&(by<h-96)&(by<lo[1]+.65*(hi[1]-lo[1]))
 bx=bx[valid];by=by[valid];assert len(bx)
 target=[min(float(lo[0]+.85*(hi[0]-lo[0])),w-97),float(lo[1]+.30*(hi[1]-lo[1]))]
 score=np.abs(by-target[1])+.35*np.abs(bx-target[0]);i=int(np.argmin(score));cx=int(bx[i]);cy=int(by[i])
 records.append(dict(source_id=sid,bounds_native=[cx-96,cy-96,cx+96,cy+96],boundary_pixel=[cx,cy],target=target))
write(ROOT/'comparisons/visualizations/leaf_boundary_roi_protocol.json',dict(source='Existing matched E10 GT/RAP only; no competing render/mesh used',rule='Largest connected RAP silhouette, fill internal holes for boundary selection only; upper65percent of FG bbox, target85percent-x/30percent-y, keep96px from image border. No segmentation or mesh changed',native_side=192,records=records))
