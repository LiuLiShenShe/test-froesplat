import json
import numpy as np
from surface_protocol import ROOT,DUAL
from build_visualizations import uint8,tile,montage,local
out=ROOT/'comparisons/visualizations'
methods=['M1_Omega_NKSR','M2_Standard_NKSR','M3_E10_GGGS_NKSR','M4_E10_GGGS_TSDF','M5_E11_2DGS_TSDF']
rows=[];edges=[]
protocol=json.loads((out/'leaf_boundary_roi_protocol.json').read_text())['records']
protocol={r['source_id']:r['bounds_native'] for r in protocol}
for sid in [53,105]:
 with np.load(ROOT/'surface_inputs/e10_gggs_surface/rendered'/f'crop_{sid:04d}.npz') as z: gt=z['gt_rgb']
 row=[tile(uint8(gt),f'{sid}: GT',(180,320))];edge=[tile(uint8(local(gt,protocol[sid])),f'{sid}: GT')]
 for m in methods:
  with np.load(ROOT/'meshes'/m/'evaluation/views'/f'raw_{sid:04d}.npz') as z: rgb=z['mesh_rgb'];mask=z['foreground_mask'];sil=z['mesh_silhouette']
  row.append(tile(uint8(rgb),m[:2],(180,320)))
  display=np.zeros((*sil.shape,3),dtype='uint8');display[mask&sil]=[210,210,210];display[mask&~sil]=[255,0,0];display[sil&~mask]=[0,90,255]
  edge.append(tile(local(display,protocol[sid]),m[:2]))
 rows.append(row);edges.append(edge)
montage(rows,out/'main_mesh_preview_raw.png');montage(edges,out/'main_mesh_true_boundary_preview_raw.png')
