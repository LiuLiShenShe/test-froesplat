from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1]
canvas=Image.new('RGB',(900,1032),'white')
for row,sid in enumerate([0,53,105]):
 with np.load(root/'surface_inputs/e10_gggs_surface/rendered'/f'crop_{sid:04d}.npz') as z: gt=z['gt_rgb']
 arrays=[(f'{sid}: GT',gt)]
 for m in ['M4_E10_GGGS_TSDF','M5_E11_2DGS_TSDF']:
  for stage in ['raw','post']:
   with np.load(root/'meshes'/m/'evaluation/views'/f'{stage}_{sid:04d}.npz') as z: rgb=z['mesh_rgb']
   arrays.append((m[:2]+' '+stage,rgb))
 for col,(name,rgb) in enumerate(arrays):
  im=Image.fromarray((np.clip(rgb,0,1)*255).round().astype('uint8')).resize((180,320),Image.Resampling.NEAREST)
  canvas.paste(im,(col*180,row*344+24));ImageDraw.Draw(canvas).text((col*180+5,row*344+5),name,fill='black')
canvas.save(root/'comparisons/visualizations/tsdf_preview_native.png')
