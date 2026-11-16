"""Reuse the validated Open3D ray caster and mesh audits with this run's cameras."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,csv,io,json,sys,time,resource
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from runtime import OUT,FRONT,CLEANER,Meter,write


def main():
 p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('method');a=p.parse_args()
 output=OUT/'mesh'/a.method/a.model;evaluation=output/'evaluation';evaluation.mkdir(exist_ok=True)
 if (evaluation/'complete.json').exists():return
 meter=Meter();start=time.perf_counter();cpu=time.process_time()
 sys.path.insert(0,str(FRONT/'experiments/KongQueZhuYu/geometry_mesh_final_selection/scripts'))
 from evaluate_meshes import cast,structure,metrics
 surface=OUT/'mesh'/f'{a.model}_surface';scale=json.loads((surface/'spatial_scale.json').read_text());diag=scale['diagonal']
 tests=[Path(n).stem for n in (OUT/'common_input/test_views.txt').read_text().splitlines()]
 reps=['0000','0048','0096','0144'];rows=[];structures=[]
 sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import read_mesh
 for stage in ['raw','post']:
  mesh=read_mesh(output/f'{stage}.ply',meter);assert len(mesh.vertices)>0 and len(mesh.triangles)>0
  structures.append(structure(mesh,a.method,stage,diag))
  v,f,c=np.asarray(mesh.vertices),np.asarray(mesh.triangles),np.asarray(mesh.vertex_colors)
  scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
  images=[]
  for name in tests:
   with np.load(io.BytesIO(meter.read(surface/'frames'/f'{name}.npz'))) as d:z,K,c2w,rgb=d['depth'],d['K'],d['c2w'],d['rgb']
   h,w=z.shape;camera=dict(height=h,width=w,Fx=float(K[0,0]),Fy=float(K[1,1]),Cx=float(K[0,2]-.5),Cy=float(K[1,2]-.5),world_view_transform=np.linalg.inv(c2w).T.tolist())
   depth,normal,colour=cast(scene,camera,v,f,c);pred=np.isfinite(depth)&(depth>0)
   mask=np.array(Image.open(OUT/'common_input/dataset/rap_masks'/f'{name}.png').convert('L'))>127
   row=dict(model=a.model,method=a.method,stage=stage,view=name,reference='RAP prediction, includes pot in some views',**metrics(pred,mask,depth,z,diag));rows.append(row)
   if name in reps:
    painted=np.zeros((*mask.shape,3),dtype=np.uint8);painted[pred&mask]=[210,210,210];painted[mask&~pred]=[255,40,40];painted[pred&~mask]=[0,100,255]
    for channel,array in [('RGB',(np.clip(colour,0,1)*255).astype('uint8')),('Normal',(np.clip((normal+1)/2,0,1)*255).astype('uint8')),('Silhouette',painted)]:
     im=Image.fromarray(array);b=io.BytesIO();im.save(b,format='PNG');meter.save(evaluation/f'{stage}_{name}_{channel}.png',b.getbuffer())
    im=Image.fromarray((np.clip(colour,0,1)*255).astype('uint8'));im.thumbnail((265,474));images.append((name,im))
   print('MESH_EVAL',a.model,a.method,stage,name,row['silhouette_iou'],flush=True)
  canvas=Image.new('RGB',(265*4,500),'#eeeeee');draw=ImageDraw.Draw(canvas)
  for i,(name,im) in enumerate(images):canvas.paste(im,(i*265,26));draw.text((i*265+4,5),f'{stage} {name}',fill='black')
  b=io.BytesIO();canvas.save(b,format='PNG');meter.save(evaluation/f'{stage}_four_views.png',b.getbuffer())
 with (evaluation/'per_view.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 summary=[]
 for s in structures:
  values=[r for r in rows if r['stage']==s['stage']]
  summary.append(dict(model=a.model,**s,**{k:float(np.mean([r[k] for r in values])) for k in values[0] if isinstance(values[0][k],(int,float))}))
 write(evaluation/'metrics.json',summary)
 write(evaluation/'complete.json',dict(status='complete',model=a.model,method=a.method,views_per_stage=len(tests),wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,io=meter.stats(),rays='Integer pixel centres, COLMAP cx/cy minus0.5; unnormalized world ray gives t_hit=camera Z',depth_reference='Own Gaussian native depth consistency, not ground-truth geometric accuracy',reference='Same archived RAP predictions for every mesh, includes flowerpot in some views; no 3D scan truth'))


if __name__=='__main__':
 main()
 sys.stdout.flush();sys.stderr.flush();os._exit(0)
