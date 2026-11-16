"""Offline interactive previews from actual Gaussian rows and actual mesh triangles."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
import argparse,io,json,sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plyfile import PlyData
from runtime import OUT,CLEANER,Meter,write


def main():
 p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--mesh');a=p.parse_args();meter=Meter()
 if a.mesh:
  folder=OUT/'mesh'/a.mesh/a.model
  sys.path.insert(0,str(CLEANER));from nksr_pipeline.common import read_mesh
  for stage in ['raw','post']:
   mesh=read_mesh(folder/f'{stage}.ply',meter);native_faces=len(mesh.triangles)
   if native_faces>120000:mesh=mesh.simplify_quadric_decimation(120000)
   v,f,c=np.asarray(mesh.vertices),np.asarray(mesh.triangles),np.asarray(mesh.vertex_colors)
   fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],vertexcolor=np.rint(c*255).astype('uint8'),flatshading=False,name='Actual mesh triangles'))
   title=f'{a.model} / {a.mesh} / {stage}: actual mesh; display {len(f):,}/{native_faces:,} triangles'
   save(fig,folder/f'{stage}_3d.html',title,meter)
  return
 folder=OUT/'gaussian'/a.model;record=json.loads((folder/'complete.json').read_text())
 data=PlyData.read(io.BytesIO(meter.read(record['native_ply'])),mmap=False)['vertex'].data
 indices=np.random.default_rng(0).choice(len(data),min(len(data),50000),replace=False);rows=data[indices]
 xyz=np.column_stack([rows[k] for k in ['x','y','z']]);rgb=np.clip(.28209479177387814*np.column_stack([rows[f'f_dc_{i}'] for i in range(3)])+.5,0,1)
 colours=['rgb(%d,%d,%d)'%tuple(v) for v in np.rint(rgb*255).astype('uint8')]
 fig=go.Figure(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',marker=dict(size=1.3,color=colours,opacity=.75),name='Native Gaussian centres; DC colours'))
 trajectory=json.loads((OUT/'common_input/camera_trajectory.json').read_text());c=np.array([r['camera_center'] for r in trajectory])
 fig.add_trace(go.Scatter3d(x=c[:,0],y=c[:,1],z=c[:,2],mode='lines',line=dict(color='orange',width=3),name='Actual COLMAP camera trajectory'))
 save(fig,folder/'centres_3d.html',f'{a.model}: native Gaussian centre preview {len(rows):,}/{len(data):,}; DC colour, not splat rendering',meter)
 write(folder/'viewer_provenance.json',dict(source=record['native_ply'],sample_seed=0,display_points=len(rows),native_points=len(data),representation='Actual Gaussian centres with SH0-derived colour; anisotropic footprint and view-dependent SH are in the native PLY and RGB renders, not represented as points here',io=meter.stats()))


def save(fig,path,title,meter):
 fig.update_layout(title=title,scene=dict(aspectmode='data',xaxis_title='COLMAP X',yaxis_title='COLMAP Y',zaxis_title='COLMAP Z'),margin=dict(l=5,r=5,t=75,b=5),height=850)
 buffer=io.StringIO();fig.write_html(buffer,include_plotlyjs=True,full_html=True)
 meter.save(path,buffer.getvalue().encode())
 print(path,flush=True)


if __name__=='__main__':main()
