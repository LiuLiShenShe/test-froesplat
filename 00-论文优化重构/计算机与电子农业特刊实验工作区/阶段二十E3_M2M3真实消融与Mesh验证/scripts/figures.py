"""Direct visualizations of measured arrays and actual Gaussian/Mesh assets."""
from common import *
from argparse import ArgumentParser
import math

def normalize_depth(depth,lo,hi):
    import numpy as np,matplotlib
    valid=np.isfinite(depth)&(depth>0)
    colors=matplotlib.colormaps['viridis'](np.clip((np.nan_to_num(depth)-lo)/(hi-lo),0,1))[...,:3]
    colors[~valid]=0;return colors

def main():
    p=ArgumentParser();p.add_argument('--sample',choices=SCENES,required=True);p.add_argument('--method',choices=['M0','M1','M2'],required=True);a=p.parse_args()
    import numpy as np,torch,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    import open3d as o3d
    from plyfile import PlyData
    core=metric_module();label=f'{a.sample}_{a.method}'
    values=rows(STAGE/f'metrics/{label}_FULL.csv');ordered=sorted(values,key=lambda r:r['loader_image_name']);selected=[ordered[i] for i in [0,len(ordered)//2,len(ordered)-1]]
    depth_max=read_json(STAGE/f'configs/{a.sample}_MESH.json')['depth_trunc']
    # One immutable M0 mesh frame is shared by every method of the scene.
    baseline=o3d.io.read_triangle_mesh(str(DATA/'meshes'/a.sample/'M0/main/raw.ply'))
    bmin=np.asarray(baseline.get_min_bound());bmax=np.asarray(baseline.get_max_bound());center=(bmin+bmax)/2;radius=max(bmax-bmin)*.53
    fig=plt.figure(figsize=(16,5.2))
    points=PlyData.read(model_path(a.sample,a.method)/'point_cloud/iteration_30000/point_cloud.ply')['vertex']
    xyz=np.stack([points[k] for k in ['x','y','z']],1);color=np.clip(np.stack([points[f'f_dc_{i}'] for i in range(3)],1)*.28209479177387814+.5,0,1)
    ax=fig.add_subplot(131,projection='3d');stride=max(1,math.ceil(len(xyz)/35000));ax.scatter(*xyz[::stride].T,c=color[::stride],s=.2,depthshade=False)
    ax.set_title(f'{a.method}: Gaussian centers ({len(xyz):,})\nDisplay every {stride}th center')
    for index,stage in enumerate(['raw','post'],2):
        mesh=o3d.io.read_triangle_mesh(str(DATA/'meshes'/a.sample/a.method/'main'/f'{stage}.ply'));mesh.compute_vertex_normals()
        v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors)
        if len(colors)!=len(v):colors=np.full_like(v,.6)
        stride=max(1,math.ceil(len(t)/45000));faces=t[::stride]
        ax=fig.add_subplot(130+index,projection='3d');collection=Poly3DCollection(v[faces],facecolors=colors[faces].mean(1),linewidths=0,rasterized=True);ax.add_collection3d(collection)
        if stage=='raw':
            boundary=np.load(DATA/'meshes'/a.sample/a.method/'main/raw_boundary_edges.npy');step=max(1,math.ceil(len(boundary)/10000))
            ax.add_collection3d(Line3DCollection(v[boundary[::step]],colors='#e93131',linewidths=.13))
        ax.set_title(f'{a.method}: {stage} Mesh ({len(t):,} faces)\nDisplay every {stride}th face'+('; red: boundary sample' if stage=='raw' else ''))
    for ax in fig.axes:
        ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
        ax.set_box_aspect([1,1,1]);ax.view_init(elev=20,azim=45);ax.set_xlabel('COLMAP x');ax.set_ylabel('COLMAP y');ax.set_zlabel('COLMAP z')
    fig.suptitle(f'{a.sample} | actual geometry, arbitrary COLMAP units; fixed M0 framing; no geometry GT')
    fig.tight_layout();fig.savefig(STAGE/f'figures/{label}_GEOMETRY.png',dpi=150);plt.close(fig)
    for r in selected:
        name=r['loader_image_name'];d=torch.load(r['float_render_path'],map_location='cpu',weights_only=True)
        ref=core.resize_reference(r['reference_path'],int(r['render_width']),int(r['render_height'])).permute(1,2,0).numpy()
        roi=core.load_roi(r['ROI_path'],int(r['render_width']),int(r['render_height'])).numpy()
        raw=np.load(DATA/'mesh_reprojection'/a.sample/a.method/f'raw_{name}.npz');post=np.load(DATA/'mesh_reprojection'/a.sample/a.method/f'post_{name}.npz')
        normal=d['normal'].permute(1,2,0).numpy();normal_display=(normal+1)/2
        normal_display[d['geometry_replay_alpha'].numpy()<=.5]=0
        displays=[('Canonical raw RGB',ref),('Gaussian RGB',d['rgb'].permute(1,2,0).numpy()),('Gaussian alpha',d['alpha'].numpy()),
                  ('Fixed R0 proxy ROI (not GT)',roi),('Gaussian depth',normalize_depth(d['depth'].numpy(),0,depth_max)),('Raw Mesh depth',normalize_depth(raw['depth'],0,depth_max)),
                  ('Post Mesh depth',normalize_depth(post['depth'],0,depth_max)),('Gaussian normal',normal_display),
                  ('Raw Mesh normal (world)',np.where(np.isfinite(raw['depth'])[...,None],(raw['normal']+1)/2,0)),
                  ('Post Mesh normal (world)',np.where(np.isfinite(post['depth'])[...,None],(post['normal']+1)/2,0))]
        fig,axes=plt.subplots(2,5,figsize=(14,9))
        for ax,(title,image) in zip(axes.ravel(),displays):ax.imshow(image,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=9);ax.axis('off')
        fig.suptitle(f'{a.sample} / {a.method} / frozen held-out {name}; depth [0,{depth_max:.3g}] arbitrary units\nNormals are in different native conventions; panels are descriptive, not angular-error comparisons.')
        fig.tight_layout();fig.savefig(STAGE/f'figures/{label}_{name}_REPROJECTION.png',dpi=130);plt.close(fig)
    write_json(STAGE/f'reports/{label}_FIGURES.json',{'status':'PASS','selected_cameras':[r['loader_image_name'] for r in selected],
               'selection':'Frozen sorted first/middle/last rule, no quality ranking','geometry_display':'Real PLY points/triangles with deterministic display subsampling only',
               'framing':'Common M0 raw mesh bounding cube; full mesh preserved externally','axes':'Arbitrary COLMAP units; no physical scale',
               'boundary_caution':'Red edges mark mesh boundaries, including true open surfaces and holes; no independent truth classification'})

if __name__=='__main__':main()
