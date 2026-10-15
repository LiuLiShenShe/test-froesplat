"""Full-triangle CPU surface visualization, with a common M0 camera frame."""
from common import *
from argparse import ArgumentParser
import math

def main():
    p=ArgumentParser();p.add_argument('--sample',choices=SCENES,required=True);p.add_argument('--method',choices=['M0','M1','M2'],required=True);a=p.parse_args()
    import numpy as np,torch,open3d as o3d,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluate_mesh import camera_from_frozen
    baseline=o3d.io.read_triangle_mesh(str(DATA/'meshes'/a.sample/'M0/main/post.ply'))
    center=(baseline.get_min_bound()+baseline.get_max_bound())/2
    distance=float(np.linalg.norm(baseline.get_max_bound()-baseline.get_min_bound())*1.1)
    row=sorted(rows(STAGE/f'metrics/{a.sample}_M0_FULL.csv'),key=lambda r:r['loader_image_name'])[0]
    native=camera_from_frozen(a.sample,row['loader_image_name'],int(row['render_width']),int(row['render_height']))
    R=native.world_view_transform.T.numpy()[:3,:3].astype(float)
    angle=math.radians(25);yaw=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
    R=yaw@R;C=center-R.T@np.array([0.,0.,distance]);E=np.eye(4);E[:3,:3]=R;E[:3,3]=-R@C
    size=720;focal=size/(2*math.tan(math.pi/6));K=np.array([[focal,0,(size-1)/2],[0,focal,(size-1)/2],[0,0,1.]])
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,stage in zip(axes,['raw','post']):
        mesh=o3d.io.read_triangle_mesh(str(DATA/'meshes'/a.sample/a.method/'main'/f'{stage}.ply'))
        tri=np.asarray(mesh.triangles);vertices=np.asarray(mesh.vertices);colors=np.asarray(mesh.vertex_colors)
        scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        rays=scene.create_rays_pinhole(o3d.core.Tensor(K),o3d.core.Tensor(E),size,size);result=scene.cast_rays(rays,nthreads=4)
        ids=result['primitive_ids'].numpy();uv=result['primitive_uvs'].numpy();normals=result['primitive_normals'].numpy()
        valid=ids<len(tri);image=np.full((size,size,3),.08)
        if valid.any():
            faces=tri[ids[valid]];weights=np.c_[1-uv[valid].sum(1),uv[valid]]
            color=(colors[faces]*weights[:,:,None]).sum(1) if len(colors)==len(vertices) else np.full((valid.sum(),3),.65)
            light=(C-center)/np.linalg.norm(C-center);intensity=.3+.7*np.abs(normals[valid]@light)
            image[valid]=np.clip(color*intensity[:,None],0,1)
        ax.imshow(image);ax.set_title(f'{a.method} {stage}: full {len(tri):,} triangles');ax.axis('off')
    fig.suptitle(f'{a.sample}: actual Mesh surfaces, common M0 framing and lighting\nCPU raycast of all triangles; arbitrary COLMAP units; no geometry GT',fontsize=12)
    fig.tight_layout();path=STAGE/f'figures/{a.sample}_{a.method}_SURFACE.png';fig.savefig(path,dpi=150);plt.close(fig)
    write_json(STAGE/f'reports/{a.sample}_{a.method}_SURFACE_VIEW.json',{'status':'PASS','figure_sha256':sha(path),'world_to_camera':E.tolist(),'K':K.tolist(),
               'frame_source':'M0 post Mesh bounding-box center and diagonal; frozen first heldout orientation yaw25deg',
               'view_type':'Supplementary descriptive 3D surface view; same frame for all methods; full triangles raycast',
               'lighting':'Common .3+.7*absolute(normal dot view direction); display only, no metric computation',
               'selection_limit':'Visualization rule introduced after inspecting M0 plots for readability, before formal M1/M2 outcomes; not a preregistered primary case'})

if __name__=='__main__':main()
