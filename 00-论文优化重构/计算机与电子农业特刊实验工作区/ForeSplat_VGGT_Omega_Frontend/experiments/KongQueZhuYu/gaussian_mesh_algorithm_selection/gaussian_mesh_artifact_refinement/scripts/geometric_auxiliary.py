from common import *
from plyfile import PlyData
from scipy.spatial import cKDTree
import open3d as o3d,csv,time
def main():
    d=ROOT/'task0_gaussian_cleaning/diagnosis';data=PlyData.read(str(SOURCE))['vertex'].data;xyz=np.column_stack([data[k] for k in ['x','y','z']]).astype(np.float64);start=time.time();dist,_=cKDTree(xyz).query(xyz,k=9,workers=8);density=8/(4*np.pi/3*np.maximum(dist[:,-1],1e-12)**3)
    source=DUAL/'Standard_VGGT_Original_RGB/foreground.ply';vggt=PlyData.read(str(source))['vertex'].data;points=np.column_stack([vggt[k] for k in ['x','y','z']]).astype(np.float64);pointdist=cKDTree(points).query(xyz,workers=8)[0];del points,vggt
    index=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text());samples=[]
    for item in index:
        c=item['camera'];C=np.linalg.inv(np.asarray(c['world_view_transform']).T)
        with np.load(item['path']) as z:dep=z['depth_surface'][::12,::12].copy();fg=z['foreground_mask'][::12,::12].copy();alpha=z['alpha'][::12,::12].copy()
        yy,xx=np.mgrid[0:c['height']:12,0:c['width']:12];valid=fg&(alpha>.5)&(dep>0)&np.isfinite(dep);cam=np.column_stack([(xx[valid]-c['Cx'])/c['Fx']*dep[valid],(yy[valid]-c['Cy'])/c['Fy']*dep[valid],dep[valid]]);samples.append(cam@C[:3,:3].T+C[:3,3])
    surface=np.concatenate(samples);renderdist=cKDTree(surface).query(xyz,workers=8)[0]
    mesh=o3d.io.read_triangle_mesh(str(SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply'));scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));meshdist=scene.compute_distance(o3d.core.Tensor(xyz.astype(np.float32)),nthreads=8).numpy()
    save_npz_fast(d/'geometric_auxiliary_evidence.npz',original_ids=np.arange(len(xyz)),local_neighbor9_distance=dist[:,-1],local_density=density,vggt_pointmap_distance=pointdist,gggs_visible_surface_sample_distance=renderdist,p3_triangle_distance=meshdist)
    with (d/'geometric_auxiliary_evidence.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['original_gaussian_id','neighbor9_distance','density_k8','VGGT_pointmap_distance','GGGS_visible_surface_distance_stride12','P3_triangle_distance'])
        w.writerows(zip(range(len(xyz)),dist[:,-1],density,pointdist,renderdist,meshdist))
    write(d/'geometric_auxiliary_complete.json',dict(gaussians=len(xyz),vggt_pointmap=str(source),rendered_foreground_surface_samples=len(surface),views=208,seconds=time.time()-start,interpretation='VGGT and stereo mesh are auxiliary reconstructions, not scan truth. Visible-surface distance is to deterministic stride12 pixel samples, an upper bound on full sampled surface distance; not botanical thickness. k8 density is local center density, not Gaussian volume occupancy.'))
    print('geometric auxiliary complete',len(xyz),len(surface),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
