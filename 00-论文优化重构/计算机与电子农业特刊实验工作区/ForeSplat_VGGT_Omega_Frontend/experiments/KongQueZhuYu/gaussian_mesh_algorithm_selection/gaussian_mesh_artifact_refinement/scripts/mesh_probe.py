"""Trace P3 component/face candidates against all original views, no edits."""
from common import *
import open3d as o3d, csv, time
from scipy import ndimage
def main():
    dest=ROOT/'task1_p3_local_repair';path=SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply'
    mesh=o3d.io.read_triangle_mesh(str(path));v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);tri=v[f];centers=tri.mean(1)
    comp,sizes,areas=mesh.cluster_connected_triangles();comp=np.asarray(comp);sizes=np.asarray(sizes)
    scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    index=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text());scale=json.loads((OLD/'configs/standard_spatial_scale.json').read_text());voxel=scale['tsdf_voxel']
    c=index[0]['camera'];E=np.asarray(c['world_view_transform']).T;C=np.linalg.inv(E)
    ray=np.r_[C[:3,3],np.array([(650-c['Cx'])/c['Fx'],(1390-c['Cy'])/c['Fy'],1])@C[:3,:3].T].astype(np.float32)[None]
    hit=scene.cast_rays(o3d.core.Tensor(ray));tid=int(hit['primitive_ids'].numpy()[0]);marker_component=int(comp[tid]);assert sizes[marker_component]<100000
    lengths=np.stack([np.linalg.norm(tri[:,i]-tri[:,(i+1)%3],axis=1) for i in range(3)],1)
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    slender=lengths.max(1)**2/np.maximum(2*area,1e-12)
    candidate=(sizes[comp]<10000)|(lengths.max(1)>5*voxel)|(slender>20)
    ids=np.flatnonzero(candidate);points=centers[ids];counts={k:np.zeros(len(ids),np.uint16) for k in ['frustum','visible','fg','bg','gggs_depth_support','stereo_depth_support']}
    records=[];start=time.time()
    for i,item in enumerate(index):
        c=item['camera'];E=np.asarray(c['world_view_transform']).T;C=np.linalg.inv(E);pc=points@E[:3,:3].T+E[:3,3];z=pc[:,2]
        uv=pc[:,:2]/np.maximum(z[:,None],1e-9)*[c['Fx'],c['Fy']]+[c['Cx'],c['Cy']]
        x=np.rint(uv[:,0]).astype(int);y=np.rint(uv[:,1]).astype(int);inside=(z>0)&(x>=0)&(x<c['width'])&(y>=0)&(y<c['height']);sel=np.flatnonzero(inside);counts['frustum'][sel]+=1
        rays=np.column_stack([np.broadcast_to(C[:3,3],(len(sel),3)),(points[sel]-C[:3,3])/z[sel,None]]).astype(np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=8)['t_hit'].numpy();visible=np.abs(hit-z[sel])<voxel*.15;sel=sel[visible];counts['visible'][sel]+=1
        with np.load(item['path']) as a:fg=a['foreground_mask'].copy();depth=a['depth_surface'].copy()
        dist=ndimage.distance_transform_edt(~fg);foreground=fg[y[sel],x[sel]];background=dist[y[sel],x[sel]]>3
        counts['fg'][sel[foreground]]+=1;counts['bg'][sel[background]]+=1
        valid=(depth[y[sel],x[sel]]>0)&(np.abs(z[sel]-depth[y[sel],x[sel]])<np.maximum(2*voxel,.02*z[sel]))&foreground
        counts['gggs_depth_support'][sel[valid]]+=1
        stereo=SEL/'P3_GS2Mesh/stereo'/f'{i:03}'/'out_DLNR_Middlebury';sd=np.load(stereo/'depth.npy',mmap_mode='r');sc=np.load(stereo/'occlusion_mask.npy',mmap_mode='r')
        valid=(np.abs(z[sel]-sd[y[sel],x[sel]])<np.maximum(3*voxel,.02*z[sel]))&foreground&sc[y[sel],x[sel]]
        counts['stereo_depth_support'][sel[valid]]+=1
        rec=dict(source_id=item['source_id'])
        for cc in [28,29,marker_component]:
            chosen=comp[ids[sel]]==cc;rec.update({f'comp{cc}_visible':int(chosen.sum()),f'comp{cc}_fg':int(foreground[chosen].sum()),f'comp{cc}_stereo':int(valid[chosen].sum())})
        records.append(rec);print('mesh evidence',item['source_id'],len(sel),flush=True)
    save_npz_fast(dest/'face_multiview_evidence.npz',original_triangle_ids=ids,component=comp[ids],max_edge=lengths[ids].max(1),slender=slender[ids],**counts)
    summary=[]
    for cc in np.unique(comp[ids]):
        chosen=comp[ids]==cc;point=v[np.unique(f[comp==cc])];rec=dict(component_id=int(cc),faces=int(sizes[cc]),bbox_min=point.min(0).tolist(),bbox_max=point.max(0).tolist())
        for k,a in counts.items():rec[k+'_sum']=int(a[chosen].sum());rec[k+'_max']=int(a[chosen].max())
        summary.append(rec)
    write(dest/'component_multiview_evidence.json',dict(source=str(path),source_sha256=sha(path),marker_component=marker_component,marker_seed_source0_pixel=[650,1390],candidates=len(ids),summary=summary,seconds=time.time()-start,records=records,method='Actual mesh triangle centroid, original P3 ray visibility, RAP, GGGS median depth and original GS2Mesh stereo. Centroid support is a screening proxy; local RGB and entire triangle must be reviewed before cutting.'))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
