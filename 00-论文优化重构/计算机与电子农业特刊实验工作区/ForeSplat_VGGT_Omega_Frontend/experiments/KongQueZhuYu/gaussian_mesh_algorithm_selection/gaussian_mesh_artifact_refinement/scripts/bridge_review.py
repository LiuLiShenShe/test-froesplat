from common import *
import open3d as o3d
from PIL import Image,ImageDraw
from evaluate_meshes import cast
def main():
    d=ROOT/'task1_p3_local_repair/wrong_connections';ids=np.load(d/'candidate_original_triangle_ids.npy');mesh=o3d.io.read_triangle_mesh(str(SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply'));v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);col=np.asarray(mesh.vertex_colors);small=o3d.geometry.TriangleMesh();small.vertices=mesh.vertices;small.triangles=o3d.utility.Vector3iVector(f[ids]);comp,sizes,_=small.cluster_connected_triangles();comp=np.asarray(comp);sizes=np.asarray(sizes);scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));index=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text());records=[];panel=Image.new('RGB',(4*256,8*280),'white');draw=ImageDraw.Draw(panel)
    for j,cc in enumerate(np.argsort(-sizes)[:8]):
        triids=ids[comp==cc];points=v[np.unique(f[triids])];center=points.mean(0);best=None
        for item in index:
            if item['source_id'] not in [0,26,53,79,105,131,157,183]:continue
            c=item['camera'];E=np.asarray(c['world_view_transform']).T;pc=center@E[:3,:3].T+E[:3,3]
            if pc[2]<=0:continue
            uv=pc[:2]/pc[2]*[c['Fx'],c['Fy']]+[c['Cx'],c['Cy']];x,y=np.rint(uv).astype(int)
            if not (0<=x<c['width'] and 0<=y<c['height']):continue
            # Existing cached full ray depth is exact original P3 visibility.
            with np.load(SEL/'P3_GS2Mesh/meshes_shared_depth_range/evaluation/views'/f"post_{item['source_id']:04d}.npz") as z:md=z['mesh_depth'][y,x]
            if md<=0 or abs(md-pc[2])>.004:continue
            score=c['Fx']*c['Fy']/pc[2]**2
            if best is None or score>best[0]:best=(score,item,x,y)
        if best is None:continue
        _,item,x,y=best;c=item['camera'];sid=item['source_id'];x1=max(0,min(c['width']-192,x-96));y1=max(0,min(c['height']-192,y-96));x2=x1+192;y2=y1+192
        depth,norm,rgb=cast(scene,c,v,f,col);pred=np.isfinite(depth)&(depth>0);E=np.asarray(c['world_view_transform']).T;gray=np.repeat((.2+.8*np.abs((norm@E[:3,:3].T)[:,:,2]))[...,None],3,2);gray[~pred]=0
        raw=np.asarray(Image.open(Path('/data/fj/F2DMAS/02-FFT/KongQueZhuYu')/f'{sid:04d}.jpg').convert('RGB').resize((c['width'],c['height'])),np.float32)/255
        with np.load(item['path']) as z:gg=z['render_raw'].copy();dd=z['depth_surface'].copy()
        for k,(label,im) in enumerate([('original RGB',raw),('GGGS',gg),('gray mesh',gray),('mesh color',rgb)]):
            crop=Image.fromarray((im[y1:y2,x1:x2].clip(0,1)*255).round().astype(np.uint8)).resize((256,256));panel.paste(crop,(k*256,j*280+24));draw.text((k*256+3,j*280+3),f'{label} region{int(cc)} source{sid}',fill='black')
        records.append(dict(region_id=int(cc),source_id=sid,original_triangle_ids=triids.tolist(),roi=[x1,y1,x2,y2],bbox_min=points.min(0).tolist(),bbox_max=points.max(0).tolist(),mesh_center_depth=float(depth[y,x]),gggs_center_depth=float(dd[y,x]),status='Unconfirmed candidate; mask overlap and local geometric disagreement do not establish a bridge between anatomically separate leaves'))
        print('bridge region',cc,len(triids),sid,flush=True)
    panel.save(d/'candidate_review.png');write(d/'candidate_review.json',dict(records=records,all_candidate_original_ids=ids.tolist(),edited=False))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
