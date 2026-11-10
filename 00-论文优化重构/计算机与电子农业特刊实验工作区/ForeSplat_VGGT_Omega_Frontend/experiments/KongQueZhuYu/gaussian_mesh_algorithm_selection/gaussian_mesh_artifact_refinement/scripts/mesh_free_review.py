from common import *
from evaluate_meshes import cast
import open3d as o3d
from PIL import Image,ImageDraw

def main():
    protocol=json.loads((SEL/'comparison/shared_free_views.json').read_text());root=ROOT/'comparisons/mesh_comparison';root.mkdir(parents=True,exist_ok=True)
    models={'G0':SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply','P3_local_v2':ROOT/'task1_p3_local_repair/mesh_versions/P3_local_cleanup_v2.ply','G1':ROOT/'task2_cleaned_gs2mesh/G1_P2_to_GS2Mesh/meshes_shared_depth_range/post.ply','G2':ROOT/'task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh/meshes_shared_depth_range/post.ply'}
    models['G2_targeted']=ROOT/'task2_cleaned_gs2mesh/G2_BestCleaned_to_GS2Mesh/local_repair/post.ply'
    records=[]
    for name,path in models.items():
        if (root/(name+'_free_contact.png')).exists():
            records.append(dict(model=name,mesh=str(path.resolve()),sha256=sha(path),views=8));continue
        m=o3d.io.read_triangle_mesh(str(path));v=np.asarray(m.vertices);f=np.asarray(m.triangles);col=np.asarray(m.vertex_colors);scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        contact=Image.new('RGB',(4*270,2*510),'white');draw=ImageDraw.Draw(contact)
        for i,c in enumerate(protocol['views']):
            out=root/name/c['name'];out.mkdir(parents=True,exist_ok=True)
            depth,norm,rgb=cast(scene,c,v,f,col);hit=np.isfinite(depth)&(depth>0);nc=norm@np.asarray(c['world_view_transform']).T[:3,:3].T;gray=np.repeat((.2+.8*np.abs(nc[:,:,2]))[...,None],3,2);gray[~hit]=protocol['background'];rgb[~hit]=protocol['background'];normal=nc*.5+.5;normal[~hit]=protocol['background']
            save_npz_fast(out/'raycast.npz',depth=np.where(hit,depth,0),normal=nc)
            for label,im in [('RGB',rgb),('Gray',gray),('Normal',normal)]:Image.fromarray((np.clip(im,0,1)*255).round().astype(np.uint8)).save(out/(label+'.png'),compress_level=1)
            im=Image.open(out/'Gray.png');im.thumbnail((270,480));x=i%4*270;y=i//4*510;contact.paste(im,(x,y+25));draw.text((x+5,y+5),c['name'],fill='black')
        contact.save(root/(name+'_free_contact.png'));records.append(dict(model=name,mesh=str(path.resolve()),sha256=sha(path),views=8));print('mesh free',name,flush=True)
    write(root/'FREE_MESH_REVIEW_PROTOCOL.json',dict(records=records,cameras=str(SEL/'comparison/shared_free_views.json'),background=protocol['background'],status='Rendered; human/model visual review recorded separately'))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
