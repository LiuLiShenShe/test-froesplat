from common import *
import torch,open3d as o3d
from scipy.spatial.transform import Rotation
from scipy import ndimage
from PIL import Image,ImageDraw
from evaluate_meshes import cast
def main():
    d=ROOT/'comparisons/free_view_images/extended';d.mkdir(parents=True,exist_ok=True);refs=json.loads((OLD/'surface_inputs/e10_gggs_surface/camera_runtime.json').read_text())['records'];r0=refs[0];pivot=np.array([-.008067,-.094974,.886115]);original=np.linalg.inv(np.asarray(r0['world_view_transform']).T);views=[]
    for name,angle,distance in [('front_far',[0,0,0],1.8),('top_orbit',[-30,0,0],1.4),('side_orbit',[0,60,0],1.5),('oblique_orbit',[-15,-45,0],1.4)]:
        q=Rotation.from_euler('xyz',angle,degrees=True).as_matrix();C=original.copy();C[:3,3]=pivot+q@(original[:3,3]-pivot)*distance;C[:3,:3]=q@original[:3,:3];E=np.linalg.inv(C);r=dict(r0,name=name,world_view_transform=E.T.tolist(),camera_center=C[:3,3].tolist(),full_proj_transform=(E.T@np.asarray(r0['projection_matrix'])).tolist());views.append(r)
    write(d/'camera_protocol.json',dict(views=views,pivot=pivot.tolist(),background=.25,scope='Additional novel orbit/far cameras. Same E10 K, native coordinates; no foreground mask ground truth.'))
    mesh=o3d.io.read_triangle_mesh(str(SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply'));v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);col=np.asarray(mesh.vertex_colors);scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));GaussianModel,native=setup();pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);torch.set_num_threads(8);records=[];panel=Image.new('RGB',(4*360,3*675),'white');draw=ImageDraw.Draw(panel)
    paths=[('E10',SOURCE),('P2',P2),('Round5',ROOT/'task0_gaussian_cleaning/additional_rounds/round_05/gaussians.ply')]
    for j,(label,path) in enumerate(paths):
        model=GaussianModel(3,0);model.load_ply(str(path))
        for i,r in enumerate(views):
            c=free_camera(r)
            with torch.no_grad():pkg=native.render(c,model,pipe,torch.full((3,),.25,device='cuda'),0.)
            rgb=pkg['render'].permute(1,2,0).cpu().numpy();alpha=pkg['mask'].squeeze().cpu().numpy();depth,normal,mesh_rgb=cast(scene,r,v,f,col);m=np.isfinite(depth)&(depth>0);outside=ndimage.distance_transform_edt(~m)>5;black=outside&(alpha>.5)&(rgb@[.2126,.7152,.0722]<.12)
            out=d/label/r['name'];out.mkdir(parents=True,exist_ok=True);save_npz_fast(out/'outputs.npz',rgb=rgb,alpha=alpha,reference_mesh_silhouette=m,black=black)
            im=Image.fromarray((rgb.clip(0,1)*255).round().astype(np.uint8));im.save(out/'RGB.png');im.thumbnail((360,640));panel.paste(im,(i*360,j*675+30));draw.text((i*360+5,j*675+5),label+' '+r['name'],fill='black');records.append(dict(model=label,camera=r['name'],black_pixels=int(black.sum()),background_alpha=float(alpha[outside].mean())))
            print('extended view',label,r['name'],int(black.sum()),flush=True)
        del model;torch.cuda.empty_cache()
    panel.save(d/'comparison.png');write(d/'review_metrics.json',dict(records=records))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
