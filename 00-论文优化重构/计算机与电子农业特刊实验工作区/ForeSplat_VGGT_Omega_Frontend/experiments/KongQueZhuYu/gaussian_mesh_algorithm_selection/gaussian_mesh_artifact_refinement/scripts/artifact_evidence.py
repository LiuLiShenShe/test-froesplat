"""Exact native alpha-compositing contribution by color Jacobian; no optimization."""
from common import *
import torch, time
from scipy import ndimage
from plyfile import PlyData
from PIL import Image

def contribution(c, model, weights):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    settings=GaussianRasterizationSettings(image_height=c.image_height,image_width=c.image_width,tanfovx=math.tan(c.FoVx/2),tanfovy=math.tan(c.FoVy/2),kernel_size=0.,bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=c.world_view_transform,projmatrix=c.full_proj_transform,sh_degree=3,sg_degree=0,campos=c.camera_center,prefiltered=False,require_depth=True,debug=False)
    colors=torch.ones((len(model.get_xyz),3),device='cuda',requires_grad=True)
    scales,opacity=model.get_scaling_n_opacity_with_3D_filter
    out=GaussianRasterizer(raster_settings=settings)(means3D=model.get_xyz.detach(),means2D=torch.zeros_like(model.get_xyz),shs=None,sg_axis=model.get_sg_axis.detach(),sg_sharpness=model.get_sg_sharpness.detach(),sg_color=model.get_sg_color.detach(),colors_precomp=colors,opacities=opacity.detach(),scales=scales.detach(),rotations=model.get_rotation.detach(),cov3Ds_precomp=None)
    w=torch.from_numpy(weights.astype(np.float32)).cuda()
    grad=torch.autograd.grad(out[0],colors,grad_outputs=w)[0].detach().cpu().numpy()
    alpha=out[3].detach().cpu().numpy().squeeze()
    expected=(weights*alpha[None]).sum((1,2));actual=grad.sum(0)
    err=np.abs(actual-expected)/np.maximum(expected,1)
    assert np.max(err)<.002,(actual,expected,err)
    return grad,out[1].cpu().numpy(),err

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args()
    dest=ROOT/'task0_gaussian_cleaning/diagnosis';dest.mkdir(exist_ok=True)
    GaussianModel,native=setup();model=GaussianModel(3,0);model.load_ply(str(SOURCE));torch.set_num_threads(8)
    cams,refs=cameras();n=len(model.get_xyz)
    values={k:np.zeros(n,np.float64) for k in ['fg_contribution','bg_contribution','black_bg_contribution','max_radius_px']}
    for k in ['visible_views','fg_visible_views','bg_visible_views']:values[k]=np.zeros(n,np.uint16)
    start=time.time();records=[]
    for i,(c,r) in enumerate(zip(cams,refs)):
        if a.limit is not None and i>=a.limit:break
        sid=r['source_id'];cache=dest/'contribution_views'/f'{sid:04d}.npz';cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            z=np.load(cache);g=z['contribution'];radii=z['radii'];err=z['relative_sum_error']
        else:
            fg=mask(c);outside=ndimage.distance_transform_edt(~fg)>3
            with np.load(OLD/'surface_inputs/e10_gggs_surface/rendered'/f'{c.image_name}.npz') as z:rgb=z['render_raw']
            black=outside&(rgb@[.2126,.7152,.0722]<.12)
            g,radii,err=contribution(c,model,np.stack([fg,outside,black]))
            save_npz_fast(cache,contribution=g,radii=radii,relative_sum_error=err)
        for j,k in enumerate(['fg_contribution','bg_contribution','black_bg_contribution']):values[k]+=g[:,j]
        values['visible_views']+=(g[:,:2].sum(1)>.01)
        values['fg_visible_views']+=(g[:,0]>.1)
        values['bg_visible_views']+=(g[:,1]>.1)
        values['max_radius_px']=np.maximum(values['max_radius_px'],radii)
        records.append(dict(source_id=sid,relative_sum_error=err.tolist(),fg_weight=float(g[:,0].sum()),bg_weight=float(g[:,1].sum()),black_weight=float(g[:,2].sum())))
        print('contribution',sid,'visible',int((g.sum(1)>.01).sum()),'maxerr',float(err.max()),flush=True)
    if a.limit is None:
        import open3d as o3d
        from evaluate_meshes import cast
        mesh=o3d.io.read_triangle_mesh(str(SEL/'P3_GS2Mesh/meshes_shared_depth_range/post.ply'))
        scene=o3d.t.geometry.RaycastingScene(nthreads=8);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);col=np.asarray(mesh.vertex_colors)
        protocol=json.loads((SEL/'comparison/shared_free_views.json').read_text())
        for k in ['free_fg_contribution','free_bg_contribution','free_black_contribution']:values[k]=np.zeros(n,np.float64)
        free_records=[]
        for r in protocol['views']:
            c=free_camera(r);cache=dest/'free_contribution_views'/f"{r['name']}.npz";cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                z=np.load(cache);g=z['contribution'];radii=z['radii'];err=z['relative_sum_error']
            else:
                depth,normal,rgbmesh=cast(scene,r,v,f,col);fg=np.isfinite(depth)&(depth>0)
                outside=ndimage.distance_transform_edt(~fg)>5
                with np.load(SEL/'comparison/visualizations/E10_free'/r['name']/'native_outputs.npz') as z:rgb=z['rgb_gray_background']
                black=outside&(rgb@[.2126,.7152,.0722]<.12)
                g,radii,err=contribution(c,model,np.stack([fg,outside,black]))
                save_npz_fast(cache,contribution=g,radii=radii,relative_sum_error=err,reference_mesh_silhouette=fg,black_roi=black)
            for j,k in enumerate(['free_fg_contribution','free_bg_contribution','free_black_contribution']):values[k]+=g[:,j]
            free_records.append(dict(name=r['name'],relative_sum_error=err.tolist(),black_weight=float(g[:,2].sum())))
            print('free contribution',r['name'],float(g[:,2].sum()),flush=True)
        write(dest/'free_evidence_complete.json',dict(records=free_records,reference='Original P3_shared mesh silhouette, conservative geometry proxy, not ground truth',black_roi='E10 gray-background luminance<.12 and >5px outside original P3 silhouette'))
    save_npz_fast(dest/'actual_contribution_evidence.npz',**values)
    write(dest/('evidence_complete.json' if a.limit is None else 'evidence_smoke.json'),dict(views=len(records),count=n,source=str(SOURCE),source_sha256=sha(SOURCE),seconds=time.time()-start,records=records,method='Native GGGS rasterizer dRGB_channel/dGaussian_color_channel = sum(T * alpha) inside fixed pixel weights; parameters detached, no training. bg excludes3px RAP band. Black ROI luminance<0.12. Sum independently validated against rendered alpha.',physical_gpu=1))
if __name__=='__main__':
    main();sys.stdout.flush();os._exit(0)
