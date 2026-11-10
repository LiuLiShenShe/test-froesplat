from common import *
import torch, time, csv
from PIL import Image,ImageDraw
from scipy import ndimage
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize
from evaluate_meshes import metrics
def png(p,img):
    Image.fromarray(np.rint(np.clip(img,0,1)*255).astype(np.uint8)).save(p,compress_level=1)
def depthimage(depth):
    import matplotlib
    d=matplotlib.colormaps['turbo'](np.clip((depth-.15)/1.42,0,1))[...,:3];d[depth<=0]=0;return d
def main():
    p=argparse.ArgumentParser();p.add_argument('--ply',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'preview_complete.json').exists():return
    GaussianModel,native=setup();model=GaussianModel(3,0);model.load_ply(str(a.ply));torch.set_num_threads(8)
    pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    import lpips
    perceptual=lpips.LPIPS(net='alex',verbose=False).cuda().eval()
    cams,refs=cameras();rows=[];start=time.time();reps=[0,26,53,79,105,131,157,183]
    for c,r in zip(cams,refs):
        sid=r['source_id']
        if sid not in reps:continue
        with torch.no_grad():pkg=native.render(c,model,pipe,torch.zeros(3,device='cuda'),0.)
        rgb=pkg['render'].permute(1,2,0).cpu().numpy();alpha=pkg['mask'].squeeze().cpu().numpy();depth=pkg['median_depth'].squeeze().cpu().numpy();normal=pkg['normal'].permute(1,2,0).cpu().numpy()
        with np.load(SEL/'P2_GGGS_Cleaning/surface_inputs/rendered'/f'{c.image_name}.npz') as z:fg=z['foreground_mask'].copy();gt=z['gt_rgb'].copy();refdepth=z['depth_surface'].copy();refnormal=z['native_normal'].copy()
        pred=alpha>.5;err=(np.clip(rgb,0,1)-gt)**2;score=metrics(pred,fg,depth,refdepth,.8903337497);sk=skeletonize(fg)
        with torch.no_grad():lp=float(perceptual(torch.from_numpy(rgb.clip(0,1).copy()).permute(2,0,1)[None].cuda()*2-1,torch.from_numpy(gt.copy()).permute(2,0,1)[None].cuda()*2-1))
        valid=fg&pred&(refdepth>0);dv=np.abs(depth[valid]-refdepth[valid]);nn=np.linalg.norm(normal,axis=2)*np.linalg.norm(refnormal,axis=2);nv=valid&(nn>1e-6);angle=np.degrees(np.arccos(np.clip((normal*refnormal).sum(2)[nv]/nn[nv],-1,1)))
        rows.append(dict(source_id=sid,full_psnr=float(-10*np.log10(err.mean())),fg_psnr=float(-10*np.log10(err[fg].mean())),ssim=float(structural_similarity(gt,rgb.clip(0,1),channel_axis=2,data_range=1.)),lpips=lp,fg_alpha=float(alpha[fg].mean()),bg_alpha=float(alpha[~fg].mean()),bg_alpha_gt05=float((alpha[~fg]>.5).mean()),thin_coverage=float(ndimage.binary_dilation(pred,iterations=3)[sk].mean()),depth_delta_median=float(np.median(dv)),depth_delta_p95=float(np.quantile(dv,.95)),depth_shift_gt2voxel=float((dv>.0034778662).mean()),normal_delta_median=float(np.median(angle)),**{k:v for k,v in score.items() if isinstance(v,(int,float))}))
        d=a.out/'training'/f'{sid:04d}';d.mkdir(parents=True,exist_ok=True);save_npz_fast(d/'outputs.npz',rgb=rgb,alpha=alpha,depth=depth,normal=normal)
        for name,im in [('RGB',rgb),('Alpha',alpha),('Depth',depthimage(depth)),('Normal',normal*.5+.5)]:png(d/f'{name}.png',im)
        print('preview train',a.out.name,sid,rows[-1]['fg_psnr'],flush=True)
    free_rows=[];protocol=json.loads((SEL/'comparison/shared_free_views.json').read_text());contact=Image.new('RGB',(4*360,2*680),'white');draw=ImageDraw.Draw(contact)
    for i,r in enumerate(protocol['views']):
        c=free_camera(r)
        with torch.no_grad():pkg=native.render(c,model,pipe,torch.full((3,),protocol['background'],device='cuda'),0.)
        rgb=pkg['render'].permute(1,2,0).cpu().numpy();alpha=pkg['mask'].squeeze().cpu().numpy();depth=pkg['median_depth'].squeeze().cpu().numpy();normal=pkg['normal'].permute(1,2,0).cpu().numpy()
        with np.load(ROOT/'task0_gaussian_cleaning/diagnosis/free_contribution_views'/f"{r['name']}.npz") as z:fg=z['reference_mesh_silhouette'].copy();roi=z['black_roi'].copy()
        outside=ndimage.distance_transform_edt(~fg)>5;black=outside&(rgb@[.2126,.7152,.0722]<.12)&(alpha>.5)
        free_rows.append(dict(name=r['name'],bg_alpha=float(alpha[outside].mean()),bg_highalpha_px=int((outside&(alpha>.5)).sum()),black_cloud_px=int(black.sum()),frozen_cloud_roi_px=int(roi.sum()),frozen_cloud_alpha_mean=float(alpha[roi].mean()) if roi.any() else 0.,frozen_cloud_black_remaining_px=int((black&roi).sum()),fg_mesh_proxy_alpha=float(alpha[fg].mean())))
        d=a.out/'free'/r['name'];d.mkdir(parents=True,exist_ok=True);save_npz_fast(d/'outputs.npz',rgb=rgb,alpha=alpha,depth=depth,normal=normal)
        for name,im in [('RGB',rgb),('Alpha',alpha),('Depth',depthimage(depth)),('Normal',normal*.5+.5)]:png(d/f'{name}.png',im)
        im=Image.open(d/'RGB.png');im.thumbnail((360,640));x=(i%4)*360;y=(i//4)*680;contact.paste(im,(x,y+30));draw.text((x+5,y+8),r['name'],fill='black')
        print('preview free',r['name'],free_rows[-1]['black_cloud_px'],flush=True)
    contact.save(a.out/'free_contact.png')
    for file,data in [('training_metrics.csv',rows),('free_metrics.csv',free_rows)]:
        with (a.out/file).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(a.out/'preview_complete.json',dict(ply=str(a.ply.resolve()),sha256=sha(a.ply),gaussian_count=len(model.get_xyz),training_views=8,free_views=8,seconds=time.time()-start,metrics_scope='Representative preview only; final uses208 training views',physical_gpu=1))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
