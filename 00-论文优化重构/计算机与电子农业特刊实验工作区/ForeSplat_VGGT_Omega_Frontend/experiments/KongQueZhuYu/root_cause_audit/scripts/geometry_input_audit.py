"""Fixed-image-index, raw RGB/RAP/mask boundary and model-detail audit, CPU."""
import io,hashlib
from PIL import Image
from scipy.ndimage import binary_erosion,distance_transform_edt,laplace
from geometry_common import *
from geometry_multiview_audit import standard_input,ROIS

RAW=Path('/data/fj/F2DMAS/02-FFT/KongQueZhuYu')
RAP=Path('/data/fj/F2DMAS/03-SAM/KongQueZhuYu')
GT=Path('/data/fj/F2DMAS/03-GT/KongQueZhuYu')

def rgb_read(path):
    image=Image.open(path)
    if image.mode=='RGBA':image=Image.alpha_composite(Image.new('RGBA',image.size,(0,0,0,255)),image)
    return np.asarray(image.convert('RGB'))

def bilinear_query(rgb,coordinates):
    h,w=rgb.shape[:2];x=np.clip(coordinates[...,0],0,w-1);y=np.clip(coordinates[...,1],0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1);dx=(x-x0)[...,None];dy=(y-y0)[...,None]
    return rgb[y0,x0]*(1-dx)*(1-dy)+rgb[y0,x1]*dx*(1-dy)+rgb[y1,x0]*(1-dx)*dy+rgb[y1,x1]*dx*dy

def original_reference_grid(w,h):
    rw,rh=288,512;sy=rh/h;sx=rw/w;a=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    y,x=np.indices((rh,rw));original=np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(a).T
    return a,original

def gray(rgb):return np.asarray(rgb,dtype=float)@np.array([.299,.587,.114])

def detail(rgb,selected):
    g=gray(rgb);l=laplace(g);values=l[selected]
    return dict(laplacian_variance=float(np.var(values)) if values.size else None,
                mean_absolute_laplacian=float(np.mean(np.abs(values))) if values.size else None,
                selected_pixels=int(selected.sum()),gray_std=float(np.std(g[selected])) if values.size else None)

def mask_compare(mask,gt):
    edge=mask&~binary_erosion(mask);gt_edge=gt&~binary_erosion(gt)
    to_gt=distance_transform_edt(~gt_edge)[edge];to_mask=distance_transform_edt(~edge)[gt_edge]
    return dict(iou=float((mask&gt).sum()/max((mask|gt).sum(),1)),precision=float((mask&gt).sum()/max(mask.sum(),1)),recall=float((mask&gt).sum()/max(gt.sum(),1)),
        rap_boundary_to_manual_p50_original_px=q(to_gt,.5),rap_boundary_to_manual_p90_original_px=q(to_gt,.9),
        manual_boundary_to_rap_p50_original_px=q(to_mask,.5),manual_boundary_to_rap_p90_original_px=q(to_mask,.9),
        rap_boundary_within5px_fraction=float((to_gt<=5).mean()) if len(to_gt) else None,
        manual_boundary_within5px_fraction=float((to_mask<=5).mean()) if len(to_mask) else None)

def main():
    sources=camera_sources();names=sources['Omega_depth']['names'];by_id={source_id(n):n for n in names}
    representative=[source_id(names[int(i)]) for i in np.linspace(0,len(names)-1,12)]
    ids=sorted(set([0,25,50,75,100,68,73]+representative))
    manifest={r['image_name']:r for r in json.loads((STANDARD/'Standard_VGGT_Frontend/metadata/input_manifest.json').read_text())['frames']}
    fingerprints=[];blur=[];masks=[];figures=[]
    for sid in ids:
        name=by_id[sid];index=names.index(name);original=rgb_read(RAW/f'{sid:04d}.jpg');rap=rgb_read(RAP/name);mask=np.asarray(Image.open(RAP/f'mask_{sid:04d}.png').convert('L'))>=127
        assert original.shape==rap.shape and original.shape[:2]==mask.shape
        interior=binary_erosion(mask,iterations=2);edge=mask&~interior;difference=np.abs(original.astype(np.int16)-rap.astype(np.int16))
        fingerprints.append(dict(name=name,source_id=sid,original_source=str(RAW/f'{sid:04d}.jpg'),original_size_wh=[original.shape[1],original.shape[0]],
            original_sha256=hashlib.sha256((RAW/f'{sid:04d}.jpg').read_bytes()).hexdigest(),rap_sha256=hashlib.sha256((RAP/name).read_bytes()).hexdigest(),
            full_mask_rgb_mae=float(difference[mask].mean()),interior_rgb_mae=float(difference[interior].mean()),interior_rgb_max=int(difference[interior].max()),
            interior_exact_rgb_pixel_fraction=float(np.all(difference[interior]==0,axis=1).mean()),edge_rgb_mae=float(difference[edge].mean()),
            background_black_fraction=float(np.all(rap[~mask]==0,axis=1).mean())))
        a,grid=original_reference_grid(original.shape[1],original.shape[0]);ref_rgb=np.asarray(Image.fromarray(original).resize((288,512),Image.Resampling.BICUBIC))
        ref_mask=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((288,512),Image.Resampling.NEAREST))>0
        ref_interior=binary_erosion(ref_mask,iterations=2)
        omega_rgb=np.load(OMEGA_DATA/'per_frame'/f'model_rgb_{Path(name).stem}.npy')
        standard_rgb,standard_mask=standard_input(name,manifest[name])
        omega_uv=grid@sources['Omega_depth']['a_actual'][index].T;standard_uv=grid@sources['Standard_depth']['a_actual'][index].T
        omega_ref=bilinear_query(omega_rgb,omega_uv[...,:2]);standard_ref=bilinear_query(standard_rgb,standard_uv[...,:2])
        regions={'foreground_interior':ref_interior}
        for region,(left,top,right,bottom) in ROIS.items():regions[region]=ref_interior&(grid[...,0]>=left)&(grid[...,0]<right)&(grid[...,1]>=top)&(grid[...,1]<bottom)
        for region,selected in regions.items():
            for method,image in [('original_02_FFT',ref_rgb),('Omega_actual_saved_RGB',omega_ref),('Standard_preprocessing_replay',standard_ref)]:
                blur.append(dict(source_id=sid,name=name,region=region,method=method,**detail(image,selected)))
        if (GT/f'gt_{sid:04d}.png').exists():
            gt=np.asarray(Image.open(GT/f'gt_{sid:04d}.png'))>0;assert gt.shape==mask.shape
            row=dict(source_id=sid,name=name,**mask_compare(mask,gt));masks.append(row)
            figures.append(dict(source_id=sid,original=original,rap=rap,mask=mask,gt=gt,omega_ref=omega_ref,standard_ref=standard_ref))
        print('Input audit',sid,'interior RGB MAE',fingerprints[-1]['interior_rgb_mae'],flush=True)
    write_csv(AUDIT/'geometry_comparison/input_source_rgb_fingerprints.csv',fingerprints)
    write_csv(AUDIT/'geometry_comparison/input_blur_detail_fixed_rois.csv',blur)
    write_csv(AUDIT/'geometry_comparison/manual_GT_mask_boundary.csv',masks)
    write_json(AUDIT/'geometry_comparison/input_audit_protocol.json',dict(original_source=str(RAW),GT5=[0,25,50,75,100],representative12=representative,diagnostic_lowconfidence_frames=[68,73],actual_audited_ids=ids,
        blur='Laplacian statistics at common288x512original-FOV reference; interior2-pixel erosion only for measurement, masks/images unchanged. Standard RGB replay exactly follows saved source and producer1024bicubic then518bilinear; not a saved tensor.',
        mask='Existing binaryRAP mask versus existing manual2D annotation; includes plant,pot,blue object; original-coordinate boundary distance; no leaf-instance/3D GT claim',
        no_new_segmentation=True,no_model_inference=True,no_source_modification=True))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(figures),4,figsize=(14,4*len(figures)))
    for row,data in enumerate(figures):
        sid=data['source_id'];axes[row,0].imshow(data['original']);axes[row,0].set_title(f'{sid:04d} original02-FFT')
        axes[row,1].imshow(data['rap']);axes[row,1].set_title('Actual RAP foreground')
        small=np.asarray(Image.fromarray(data['original']).resize((432,768),Image.Resampling.BILINEAR))
        m=np.asarray(Image.fromarray(data['mask'].astype(np.uint8)*255).resize((432,768),Image.Resampling.NEAREST))>0
        g=np.asarray(Image.fromarray(data['gt'].astype(np.uint8)*255).resize((432,768),Image.Resampling.NEAREST))>0
        axes[row,2].imshow(small);axes[row,2].contour(m,levels=[.5],colors=['cyan'],linewidths=.4);axes[row,2].contour(g,levels=[.5],colors=['magenta'],linewidths=.4);axes[row,2].set_title('RAPcyan | manualGTmagenta')
        left,top,right,bottom=ROIS['leaf_window'];axes[row,3].imshow(data['original'][top:bottom,left:right]);axes[row,3].set_title('Fixed original leaf window')
        for ax in axes[row]:ax.axis('off')
    fig.suptitle('Fixed source IDs: raw RGB, RAP foreground, existing manual2D mask boundary\nDirect file inspection; images/masks unchanged; no new segmentation or3D GT claim',fontsize=11);fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(AUDIT/'geometry_comparison/original_RAP_manual_GT_five_views.png',dpi=140);plt.close(fig)
    write_json(AUDIT/'geometry_comparison/input_audit_summary.json',dict(images_checked=len(fingerprints),manual_gt_images=len(masks),
        interior_rgb_mae_median=q([r['interior_rgb_mae'] for r in fingerprints],.5),interior_exact_rgb_fraction_median=q([r['interior_exact_rgb_pixel_fraction'] for r in fingerprints],.5),
        manual_mask_iou_median=q([r['iou'] for r in masks],.5),manual_mask_recall_median=q([r['recall'] for r in masks],.5),manual_mask_precision_median=q([r['precision'] for r in masks],.5),
        mask_status='Only fixed existing2D references; no inference about unseen surfaces or complete208 segmentation accuracy',
        blur_status='Descriptive texture/detail statistic; real leaf texture,distance,illumination and resampling also affect Laplacian variance, so low variance alone does not prove motion blur'))

if __name__=='__main__':main()
