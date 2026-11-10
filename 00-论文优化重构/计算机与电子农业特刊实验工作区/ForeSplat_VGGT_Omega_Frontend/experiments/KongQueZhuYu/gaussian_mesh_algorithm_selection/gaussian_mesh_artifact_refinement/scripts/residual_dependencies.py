from common import *
from artifact_evidence import contribution
from scipy import ndimage
from plyfile import PlyData
import torch,csv
def main():
    d=ROOT/'task0_gaussian_cleaning/best_cleaned';GaussianModel,native=setup();model=GaussianModel(3,0);model.load_ply(str(d/'cleaned_gaussians.ply'));torch.set_num_threads(8);ids=np.load(d/'original_gaussian_ids.npy');n=len(ids);fg=np.zeros(n);bg=np.zeros(n);black=np.zeros(n);cams,refs=cameras();pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);perview=[]
    for c,r in zip(cams,refs):
        if r['source_id'] not in [0,26,53,79,105,131,157,183]:continue
        m=mask(c);outside=ndimage.distance_transform_edt(~m)>3;g,rad,err=contribution(c,model,np.stack([m,outside,outside]));fg+=g[:,0]/8;bg+=g[:,1]/8
    protocol=json.loads((SEL/'comparison/shared_free_views.json').read_text())
    for r in protocol['views']:
        c=free_camera(r)
        with np.load(ROOT/'task0_gaussian_cleaning/diagnosis/free_contribution_views'/f"{r['name']}.npz") as z:m=z['reference_mesh_silhouette'].copy()
        outside=ndimage.distance_transform_edt(~m)>5
        with torch.no_grad():rgb=native.render(c,model,pipe,torch.full((3,),protocol['background'],device='cuda'),0.)['render'].permute(1,2,0).cpu().numpy()
        cloud=outside&(rgb@[.2126,.7152,.0722]<.12);g,rad,err=contribution(c,model,np.stack([m,outside,cloud]));bg+=g[:,1]/8;black+=g[:,2]/8;top=np.argsort(-g[:,2])[:50];perview.append(dict(camera=r['name'],top_remaining_original_ids=ids[top].tolist(),remaining_black_weight=float(g[:,2].sum())))
    original=np.load(ROOT/'task0_gaussian_cleaning/diagnosis/actual_contribution_evidence.npz')['fg_contribution'];fg=np.maximum(fg,original[ids]/208);ratio=bg/np.maximum(fg+bg,1e-12);sel=np.flatnonzero(black>.05);sel=sel[np.argsort(-black[sel])];data=PlyData.read(str(d/'cleaned_gaussians.ply'))['vertex'].data
    with (d/'remaining_artifact_dependencies.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['original_gaussian_id','x','y','z','fg_alpha_pixel_weight_per_view','bg_alpha_pixel_weight_per_view','black_alpha_pixel_weight_per_freeview','joint_harm_ratio','scale0','scale1','scale2','opacity_logit'])
        for i in sel:w.writerow([ids[i],*[data[k][i] for k in ['x','y','z']],fg[i],bg[i],black[i],ratio[i],*[data[f'scale_{k}'][i] for k in range(3)],data['opacity'][i]])
    save_npz_fast(d/'remaining_actual_contribution.npz',original_ids=ids,fg=fg,bg=bg,black=black,harm_ratio=ratio)
    write(d/'remaining_dependencies.json',dict(gaussians_with_remaining_black_weight_gt005=len(sel),fraction_black_weight_on_ratio_le095=float(black[ratio<=.95].sum()/black.sum()),fraction_black_weight_on_ratio_le08=float(black[ratio<=.8].sum()/black.sum()),top100_original_ids=ids[sel[:100]].tolist(),per_view=perview,interpretation='Actual final-model native alpha weights. High foreground benefit does not certify true geometry, but deletion alone removes observed foreground support. Remaining dark pixel regions may include imperfect P3 silhouette proxy or real dark/occluded leaf texture; region-specific RGB/geometry validation is required.',final_ply_sha256=sha(d/'cleaned_gaussians.ply')))
    print('final residual dependency evidence',len(sel),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
