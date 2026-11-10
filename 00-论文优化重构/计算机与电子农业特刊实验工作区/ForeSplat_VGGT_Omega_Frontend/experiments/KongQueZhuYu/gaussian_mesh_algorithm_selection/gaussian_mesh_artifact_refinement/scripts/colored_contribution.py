from common import *
from artifact_evidence import contribution
from scipy import ndimage
import torch,csv
def main():
    GaussianModel,native=setup();from utils.sh_utils import eval_sh
    protocol=json.loads((SEL/'comparison/shared_free_views.json').read_text());rows=[];totals={};torch.set_num_threads(8)
    for label,path in [('E10',SOURCE),('Best',ROOT/'task0_gaussian_cleaning/best_cleaned/cleaned_gaussians.ply')]:
        model=GaussianModel(3,0);model.load_ply(str(path));n=len(model.get_xyz);dark=np.zeros(n);color=np.zeros(n)
        for r in protocol['views']:
            c=free_camera(r);cache=ROOT/'task0_gaussian_cleaning/diagnosis/free_contribution_views'/f"{r['name']}.npz"
            with np.load(cache) as z:
                if label=='E10':g=z['contribution'].copy()
                m=z['reference_mesh_silhouette'].copy()
            if label=='Best':
                with np.load(ROOT/'task0_gaussian_cleaning/best_cleaned/preview/free'/r['name']/'outputs.npz') as z:rgb=z['rgb'].copy()
                outside=ndimage.distance_transform_edt(~m)>5;cloud=outside&(rgb@[.2126,.7152,.0722]<.12);g,rad,err=contribution(c,model,np.stack([m,outside,cloud]))
            with torch.no_grad():
                dirs=torch.nn.functional.normalize(model.get_xyz-c.camera_center,dim=1);rgb=(eval_sh(3,model.get_features.transpose(1,2),dirs)+.5).clamp_min(0).cpu().numpy();lum=rgb@[.2126,.7152,.0722]
            dark+=g[:,2]*(lum<.12);color+=g[:,2]*lum
            rows.append(dict(model=label,camera=r['name'],black_roi_alpha_weight=float(g[:,2].sum()),dark_SH_gaussian_alpha_weight=float((g[:,2]*(lum<.12)).sum()),colored_luminance_contribution=float((g[:,2]*lum).sum())))
        ids=np.arange(n) if label=='E10' else np.load(ROOT/'task0_gaussian_cleaning/best_cleaned/original_gaussian_ids.npy');save_npz_fast(ROOT/'task0_gaussian_cleaning/diagnosis'/f'{label}_actual_dark_SH_contribution.npz',original_ids=ids,dark_alpha_contribution=dark,luminance_contribution=color);totals[label]=dict(dark_alpha_weight=float(dark.sum()),luminance_contribution=float(color.sum()),dark_contributing_gaussians=int((dark>.05).sum()))
    with (ROOT/'task0_gaussian_cleaning/diagnosis/actual_black_color_contribution.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'task0_gaussian_cleaning/diagnosis/actual_black_color_contribution.json',dict(totals=totals,method='Native degree3 SH evaluated in the same mean-camera direction as computeColorFromSHSG, +0.5, clamp_min0; multiply constant Gaussian color by exact native T*alpha contribution. Dark-color threshold luminance<0.12.'))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
