"""Reattribute newly revealed dark layers in the current model before editing."""
from common import *
from artifact_evidence import contribution
from plyfile import PlyData,PlyElement
from scipy import ndimage
import torch,csv,time
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    GaussianModel,native=setup();model=GaussianModel(3,0);model.load_ply(str(a.input/'gaussians.ply'));torch.set_num_threads(8)
    ids=np.load(a.input/'original_gaussian_ids.npy');ev={k:v for k,v in np.load(ROOT/'task0_gaussian_cleaning/diagnosis/actual_contribution_evidence.npz').items()};n=len(ids);fg=np.zeros(n);bg=np.zeros(n);black=np.zeros(n);cams,refs=cameras();pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    for c,r in zip(cams,refs):
        if r['source_id'] not in [0,26,53,79,105,131,157,183]:continue
        m=mask(c);outside=ndimage.distance_transform_edt(~m)>3;g,radii,err=contribution(c,model,np.stack([m,outside,outside]));fg+=g[:,0]/8;bg+=g[:,1]/8
        print('re-evidence training',r['source_id'],flush=True)
    protocol=json.loads((SEL/'comparison/shared_free_views.json').read_text());regions=[]
    for r in protocol['views']:
        c=free_camera(r)
        with np.load(ROOT/'task0_gaussian_cleaning/diagnosis/free_contribution_views'/f"{r['name']}.npz") as z:m=z['reference_mesh_silhouette'].copy()
        outside=ndimage.distance_transform_edt(~m)>5
        with torch.no_grad():rgb=native.render(c,model,pipe,torch.full((3,),protocol['background'],device='cuda'),0.)['render'].permute(1,2,0).cpu().numpy()
        cloud=outside&(rgb@[.2126,.7152,.0722]<.12);g,radii,err=contribution(c,model,np.stack([m,outside,cloud]));bg+=g[:,1]/8;black+=g[:,2]/8
        top=np.argsort(-g[:,2])[:100];regions.append(dict(name=r['name'],dominant_original_ids=ids[top].tolist(),black_weight=float(g[:,2].sum())))
        save_npz_fast(a.out/(r['name']+'_evidence.npz'),contribution=g,original_ids=ids,cloud=cloud,radii=radii)
        print('re-evidence free',r['name'],flush=True)
    fg=np.maximum(fg,ev['fg_contribution'][ids]/208);ratio=bg/np.maximum(fg+bg,1e-12);remove=(ratio>.95)&(bg>.05)
    data=PlyData.read(str(a.input/'gaussians.ply'))['vertex'].data.copy()
    for file,sel in [('gaussians.ply',~remove),('removed_gaussians.ply',remove)]:PlyData([PlyElement.describe(data[sel].copy(),'vertex')],text=False,byte_order='<').write(str(a.out/file))
    np.save(a.out/'original_gaussian_ids.npy',ids[~remove]);np.save(a.out/'removed_this_round_original_gaussian_ids.npy',ids[remove]);save_npz_fast(a.out/'newly_visible_evidence.npz',original_ids=ids,fg=fg,bg=bg,black=black,harm_ratio=ratio,removed=remove)
    write(a.out/'edit_manifest.json',dict(input=str(a.input.resolve()),input_sha256=sha(a.input/'gaussians.ply'),before=n,after=int((~remove).sum()),removed_this_round=int(remove.sum()),prior_fg_contribution_fraction=float(fg[remove].sum()/fg.sum()),removed_current_black_contribution_fraction=float(black[remove].sum()/max(black.sum(),1)),rules='Newly visible native contributions: max(original208 foreground mean,current8 foreground mean), current8 training background plus8 free background; harm ratio>.95 and bg>.05. No scale/color/opacity edits.',regions=regions))
    print('iterative deleted',int(remove.sum()),'remaining',int((~remove).sum()),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
