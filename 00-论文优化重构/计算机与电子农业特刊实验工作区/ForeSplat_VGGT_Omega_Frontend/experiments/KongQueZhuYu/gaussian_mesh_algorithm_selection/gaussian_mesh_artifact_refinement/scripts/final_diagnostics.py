from common import *
from artifact_evidence import contribution
from plyfile import PlyData, PlyElement
from scipy import ndimage
import torch, csv

def main():
    d=ROOT/'task0_gaussian_cleaning/best_cleaned'
    GaussianModel,native=setup(); model=GaussianModel(3,0)
    model.load_ply(str(d/'cleaned_gaussians.ply')); torch.set_num_threads(8)
    ids=np.load(d/'original_gaussian_ids.npy'); cams,refs=cameras()
    v=PlyData.read(str(d/'cleaned_gaussians.ply'))['vertex'].data
    xyz=np.column_stack([v[k] for k in ['x','y','z']]); sums=np.zeros(len(v),np.float64); records=[]
    pipe=argparse.Namespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    for sid in [37,38,36,41,42,40]:
        c=cams[next(i for i,r in enumerate(refs) if r['source_id']==sid)]
        with np.load(d/'historical_208/rendered'/f'{c.image_name}.npz') as z:
            rgb=z['render_raw']; gt=z['gt_rgb']; fg=z['foreground_mask']; depth=z['depth_surface']
        with np.load(SEL/'P2_GGGS_Cleaning/surface_inputs/rendered'/f'{c.image_name}.npz') as z:olddepth=z['depth_surface']
        # Fixed, observed changed pot region; excludes near-black render backgrounds.
        roi=fg&(depth>0)&(olddepth>0)&(np.abs(depth-olddepth)>.02)&(rgb.mean(2)>gt.mean(2)+.15)
        g,rad,err=contribution(c,model,np.stack([roi,fg,roi])); sums+=g[:,0]
        order=np.argsort(-g[:,0]); cumulative=np.cumsum(g[order,0]); sel=order[:np.searchsorted(cumulative,cumulative[-1]*.95)+1]
        np.save(d/f'pot_bright_source{sid}_original_ids.npy',ids[sel])
        records.append(dict(source_id=sid,bright_changed_pixels=int(roi.sum()),contribution=float(g[:,0].sum()),top30_original_ids=ids[order[:30]].tolist(),bbox_min=xyz[sel].min(0).tolist(),bbox_max=xyz[sel].max(0).tolist(),relative_sum_error=err.tolist()))
        print('pot bright attribution',sid,len(sel),flush=True)
    order=np.argsort(-sums); keep=order[sums[order]>.1]
    with (d/'pot_bright_artifact_dependencies.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['original_id','bright_pot_alpha_weight_6views','x','y','z','opacity_logit','scale_0_log','scale_1_log','scale_2_log'])
        for j in keep:w.writerow([int(ids[j]),float(sums[j]),*xyz[j].tolist(),float(v['opacity'][j]),*[float(v[f'scale_{k}'][j]) for k in range(3)]])
    PlyData([PlyElement.describe(v[keep].copy(),'vertex')],text=False,byte_order='<').write(str(d/'pot_bright_contributors.ply'))
    write(d/'POT_BRIGHT_REGRESSION_DIAGNOSIS.json',dict(records=records,top100_original_ids=ids[order[:100]].tolist(),status='Unresolved: removal of dark occluding layers reveals inaccurate bright pot texture; no photometric training performed',source_sha256=sha(d/'cleaned_gaussians.ply')))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
