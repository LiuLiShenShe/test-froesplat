from common import *
from scipy import ndimage
import csv
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
def compare_one(job):
    label,i,pair=job
    baseline=SEL/'P3_GS2Mesh';dest=ROOT/'comparisons';route=ROOT/'task2_cleaned_gs2mesh'/label
    c=pair['left'];cached=dest/'stereo_rows'/label/f"{c['source_id']:04d}.json"
    if cached.exists():return json.loads(cached.read_text())
    a=baseline/'stereo'/f'{i:03}';b=route/'stereo'/f'{i:03}';fg=np.load(a/'left_mask.npy');assert np.array_equal(fg,np.load(b/'left_mask.npy'))
    da=np.load(a/'out_DLNR_Middlebury/depth.npy');db=np.load(b/'out_DLNR_Middlebury/depth.npy');oa=np.load(a/'out_DLNR_Middlebury/occlusion_mask.npy');ob=np.load(b/'out_DLNR_Middlebury/occlusion_mask.npy');ca=np.load(a/'out_DLNR_Middlebury/confidence.npy');cb=np.load(b/'out_DLNR_Middlebury/confidence.npy');valid_a=fg&oa&(da>=.1511057476)&(da<=1.5692457557);valid_b=fg&ob&(db>=.1511057476)&(db<=1.5692457557);common=valid_a&valid_b;delta=np.abs(db[common]-da[common]);edge=fg&~ndimage.binary_erosion(fg,iterations=5);inside=ndimage.binary_erosion(fg,iterations=5)
    left_a=np.asarray(Image.open(a/'left.png'),np.float32)/255;left_b=np.asarray(Image.open(b/'left.png'),np.float32)/255;right_a=np.asarray(Image.open(a/'right.png'),np.float32)/255;right_b=np.asarray(Image.open(b/'right.png'),np.float32)/255
    rec=dict(experiment=label,source_id=c['source_id'],left_fg_rgb_mean_abs=float(np.abs(left_a-left_b)[fg].mean()),left_bg_rgb_mean_abs=float(np.abs(left_a-left_b)[~fg].mean()),right_full_rgb_mean_abs=float(np.abs(right_a-right_b).mean()),G0_fg_stereo_support=float(valid_a[fg].mean()),new_fg_stereo_support=float(valid_b[fg].mean()),support_newly_gained_px=int((valid_b&~valid_a).sum()),support_newly_lost_px=int((valid_a&~valid_b).sum()),G0_fg_confidence=float(ca[fg].mean()),new_fg_confidence=float(cb[fg].mean()),depth_delta_median=float(np.median(delta)),depth_delta_p95=float(np.quantile(delta,.95)),depth_delta_mean=float(delta.mean()),depth_changed_gt2voxel_fraction=float((delta>.0034778662).mean()),G0_leaf_boundary_support=float(valid_a[edge].mean()),new_leaf_boundary_support=float(valid_b[edge].mean()),G0_interior_missing_fraction=float((~valid_a)[inside].mean()),new_interior_missing_fraction=float((~valid_b)[inside].mean()),common_fg_depth_pixels=int(common.sum()))
    write(cached,rec);print('stereo comparison',label,c['source_id'],rec['depth_delta_mean'],flush=True)

    return rec

def main():
    dest=ROOT/'comparisons';jobs=[]
    for label in ['G1_P2_to_GS2Mesh','G2_BestCleaned_to_GS2Mesh']:
        route=ROOT/'task2_cleaned_gs2mesh'/label;assert (route/'stereo_complete.json').exists()
        cameras=json.loads((route/'stereo/camera_data.json').read_text())['pairs']
        jobs.extend((label,i,pair) for i,pair in enumerate(cameras))
    with ThreadPoolExecutor(max_workers=16) as pool:rows=list(pool.map(compare_one,jobs))
    with (dest/'stereo_depth_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(dest/'stereo_depth_comparison_summary.json',dict(summaries={label:{k:float(np.mean([r[k] for r in rows if r['experiment']==label])) for k in rows[0] if k not in ['experiment','source_id']} for label in ['G1_P2_to_GS2Mesh','G2_BestCleaned_to_GS2Mesh']},interpretation='Stereo depth and mask support differences are reconstructed estimates, not scan accuracy. Shared settings and fresh independent RGB/depth caches verified. G2 additionally excludes semantic marker; leaf-region evaluation is needed to separate that effect.'))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
