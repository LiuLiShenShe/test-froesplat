"""P3 actual depth-stage support audit; proxy gaps are not labelled physical leaf holes."""
import csv,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1]
def main():
    start=time.time()
    doc=json.loads((ROOT/"stereo/camera_data.json").read_text())
    rows=[]
    for i,pair in enumerate(doc["pairs"]):
        folder=ROOT/"stereo"/f"{i:03}";model=folder/"out_DLNR_Middlebury"
        if not (model/"complete.json").exists():continue
        mask=np.load(folder/"left_mask.npy").astype(bool)
        z=np.load(model/"depth.npy");consistent=np.load(model/"occlusion_mask.npy").astype(bool)
        good=mask&np.isfinite(z)&(z>0)
        supported=good&consistent
        interior=mask&(ndimage.distance_transform_edt(mask)>5)
        native=supported&(z>=4*doc["baseline"])&(z<=20*doc["baseline"])
        shared=supported&(z>=4*doc["baseline"])&(z<=1.5692457556724548)
        n=int(mask.sum());inner=int(interior.sum())
        rows.append(dict(source_id=pair["left"]["source_id"],foreground_pixels=n,foreground_interior_gt5px_pixels=inner,
             valid_depth_fraction=float(good.sum()/max(n,1)),stereo_consistent_foreground_fraction=float(supported.sum()/max(n,1)),
             stereo_interior_missing_fraction=float((interior&~supported).sum()/max(inner,1)),
             native_range_supported_fraction=float(native.sum()/max(n,1)),
             shared_range_supported_fraction=float(shared.sum()/max(n,1)),
             native_range_interior_missing_fraction=float((interior&~native).sum()/max(inner,1)),
             shared_range_interior_missing_fraction=float((interior&~shared).sum()/max(inner,1)),
             depth_fg_p05=float(np.quantile(z[good],.05)) if good.any() else None,
             depth_fg_median=float(np.median(z[good])) if good.any() else None,
             depth_fg_p95=float(np.quantile(z[good],.95)) if good.any() else None))
    if rows:
        with (ROOT/"DEPTH_STAGE_SUPPORT.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status="complete" if len(rows)==208 else "partial",views=len(rows),wall_seconds=time.time()-start,
       meaning="Actual stereo/fusion-input support under RAP; >5px interior proxy missing area, not confirmed3D holes",
       native_depth_range=[4*doc["baseline"],20*doc["baseline"]],
       shared_depth_range=[4*doc["baseline"],1.5692457556724548],
       foreground_scope="Existing RAP includes plant,pot,reference marker; no new plant-only segmentation")
    if rows:
        for name in ["valid_depth_fraction","stereo_consistent_foreground_fraction","stereo_interior_missing_fraction",
                     "native_range_supported_fraction","shared_range_supported_fraction",
                     "native_range_interior_missing_fraction","shared_range_interior_missing_fraction"]:
            summary[name+"_mean_views"]=float(np.mean([r[name] for r in rows]))
    (ROOT/"DEPTH_STAGE_SUPPORT.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(summary,flush=True)
if __name__=="__main__":main()

