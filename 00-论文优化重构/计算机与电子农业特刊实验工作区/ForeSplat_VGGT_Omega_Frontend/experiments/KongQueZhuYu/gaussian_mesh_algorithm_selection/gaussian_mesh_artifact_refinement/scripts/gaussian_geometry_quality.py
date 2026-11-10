from common import *
from scipy import ndimage
from skimage.morphology import skeletonize
from evaluate_meshes import metrics
import csv
def main():
    d=ROOT/'task0_gaussian_cleaning/best_cleaned';rows=[];index=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text())
    for item in index:
        sid=item['source_id'];name=item['camera']['name'];cached=ROOT/'comparisons/gaussian_geometry_rows'/f'{sid:04d}.json'
        if cached.exists():rows.append(json.loads(cached.read_text()));continue
        with np.load(SEL/'P2_GGGS_Cleaning/surface_inputs/rendered'/f'{name}.npz') as z:fg=z['foreground_mask'].copy();ad=z['depth_surface'].copy();aa=z['alpha'].copy();an=z['native_normal'].copy()
        with np.load(d/'historical_208/rendered'/f'{name}.npz') as z:bd=z['depth_surface'].copy();ba=z['alpha'].copy();bn=z['native_normal'].copy()
        valid=fg&(aa>.5)&(ba>.5)&(ad>0)&(bd>0);delta=np.abs(bd[valid]-ad[valid]);nn=np.linalg.norm(an,axis=2)*np.linalg.norm(bn,axis=2);nv=valid&(nn>1e-6);angle=np.degrees(np.arccos(np.clip((an*bn).sum(2)[nv]/nn[nv],-1,1)));sk=skeletonize(fg);pred=ba>.5
        m=metrics(pred,fg,bd,ad,.8903337497);rec=dict(source_id=sid,thin_structure_coverage_3px=float(ndimage.binary_dilation(pred,iterations=3)[sk].mean()),depth_delta_median=float(np.median(delta)),depth_delta_p95=float(np.quantile(delta,.95)),depth_shift_gt2voxel_fraction=float((delta>.0034778662).mean()),normal_delta_median=float(np.median(angle)),normal_delta_p95=float(np.quantile(angle,.95)),foreground_alpha50_lost_px=int((fg&(aa>.5)&~pred).sum()),**{k:v for k,v in m.items() if isinstance(v,(int,float))});rows.append(rec);write(cached,rec);print('Gaussian geometry',sid,rec['depth_shift_gt2voxel_fraction'],flush=True)
    with (ROOT/'comparisons/gaussian_geometry_quality.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'comparisons/gaussian_geometry_quality_summary.json',dict(means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='source_id'},scope='Historical Scene R5 vs original P2, same208 cameras; skeleton of RAP foreground includes pot/marker, a thin-structure proxy. Depth disagreement is not scan error.'))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
