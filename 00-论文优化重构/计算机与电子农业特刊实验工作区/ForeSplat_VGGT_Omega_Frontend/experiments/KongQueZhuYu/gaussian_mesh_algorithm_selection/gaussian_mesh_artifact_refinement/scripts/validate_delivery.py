from common import *
import csv,torch,open3d as o3d
from plyfile import PlyData

def main():
    inventory=list(csv.DictReader((ROOT/'artifact_index/ALL_ARTIFACTS.csv').open()));previous=ROOT/'DELIVERY_VALIDATION.json'
    records=json.loads(previous.read_text())['records'] if previous.exists() else [];seen={Path(r['path']) for r in records}
    GaussianModel,native=setup();torch.set_num_threads(8)
    for row in inventory:
        p=Path(row['resolved_path'])
        if p in seen:continue
        seen.add(p)
        if row['kind']=='Gaussian':
            data=PlyData.read(str(p))['vertex'].data
            for k in data.dtype.names:assert np.isfinite(data[k]).all(),(p,k)
            model=GaussianModel(3,0);model.load_ply(str(p));assert len(model.get_xyz)==int(row['vertices']);sc,op=model.get_scaling_n_opacity_with_3D_filter;assert torch.isfinite(sc).all() and torch.isfinite(op).all();del model,sc,op;torch.cuda.empty_cache()
        else:
            model=o3d.io.read_triangle_mesh(str(p));assert len(model.vertices)==int(row['vertices']) and len(model.triangles)==int(row['triangles']);f=np.asarray(model.triangles);assert f.min()>=0 and f.max()<len(model.vertices);assert np.isfinite(np.asarray(model.vertices)).all();del model
        records.append(dict(path=str(p),kind=row['kind'],vertices=int(row['vertices']),triangles=int(row['triangles']),native_loaded=True));print('validated',p.name,flush=True)
    old=list(csv.DictReader((SEL/'P2_GGGS_Cleaning/surface_inputs/per_view_metrics.csv').open())) if (SEL/'P2_GGGS_Cleaning/surface_inputs/per_view_metrics.csv').exists() else []
    for name in ['historical_208','scene_208']:
        r=list(csv.DictReader((ROOT/'task0_gaussian_cleaning/best_cleaned'/name/'per_view_metrics.csv').open()));assert len(r)==208
        if old:
            prior={x['source_id']:x for x in old}
            for x in r:assert x['gt_sha256']==prior[x['source_id']]['gt_sha256'] and x['mask_sha256']==prior[x['source_id']]['mask_sha256']
    assert len(list(csv.DictReader((ROOT/'comparisons/stereo_depth_comparison.csv').open())))==416
    review=json.loads((ROOT/'comparisons/VISUAL_REVIEW.json').read_text())
    for rel,observation in [('comparisons/before_after_images/free_free_0053_-15_all_rounds.png','Read actual six-version RGB/Alpha/Depth/Normal atlas: upper black cloud removed, leaf structure remains; residual streaks visible'),('comparisons/before_after_images/training_0157_all_rounds.png','Read source157 six-version atlas: original large invisible-on-black background alpha/depth removed; plant RGB/tip preserved')]:
        review['records'].append(dict(path=str(ROOT/rel),observation=observation,actually_read=True))
    write(ROOT/'comparisons/VISUAL_REVIEW.json',review)
    write(ROOT/'DELIVERY_VALIDATION.json',dict(status='passed',unique_models_loaded=len(records),gaussian_native_loader='Original corrected GGGS GaussianModel, SH3, SG0; native filtered scales/opacities verified finite',mesh_loader='Open3D legacy TriangleMesh, finite vertices and valid triangle indices',records=records,all_historical_gt_mask_hashes_match_P2=bool(old),stereo_comparison_rows=416,interactive_viewer_session=False,quality_validation_separate='Resolved/Improved/Unresolved in FINAL report; successful file parsing does not certify clean asset quality'))
    print('Delivery validation passed',len(records),flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
