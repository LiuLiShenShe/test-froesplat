"""Accept real experiment artifacts and matched views after full execution."""
import csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from surface_protocol import ROOT,DUAL,write

METHODS=['M1_Omega_NKSR','M2_Standard_NKSR','M3_E10_GGGS_NKSR','M4_E10_GGGS_TSDF','M5_E11_2DGS_TSDF','E8_GGGS_NKSR'] + (['E11_2DGS_NKSR'] if (ROOT/'meshes/supplementary/E11_2DGS_NKSR/complete.json').exists() else [])
def table(path):return list(csv.DictReader(path.open()))
def ply_count(path):
    with path.open('rb') as f:
        counts={}
        for line in f:
            value=line.decode('ascii').strip()
            if value.startswith('element '):_,kind,number=value.split();counts[kind]=int(number)
            if value=='end_header':break
        else:raise AssertionError(f'No PLY header: {path}')
    return counts
def npz_shapes(path):
    out={}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version=np.lib.format.read_magic(f)
                assert version in [(1,0),(2,0)],(path,name,version)
                reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,order,dtype=reader(f)
                out[name[:-4]]=dict(shape=list(shape),dtype=str(dtype))
    return out
def main():
    expected=[int(r['source_index']) for r in table(DUAL/'shared_manifest.csv')];assert len(expected)==208 and len(set(expected))==208
    training=json.loads((ROOT/'E11_2DGS_Corrected/training_complete.json').read_text());assert training['iterations']==30000 and training['seed']==0 and training['physical_gpu']==1
    count=ply_count(ROOT/'E11_2DGS_Corrected/checkpoints/point_cloud/iteration_30000/point_cloud.ply')['vertex'];assert count==training['surfel_count']==248629
    assert int(table(ROOT/'E11_2DGS_Corrected/train_metrics.csv')[-1]['iteration'])==30000
    optional=ROOT/'meshes/supplementary/E11_2DGS_NKSR'
    if not (optional/'complete.json').exists():
        resource=json.loads((optional/'RESOURCE_LIMIT.json').read_text())
        assert resource['status']=='optional_resource_limited' and len(resource['attempts'])==2 and not resource['successful_mesh']
    checks=[]
    for exp in ['E8','E10','E11']:
        name=f'{exp.lower()}_{"2dgs" if exp=="E11" else "gggs"}_surface';folder=ROOT/'surface_inputs'/name
        index=json.loads((folder/'render_index.json').read_text());assert [r['source_id'] for r in index]==expected
        meta=json.loads((folder/'render_complete.json').read_text());assert meta['status']=='complete' and meta['views']==208 and meta['physical_gpu']==1 and meta['all208_GT_and_Mask_hashes_match_E10_or_E8']
        old={int(r['source_id']):r for r in table(DUAL/'render_comparison'/('E10' if exp=='E11' else exp)/'per_view_metrics.csv')}
        for row in index:
            c=row['camera'];assert (c['width'],c['height'])==(1080,1920)
            assert row['gt_sha256']==old[row['source_id']]['gt_sha256'] and row['mask_sha256']==old[row['source_id']]['mask_sha256']
            shapes=npz_shapes(Path(row['path']))
            for key in ['render_raw','gt_rgb','native_normal']:assert shapes[key]['shape']==[1920,1080,3]
            for key in ['depth_surface','alpha','foreground_mask']:assert shapes[key]['shape']==[1920,1080]
        assert json.loads((folder/'complete.json').read_text())['status']=='complete'
        checks.append(dict(experiment=exp,views=208,matched_GT_and_masks=True,native_arrays=True))
    for name,number in [('omega_direct',13640966),('standard_direct',7794589)]:
        folder=ROOT/'surface_inputs'/name
        meta=json.loads((folder/'complete.json').read_text())
        assert meta['raw_point_count']==number==ply_count(folder/'original_foreground.ply')['vertex']
        assert sum(r['full_foreground_points'] for r in meta['frames'])==number
        shapes=npz_shapes(folder/'nksr_input.npz')
        assert shapes['xyz']['shape']==[meta['nksr_points'],3] and shapes['normal']['shape']==[meta['nksr_points'],3]
        assert [r['source_id'] for r in meta['frames']]==expected
    evaluation=json.loads((ROOT/'E11_2DGS_Corrected/evaluation_complete.json').read_text());assert evaluation['status']=='complete' and evaluation['views']==208
    assert [int(r['source_id']) for r in table(ROOT/'E11_2DGS_Corrected/per_view_metrics.csv')]==expected
    meshes=[]
    for method in METHODS:
        folder=ROOT/'meshes'/method if method.startswith('M') else ROOT/'meshes/supplementary'/method
        paths=[folder]
        if method.startswith(('M4','M5')):paths.append(folder/'supplementary_alpha_filtered')
        for path in paths:
            meta=json.loads((path/'complete.json').read_text());assert meta['status']=='complete'
            ev=json.loads((path/'evaluation/complete.json').read_text());assert ev['status']=='complete' and ev['views_per_stage']==208
            for stage in ['raw','post']:
                counts=ply_count(path/f'{stage}.ply');assert counts['vertex']==meta[f'{stage}_vertices'] and counts['face']==meta[f'{stage}_faces'] and counts['face']>0
                assert [int(r['source_id']) for r in table(path/'evaluation'/f'{stage}_per_view.csv')]==expected
                assert len(list((path/'evaluation/views').glob(f'{stage}_*.npz')))==208
            assert (path/'components_and_removals.npz').exists()
            if 'NKSR' in method:assert meta['physical_gpu']==1 and meta['weight_sha256']=='c2edb6437d6485c2ed48a756523849628f0c34e232c601113ff657db8714e955'
            meshes.append(dict(method=method,path=str(path),raw_faces=meta['raw_faces'],post_faces=meta['post_faces'],evaluated_views=208))
    for name in ['rendering_comparison.csv','mesh_structure_comparison.csv','mesh_reprojection_comparison.csv','mesh_scan_comparison.csv']:assert (ROOT/'comparisons'/name).exists()
    for name in ['E11_BACKEND_ANALYSIS.md','MESH_SELECTION_ANALYSIS.md','FINAL_PIPELINE_DECISION.md','E11_2DGS_Corrected/RUN_REPORT.md']:assert (ROOT/name).is_file()
    protocol=json.loads((ROOT/'comparisons/visualizations/protocol.json').read_text())
    assert protocol['source_ids']==[0,26,53,79,105,131,157,183] and protocol['raw_and_post']
    visual=ROOT/'comparisons/visualizations'
    for kind in ['photo_whole','photo_leaf','photo_edge','photo_error','photo_alpha','photo_depth','photo_normal']:
        for group in [1,2]:assert (visual/f'{kind}_{group}.png').is_file()
    for kind in ['mesh_whole','mesh_leaf','mesh_depth','mesh_normal','mesh_silhouette','mesh_edge','mesh_edge_depth','mesh_edge_normal','mesh_edge_silhouette']:
        for stage in ['raw','post']:
            for group in [1,2,3,4]:assert (visual/f'{kind}_{stage}_{group}.png').is_file()
    from PIL import Image
    for name in ['photo_whole_2.png','mesh_normal_post_3.png','mesh_edge_depth_raw_4.png','mesh_same_frontend_raw_4.png']:
        with Image.open(visual/name) as im:assert im.width*im.height<=1200000
    provenance=json.loads((ROOT/'configs/provenance.json').read_text());provenance['delivered_script_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts').glob('*.py'))};write(ROOT/'configs/provenance.json',provenance)
    write(ROOT/'configs/DELIVERY_VERIFICATION.json',dict(status='passed',E11_iterations=30000,E11_surfels=count,views=checks,mesh_artifacts=meshes,main_meshes=5,supplementary_meshes=len(meshes)-5,optional_resource_limit=json.loads((ROOT/"meshes/supplementary/E11_2DGS_NKSR/RESOURCE_LIMIT.json").read_text()) if (ROOT/"meshes/supplementary/E11_2DGS_NKSR/RESOURCE_LIMIT.json").exists() else None,all_raw_and_post_preserved=True,all208_reprojection=True,reported_scan_accuracy=False,source_ids=expected))
    print('Full real experiment delivery verification passed.')

if __name__=='__main__':main()
