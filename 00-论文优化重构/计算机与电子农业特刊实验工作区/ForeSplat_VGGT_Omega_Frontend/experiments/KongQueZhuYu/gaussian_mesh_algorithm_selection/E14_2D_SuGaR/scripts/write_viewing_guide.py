"""Index only actual E14 model files after completion, with native/viewer format distinction."""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1]
def header(path):
    lines=[]
    with path.open('rb') as f:
        while True:
            line=f.readline()
            if not line:break
            lines.append(line.decode('ascii','replace').strip())
            if line.strip()==b'end_header':break
    return lines
def main():
    assert (ROOT/'training_complete.json').exists()
    done=json.loads((ROOT/'refinement_complete.json').read_text()) if (ROOT/'refinement_complete.json').exists() else {}
    records=[];missing=[]
    paths=[
        ('Point Cloud PLY','initialization',ROOT/'training/input.ply','RGB and prior normals, not Gaussian','MeshLab/Open3D'),
        ('Gaussian PLY','initial_native',ROOT/'training/point_cloud/iteration_1/point_cloud.ply','Native2D surfels, two scales','Native2DGS renderer'),
        ('Gaussian PLY','cluster_before',ROOT/'training/point_cloud/iteration_7000/point_cloud.ply','Native2D surfels, two scales','Native2DGS renderer'),
        ('Gaussian PLY','cluster_after',ROOT/'cluster_pruning/after.ply','Native2D surfels, two scales','Native2DGS renderer'),
        ('Checkpoint','cluster_before_state',ROOT/'cluster_pruning/before_pruning_checkpoint.pth','Exact native7K pre-DBSCAN state before optimizer.step','Native2DGS'),
        ('Checkpoint','cluster_after_state',ROOT/'cluster_pruning/after_pruning_checkpoint.pth','Exact native7K post-DBSCAN state before optimizer.step','Native2DGS'),
        ('Gaussian PLY','cluster_removed',ROOT/'cluster_pruning/removed.ply','Actual removed native2D surfels','Native2DGS renderer'),
        ('Gaussian PLY','final_30000',ROOT/'training/point_cloud/iteration_30000/point_cloud.ply','Native2D surfels, two scales','Native2DGS renderer'),
        ('Checkpoint','native_30000',ROOT/'training/chkpnt30000.pth','Native2DGS optimization checkpoint','Native2DGS'),
        ('Mesh PLY','initial_raw',ROOT/'initial_mesh/raw.ply','Native TSDF triangle Mesh, vertex colors','Blender/MeshLab/Open3D'),
        ('Mesh PLY','initial_post',ROOT/'initial_mesh/post.ply','Common components<50faces removed','Blender/MeshLab/Open3D'),
        ('Mesh PLY','initial_native_largest_cluster',ROOT/'initial_mesh/native_largest_cluster.ply','Native largest-cluster Mesh alternative','Blender/MeshLab/Open3D'),
]
    if done.get('status')=='complete':
        paths+=[
            ('Gaussian Model','refined_native',Path(done['native_model']),'Native mesh-bound SuGaR checkpoint','NativeSuGaR'),
            ('Gaussian PLY','refined_3d_export',Path(done['gaussian_ply']),'Native exported3D Gaussian, three scales','SuperSplat'),
            ('Mesh PLY','refined_raw',ROOT/'refinement/refined_raw.ply','Actual native refined OBJ geometry','Blender/MeshLab/Open3D'),
            ('Mesh PLY','refined_post',ROOT/'refinement/mesh_post/post.ply','Refined Mesh, common component rule','Blender/MeshLab/Open3D'),
            ('Textured OBJ','refined_textured',Path(done['textured_obj']),'Native OBJ withMTL/PNG','Blender/MeshLab')]
    else:
        missing.append(dict(artifact_type='Gaussian Model / Mesh / Textured OBJ',stage='SuGaR_refinement',status='unavailable',reason=done.get('reason','No actual completed refinement stage')))
    for meshstage in ['cluster_before','cluster_after']:
        for version in ['raw','post']:
            paths.append(('Mesh PLY',f'{meshstage}_{version}',ROOT/'cluster_pruning'/(meshstage+'_mesh')/(version+'.ply'),'Same-iteration native DBSCAN intervention TSDF','Blender/MeshLab/Open3D'))
    for kind,stage,path,description,viewer in paths:
        if not path.is_file():
            missing.append(dict(artifact_type=kind,stage=stage,status='unavailable',reason='No actual output file'));continue
        h=header(path) if path.suffix=='.ply' else []
        count=next((int(x.split()[-1]) for x in h if x.startswith('element vertex ')),None)
        faces=next((int(x.split()[-1]) for x in h if x.startswith('element face ')),None)
        props=[x.split()[-1] for x in h if x.startswith('property ')]
        if stage=='refined_3d_export':assert all(f'scale_{i}' in props for i in range(3))
        records.append(dict(experiment_id='E14',artifact_type=kind,stage=stage,absolute_path=str(path.resolve()),relative_path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),file_size_bytes=path.stat().st_size,vertices_or_gaussians=count,faces=faces,viewer=viewer,description=description,status='exists'))
    for dep in done.get('texture_dependencies',[]):
        path=Path(dep);assert path.is_file()
        records.append(dict(experiment_id='E14',artifact_type='Texture Material' if path.suffix=='.mtl' else 'Texture Image',stage='refined_textured',absolute_path=str(path.resolve()),relative_path=str(path.relative_to(ROOT)),file_size_bytes=path.stat().st_size,vertices_or_gaussians=None,faces=None,viewer='Blender/MeshLab together withOBJ',description='Native OBJ dependency',status='exists'))
    for r in records:
        checkpoint=ROOT/'training/chkpnt30000.pth'
        if r['stage'].startswith('cluster'):
            checkpoint=ROOT/'cluster_pruning'/('after_pruning_checkpoint.pth' if r['stage'].startswith('cluster_after') else 'before_pruning_checkpoint.pth')
        elif r['stage'].startswith('refined'):
            checkpoint=Path(done['native_model']) if done.get('native_model') else None
        elif r['stage'] in ['initialization','initial_native']:checkpoint=None
        r.update(method='2D-SuGaR',format=Path(r['absolute_path']).suffix.lstrip('.'),coordinate_system='Corrected standard VGGT/COLMAP arbitrary world gauge',source_checkpoint=str(checkpoint.resolve()) if checkpoint and checkpoint.is_file() else None)
    with (ROOT/'E14_ARTIFACTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    data=ROOT/'viewer_dataset';(data/'sparse/0').mkdir(parents=True,exist_ok=True)
    for name in ['images','masks']:
        if not (data/name).exists():(data/name).symlink_to(ROOT/'input_colmap'/name,target_is_directory=True)
    for p in (ROOT/'input_colmap/sparse/0').iterdir():
        if p.is_file() and not (data/'sparse/0'/p.name).exists():(data/'sparse/0'/p.name).symlink_to(p.resolve())
    models=ROOT/'viewer_models/pruned/point_cloud/iteration_7000';models.mkdir(parents=True,exist_ok=True)
    if not (models/'point_cloud.ply').exists():(models/'point_cloud.ply').symlink_to(ROOT/'cluster_pruning/after.ply')
    command=f'{ROOT/".venv/bin/python"} {ROOT/"official/gaussian_splatting_2d/view.py"} -s {data} -m {ROOT/"training"} -r 2 --iteration 30000 --depth_ratio 1.0 --initialization_prior ""'
    webcmd=f'{ROOT/".venv/bin/python"} {ROOT/"scripts/view_native_web.py"} --stage final --port 8864'
    lines=['# E14 actual models and viewing','',
        'Direct native browser viewer after the experiment finishes (binds only127.0.0.1, GPU1):','',chr(96)+webcmd+chr(96),'',
        'Open http://127.0.0.1:8864; on a remote server forward that port through SSH. Use --stage pruned/before/initial/refined to inspect actual corresponding models. Refined depth/normal are explicitly auxiliary. No server is started by the experiment driver.','',
        'Use --ply /actual/absolute/native2DGS.ply --label E11 for an existing E11 native2D surfel PLY. The viewer preserves its representation; E11 training losses/priors remain E11, and this is not an E14 model conversion.','',
        'Initial/pruned/30K PLYs are native2D surfels with two scales. SuperSplat3DGS import does not provide an equivalent rendering. Use the official2DGS renderer/network viewer; native fixed/freeview images are in visual_views/surfel/.','',
        'The refinement export has three Gaussian scales and is compatible with3DGS/SuperSplat. The .pt is the original mesh-bound SuGaR representation; Gaussian PLY alone does not carry mesh binding.','',
        'Native2DGS rendering command for the actual30K model (requires a native network viewer to connect):','',chr(96)+command+chr(96),'',
        'For actual7000 post-pruning PLY, use viewer_models/pruned as -m and --iteration 7000. The lightweight viewer dataset keeps corrected cameras/RGB and omits dense prior loading. The GUI camera adapter preserves supplied projection/pixel centres.','',
        'Load the native OBJ together with its sibling MTL and PNG. Use geometry-only PLY for gray/wireframe/normal comparison; initial Mesh has vertex colors rather than UV texture.','',
        'Actual fixed8 Mesh gray/normal/vertex-color/visible-face-edge wireframes are in visual_views/mesh/. Only refined_native_textured_OBJ provides native UV texture; missing vertex color/texture is explicitly unavailable.','',
        'Every file retains standardVGGT/COLMAP arbitrary world gauge. If the model appears upside down, orbit the view180° aboutX and keep the original geometry unchanged. This view rotation does not establish metric scale.','',
        'E14 inspection order: initialization → Gaussian iteration1 → iteration7000 before/after plusremoved →30000 → initialraw/post Mesh → refinementGaussian/PT → refinedgrayMesh → native texturedOBJ.','',
        'RefinedGaussian nativeDepth/Normal are unavailable from the original3DGS API. Centre-Z expected-depth/axis-normal maps are explicitly auxiliary. Geometry evaluation uses actual refinedMesh raycastDepth/Normal.','',
        '| Artifact | Stage | Actual absolute path | Bytes | Viewer |','|---|---|---|---:|---|']
    for r in records:lines.append(f"|{r['artifact_type']}|{r['stage']}|{r['absolute_path']}|{r['file_size_bytes']}|{r['viewer']}|")
    if missing:
        lines+=['','Unavailable model stages: '+json.dumps(missing,ensure_ascii=False),'']
    lines+=['','Sources: actual existing outputs; [official2D-SuGaR](https://github.com/prajwalcr/2d-sugar/tree/e62770ec6ab50076ebb6c0e3b4ac39f777f14d27), actual view.py.']
    (ROOT/'HOW_TO_VIEW_E14.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'artifact_guide_complete.json').write_text(json.dumps(dict(status='complete' if not missing else 'partial',records=len(records),all_paths_exist=True,unavailable_stages=missing,guide=str(ROOT/'HOW_TO_VIEW_E14.md'),index=str(ROOT/'E14_ARTIFACTS.csv')),indent=2)+'\n')
if __name__=='__main__':main()

