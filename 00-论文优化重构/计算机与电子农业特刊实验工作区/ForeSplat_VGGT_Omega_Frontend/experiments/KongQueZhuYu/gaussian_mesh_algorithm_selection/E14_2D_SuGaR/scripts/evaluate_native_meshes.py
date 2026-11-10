"""Historical M1–M5 evaluation functions on native 7K pre/post, initial30K and refined meshes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
import sys,json,types,time,traceback,hashlib
from pathlib import Path
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT.parent.parent/'geometry_mesh_final_selection'
sys.path.append(str(HIST/'scripts'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    early=ROOT/'prestarted_mesh_evaluation.json'
    if early.is_file():
        earlier=json.loads(early.read_text()).get('pid')
        if earlier and earlier!=os.getpid():
            proc=Path('/proc')/str(earlier)
            while proc.exists():
                try:
                    state=(proc/'stat').read_text().split(') ',1)[1].split()[0]
                    command=(proc/'cmdline').read_bytes()
                except FileNotFoundError:break
                if state=='Z' or b'evaluate_native_meshes.py' not in command:break
                print('Waiting for the previously started same-stage Mesh evaluation',earlier,flush=True)
                time.sleep(5)
    marker=ROOT/'mesh_evaluation_complete.json'
    if marker.exists() and json.loads(marker.read_text())['status']=='complete':return
    assert (ROOT/'native_render_mesh_complete.json').exists()
    cams=json.loads((ROOT/'runtime_camera_validation.json').read_text())['records']
    inp=ROOT/'surface_inputs';inp.mkdir(exist_ok=True)
    for name,stage in [('e14_surface','final_30000'),('e14_prune_reference','cluster_before')]:
        folder=ROOT/'stage_renders'/stage
        write(folder/'camera_runtime.json',dict(records=cams))
        link=inp/name
        if not link.exists():link.symlink_to(folder,target_is_directory=True)
    scale=json.loads((HIST/'configs/standard_spatial_scale.json').read_text())
    refined_marker=ROOT/'refinement_complete.json'
    if refined_marker.exists() and json.loads(refined_marker.read_text()).get('status')=='complete':
        refined=json.loads(refined_marker.read_text());refined['scale']=scale
        write(ROOT/'refinement/mesh_post/complete.json',refined)
    source_path=HIST/'scripts/evaluate_meshes.py';source=source_path.read_text()
    # Only the stage-loop list is adapted to permit truthful missing/empty artifact handling.
    # structure, cast, metrics, boundary scores, masks, and their equations remain byte-for-byte.
    assert "for stage in ['raw','post']:" in source
    module=types.ModuleType('historical_mesh_eval');module.__file__=str(source_path)
    exec(compile(source.replace("for stage in ['raw','post']:","for stage in MESH_STAGES:"),str(source_path),'exec'),module.__dict__)
    module.ROOT=ROOT
    module.METHOD_INPUT.update(E14_cluster_before='e14_prune_reference',E14_cluster_after='e14_prune_reference',E14_initial='e14_surface',E14_refined='e14_surface')
    started=time.time();results=[]
    methods=[('E14_cluster_before',ROOT/'cluster_pruning/cluster_before_mesh'),('E14_cluster_after',ROOT/'cluster_pruning/cluster_after_mesh'),('E14_initial',ROOT/'initial_mesh'),('E14_refined',ROOT/'refinement/mesh_post')]
    for method,directory in methods:
        record=dict(method=method,directory=str(directory.resolve()),status='failed')
        try:
            if not (directory/'complete.json').is_file():raise FileNotFoundError('Mesh stage did not produce a completion record')
            available=[];missing=[]
            for stage in ['raw','post']:
                path=directory/f'{stage}.ply'
                if not path.is_file():missing.append(dict(stage=stage,reason='No actual Mesh file'));continue
                mesh=o3d.io.read_triangle_mesh(str(path))
                if not len(mesh.vertices) or not len(mesh.triangles):missing.append(dict(stage=stage,reason='Actual file has no triangle surface'));continue
                available.append(stage)
            if not available:raise ValueError('Neither raw nor post Mesh contains a valid triangle surface')
            module.MESH_STAGES=available
            sys.argv=['evaluate_meshes.py','--mesh-dir',str(directory),'--method',method]
            module.main()
            record.update(status='complete' if len(available)==2 else 'partial',evaluated_stages=available,unavailable_stages=missing,evaluation=str((directory/'evaluation').resolve()))
        except Exception as err:
            record.update(reason=str(err),traceback=traceback.format_exc())
        results.append(record)
        write(directory/'evaluation_status.json',record) if directory.is_dir() else None
        print('E14 mesh evaluation',method,record['status'],flush=True)
    status='complete' if all(x['status']=='complete' for x in results) else 'partial'
    write(marker,dict(status=status,views_per_available_stage=208,methods=results,elapsed_seconds=time.time()-started,metric_source=str(source_path.resolve()),metric_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),adaptation='Only raw/post stage list changes when files are unavailable; all original metric equations preserved',reference='7K before/after use the SAME7K pre-pruning median depth;30K/refined use the SAME30K median depth. All same RAP/cameras; no scan',pruning_attribution='Same-iteration comparison isolates DBSCAN from later2DGS optimization'))
if __name__=='__main__':main()

