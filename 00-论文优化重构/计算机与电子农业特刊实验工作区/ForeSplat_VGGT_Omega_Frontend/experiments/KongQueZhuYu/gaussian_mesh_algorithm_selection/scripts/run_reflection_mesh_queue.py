"""Complete spatial reflection-to-mesh evidence as actual models become ready."""
import os
import subprocess
import time
import fcntl
from pathlib import Path
from queue_io import read_status, write_status

OUT = Path(__file__).resolve().parents[1]
OLD = OUT.parent / 'geometry_mesh_final_selection'
PY = '/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'
lock = (OUT / 'reflection_mesh_execution.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
routes = [
    ('M4', 'E10', OLD/'meshes/M4_E10_GGGS_TSDF/post.ply', OLD/'surface_inputs/e10_gggs_surface'),
    ('P1', 'E10', OUT/'P1_GGGS_Tetrahedra/recon_post.ply', OLD/'surface_inputs/e10_gggs_surface'),
    ('P2_TSDF', 'P2', OUT/'P2_GGGS_Cleaning/TSDF/post.ply', OUT/'P2_GGGS_Cleaning/surface_inputs'),
    ('P2_Tetra', 'P2', OUT/'P2_GGGS_Cleaning/Tetrahedra/recon_post.ply', OUT/'P2_GGGS_Cleaning/surface_inputs'),
    ('P3_shared', 'E10', OUT/'P3_GS2Mesh/meshes_shared_depth_range/post.ply', OLD/'surface_inputs/e10_gggs_surface'),
    ('P4_median', 'E11', OUT/'P4_2DGS_Depth_Mesh/median_mask_alpha/post.ply', OLD/'surface_inputs/e11_2dgs_surface'),
    ('E12_TSDF', 'E12', OUT/'E12_GGGS_EO/TSDF/post.ply', OUT/'E12_GGGS_EO/surface_inputs'),
    ('E12_Tetra', 'E12', OUT/'E12_GGGS_EO/Tetrahedra/recon_post.ply', OUT/'E12_GGGS_EO/surface_inputs'),
    ('E13_QGS', 'E13', OUT/'E13_QGS/meshes/native_TSDF/post.ply', OUT/'E13_QGS/surface_inputs'),
    ('E14_initial', 'E14_30K', OUT/'E14_2D_SuGaR/initial_mesh/post.ply', OUT/'E14_2D_SuGaR/stage_renders/final_30000'),
    ('E14_refined', 'E14_refined_aux', OUT/'E14_2D_SuGaR/refinement/mesh_post/post.ply', OUT/'E14_2D_SuGaR/stage_renders/refined'),
]
for method, source, mesh, inp in routes:
    marker = OUT/'comparison/reflection_mesh_correspondence'/(method+'_complete.json')
    if read_status(marker).get('status') == 'complete':
        continue
    source_marker = OUT/'comparison/reflection_analysis'/(source+'_summary.json')
    while not mesh.is_file() or read_status(source_marker).get('status') != 'complete':
        write_status(OUT/'REFLECTION_MESH_QUEUE_STATUS.json', dict(status='waiting', method=method,
            mesh=str(mesh), source_marker=str(source_marker), pid=os.getpid()))
        time.sleep(5)
    cmd = ['uv', 'run', '--no-project', '--python', PY, 'python', str(OUT/'scripts/join_reflection_mesh.py'),
           '--method', method, '--source', source, '--mesh', str(mesh), '--input', str(inp)]
    write_status(OUT/'REFLECTION_MESH_QUEUE_STATUS.json', dict(status='running', method=method, command=cmd, pid=os.getpid()))
    with (OUT/'logs'/('reflection_mesh_'+method+'.log')).open('a') as logfile:
        started=time.time()
        result = subprocess.run(cmd, cwd=OUT, stdout=logfile, stderr=subprocess.STDOUT)
    import json
    with (OUT/'actual_commands.jsonl').open('a') as journal:
        journal.write(json.dumps(dict(stage='reflection_mesh_'+method,command=cmd,returncode=result.returncode,wall_seconds=time.time()-started))+'\n')
    if result.returncode or read_status(marker).get('status') != 'complete':
        write_status(OUT/'REFLECTION_MESH_QUEUE_STATUS.json', dict(status='failed', method=method, returncode=result.returncode))
        raise SystemExit(result.returncode or 1)
    print(method, 'complete', flush=True)
write_status(OUT/'REFLECTION_MESH_QUEUE_STATUS.json', dict(status='complete', methods=[r[0] for r in routes]))
