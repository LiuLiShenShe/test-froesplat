"""Generate the final queue's exact per-file visual and geometry caches now.

CPU only, with an execution lock. Running this while E13 trains does not
retrain anything, and finalization reuses these identical completed outputs.
"""
import csv
import hashlib
import json
import subprocess
import fcntl
import os
import time
from pathlib import Path
from queue_io import write_status

OUT=Path(__file__).resolve().parents[1]
OLD=OUT.parent/'geometry_mesh_final_selection'
PY='/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python'
lock=(OUT/'available_mesh_visual_execution.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=list(csv.DictReader((OUT/'artifact_index/ALL_ARTIFACTS.csv').open()))
seen=set();supplements=[]
for item in rows:
    if item['artifact_type'] not in ['Mesh PLY','Mesh OBJ','Textured OBJ']:continue
    if item['status'].startswith(('exists_empty_failed','exists_archived_UV_seam')):continue
    if item['experiment_id'] in ['E1','E2','E3','E4','E5','E6']:continue
    mesh=Path(item['absolute_path']).resolve(strict=True)
    if str(mesh) in seen:continue
    seen.add(str(mesh));eid=item['experiment_id'];alias=eid+'_'+hashlib.sha256(str(mesh).encode()).hexdigest()[:12]
    inp=OLD/'surface_inputs/e8_gggs_surface' if eid=='M1' else OLD/'surface_inputs/e10_gggs_surface'
    if eid in ['P4','M5']:inp=OLD/'surface_inputs/e11_2dgs_surface'
    if eid in ['P2','E12','E13']:inp=OUT/{'P2':'P2_GGGS_Cleaning','E12':'E12_GGGS_EO','E13':'E13_QGS'}[eid]/'surface_inputs'
    if eid=='E14':inp=OUT/'E14_2D_SuGaR/stage_renders/final_30000'
    target=OUT/'comparison/visualizations/mesh'/alias
    commands=[]
    if not (target/'complete.json').is_file():
        commands.append(('views',[OUT/'E14_2D_SuGaR/scripts/render_mesh_geometry_visuals.py','--mesh',mesh,'--input',inp,'--out',target,'--normal','camera']))
    geometry=OUT/'comparison/surface_supplement'/alias
    if '_vertex_color.ply' not in mesh.name and not (geometry/'supplement_complete.json').is_file():
        args=[OUT/'P3_GS2Mesh/scripts/mesh_surface_supplement.py','--mesh',mesh,'--out',geometry,'--frontend','omega' if eid=='M1' else 'standard','--mode','all','--support-cap','200000','--queries','3000','--seed','0','--workers','4']
        cache=mesh.parent/'unified_evaluation/views'
        if not cache.is_dir():cache=mesh.parent/'evaluation/views'
        stage={'recon_init.ply':'raw','recon.ply':'binary','recon_post.ply':'post'}.get(mesh.name,mesh.stem)
        camera=inp/'camera_runtime.json'
        if cache.is_dir() and camera.is_file() and all((cache/f'{stage}_{sid:04d}.npz').is_file() for sid in [0,26,53,79,105,131,157,183]):
            args+=['--camera-index',camera,'--evaluation-dir',cache.parent,'--stage',stage]
        commands.append(('geometry',args))
    for stage,args in commands:
        cmd=['uv','run','--no-project','--python',PY,'python',*map(str,args)]
        write_status(OUT/'AVAILABLE_MESH_VISUAL_STATUS.json',dict(status='running',alias=alias,stage=stage,mesh=str(mesh),pid=os.getpid()))
        with (OUT/'logs'/f'available_{alias}_{stage}.log').open('a') as log:
            started=time.time()
            rc=subprocess.run(cmd,cwd=OUT,stdout=log,stderr=subprocess.STDOUT).returncode
        with (OUT/'actual_commands.jsonl').open('a') as journal:
            journal.write(json.dumps(dict(stage='available_'+alias+'_'+stage,command=cmd,returncode=rc,wall_seconds=time.time()-started))+'\n')
        if rc:
            write_status(OUT/'AVAILABLE_MESH_VISUAL_STATUS.json',dict(status='failed',alias=alias,stage=stage,returncode=rc))
            raise SystemExit(rc)
    if (geometry/'supplement_complete.json').is_file():
        supplements.append(dict(experiment_id=eid,mesh=str(mesh),stage=item['stage'],summary=str((geometry/'supplement_complete.json').resolve()),roughness_csv=str((geometry/'fixed_scale_roughness.csv').resolve()),boundary_json=str((geometry/'boundary_complete.json').resolve()),visualization=str((target/'complete.json').resolve())))
    print(alias,'actual mesh caches complete',flush=True)
write_status(OUT/'comparison/SURFACE_SUPPLEMENT_INDEX.json',supplements)
write_status(OUT/'AVAILABLE_MESH_VISUAL_STATUS.json',dict(status='complete_available',mesh_files=len(seen),supplemented_files=len(supplements),pid=os.getpid()))
