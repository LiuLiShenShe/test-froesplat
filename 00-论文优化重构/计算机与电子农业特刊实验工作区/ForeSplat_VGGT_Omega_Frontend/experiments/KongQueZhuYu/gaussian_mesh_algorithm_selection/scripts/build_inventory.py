"""Index only real files, classify PLY by actual properties (not filename)."""
import csv,json,os,re,zipfile
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1]
PLANT=OUT.parent; BASE=PLANT.parents[1]; REPO=Path('/data/fj/F2DMAS')
DUAL=PLANT/'geometry_rgb_dual_frontend'; MESH=PLANT/'geometry_mesh_final_selection'
STANDARD=BASE.parent/'ForeSplat_Standard_VGGT_Frontend/experiments/KongQueZhuYu/standard_vggt_comparison'
HISTORY={
'E1':PLANT/'E1_GGGS_Baseline','E2':PLANT/'E2_GGGS_EO',
'E3':PLANT/'E3_GGGS_Foreground','E4':PLANT/'E4_ForeSplat_GGGS',
'E5':STANDARD/'E5_Standard_VGGT_GGGS_Baseline','E6':STANDARD/'E6_2DGS_Standard_VGGT',
'E7':DUAL/'E7_Omega_Original_Interface','E8':DUAL/'E8_Omega_Corrected_Interface',
'E9':DUAL/'E9_Standard_Original_Interface','E10':DUAL/'E10_Standard_Corrected_Interface',
'E11':MESH/'E11_2DGS_Corrected','E12':OUT/'E12_GGGS_EO',
'E13':OUT/'E13_QGS','E14':OUT/'E14_2D_SuGaR',
'P1':OUT/'P1_GGGS_Tetrahedra','P2':OUT/'P2_GGGS_Cleaning',
'P3':OUT/'P3_GS2Mesh','P4':OUT/'P4_2DGS_Depth_Mesh'}
FIELDS='experiment_id method artifact_type stage format absolute_path relative_path file_size_bytes vertices_or_gaussians faces coordinate_system source_checkpoint viewer status'.split()
def ply_header(p):
    nv=nf=0;props=[];element=None;has_faces=False
    with p.open('rb') as f:
        for _ in range(300):
            s=f.readline().decode('ascii','replace').strip()
            if s.startswith('element '):
                _,element,count=s.split()
                if element=='vertex':nv=int(count)
                if element=='face':nf=int(count);has_faces=True
            if s.startswith('property ') and element=='vertex':props.append(s.split()[-1])
            if s=='end_header':break
        else:raise ValueError('invalid PLY header '+str(p))
    return nv,nf,props,has_faces
def writecsv(p,rows,fields):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
rows=[];seen=set()
def artifact_coordinates(e,p,kind,stage,gauge):
    """Describe the stored quantity, rather than assigning every map world XYZ."""
    world=f'{gauge} world gauge; source-specific saved W2C; uncalibrated scale'
    array=re.search(r'NPZ array ([A-Za-z0-9_]+)',stage)
    channel=array.group(1) if array else p.stem
    if kind=='Exposure Parameters':
        return 'Per-camera RGB affine 3x4 correction; image-color space, no spatial coordinates'
    if kind in ['RGB Map','Alpha Map','Stereo Confidence']:
        return 'Source-camera image pixel grid; RGB/alpha/confidence values, no world XYZ'
    if kind in ['Material','Texture']:
        return 'OBJ material / UV texture coordinates; associated mesh retains its own world gauge'
    if kind=='Native Viewer':
        return 'Viewer source code; camera-only view transforms, no stored experimental geometry'
    if kind=='Depth Map':
        if e=='E14' and 'estimated_dense_depth' in p.parts:
            return ('Metric3Dv2 predicted camera-axis Z on original input pixel grid; focal-decanonicalized BEFORE '
                    'per-image COLMAP-track median scaling in input_colmap/prior_depth_scale.json; not VGGT world XYZ or measured ground truth')
        detail='; AUX alpha-weighted Gaussian-center Z, not native surface depth' if channel.endswith('_aux') else ''
        return f'Source-camera image pixel grid; camera-axis Z in {gauge} arbitrary scale, not world XYZ'+detail
    if kind=='Normal Map':
        if e=='E14' and 'estimated_dense_normal' in p.parts:
            return ('Metric3Dv2 camera-coordinate predicted normal on original input pixel grid; BEFORE native '
                    'camera-to-world rotation and normalization; no spatial length units')
        native_world=e in ['E6','E11','P4','E14']
        frame='camera' if 'camera' in channel else 'world' if ('world' in channel or channel.endswith('_aux') or (channel=='native_normal' and native_world)) else 'camera'
        detail='; AUX smallest-scale Gaussian axis, not native surface normal' if channel.endswith('_aux') else '; native accumulations may be alpha-weighted; unit channels are explicitly named'
        return f'Source-camera image pixel grid; {frame}-coordinate normal vectors in {gauge} camera/world orientation; no spatial length units'+detail
    return world
def add(e,p,stage='',method='',kind=None,source='',artifact_status=''):
    if not p.is_file():return
    real=p.resolve()
    nv=nf='';viewer='native loader'
    if p.suffix=='.ply':
        nv,nf,props,has_faces=ply_header(real)
        kind='Mesh PLY' if has_faces else ('Gaussian PLY' if 'opacity' in props and 'f_dc_0' in props and 'scale_0' in props else 'Point Cloud PLY')
        if has_faces and not nf:
            artifact_status='exists_empty_failed_Mesh_attempt_no_triangle_surface'
        scales=[x for x in props if x.startswith('scale_')]
        if not nf and len(scales)==6 and kind=='Gaussian PLY':kind='Gaussian Model'
        viewer='Blender / MeshLab / Open3D' if nf else ('PLY point viewer' if kind=='Point Cloud PLY' else ('Native QGS CUDA renderer (6 scale fields)' if len(scales)==6 else ('Native 2DGS Viewer (2 scales)' if len(scales)==2 else 'SuperSplat (appearance; GGGS native renderer for exact surface)')))
    elif p.suffix=='.obj':
        with p.open(errors='replace') as stream:
            nv=nf=0
            for line in stream:
                nv += line.startswith('v ')
                nf += line.startswith('f ')
        mtls=[p.parent/line.split(maxsplit=1)[1].strip() for line in p.read_text(errors='replace').splitlines() if line.startswith('mtllib ')]
        kind='Textured OBJ' if mtls and all(m.is_file() for m in mtls) else 'Mesh OBJ'
        viewer='Blender / MeshLab (load actual MTL and texture together)' if kind=='Textured OBJ' else 'Blender / MeshLab'
    elif p.suffix in ['.pth','.pt']:kind=kind or 'Checkpoint'
    elif p.suffix=='.npz':
        kind=kind or ('Exposure Parameters' if 'exposure' in p.name else 'Gaussian Model')
        if kind in ['Quadratic Primitive Parameters','Point Cloud Index Mapping']:
            with np.load(real,allow_pickle=False) as archive:
                if kind=='Quadratic Primitive Parameters':
                    nv=int(archive['xyz'].shape[0])
                    assert archive['xyz'].shape==(nv,3) and archive['raw_scale_parameters'].shape==(nv,6)
                    viewer='NumPy native QGS geometry/curvature archive; appearance remains in indexed six-scale native PLY'
                else:
                    nv=int(archive['native_primitive_indices'].shape[0])
                    assert archive['native_primitive_indices'].ndim==1
                    viewer='NumPy row-to-native-primitive mapping for the point-cloud-only viewer export'
            artifact_status='exists_native_array_shapes_verified'
        if kind=='Gaussian Model':
            required=['xyz','features_dc','features_rest','opacity','scaling','rotation']
            shapes={}
            with zipfile.ZipFile(real) as archive:
                for name in required:
                    with archive.open(name+'.npy') as stream:
                        version=np.lib.format.read_magic(stream)
                        reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                        if reader is None:raise ValueError('Unsupported actual NPY header version '+str(version))
                        shapes[name]=reader(stream)[0]
            nv=int(shapes['xyz'][0])
            assert shapes['xyz']==(nv,3) and all(shape[0]==nv for shape in shapes.values()),('Native NPZ parameter counts differ',real,shapes)
            with np.load(real,allow_pickle=False) as archive:
                if 'num_gaussians' in archive:assert int(archive['num_gaussians'])==nv,('Native NPZ count metadata differs',real)
            viewer='NumPy parameter archive; use the corresponding indexed native Gaussian PLY for viewing'
            artifact_status=artifact_status or 'exists_native_parameter_NPY_headers_and_count_verified'
    elif p.suffix in ['.mtl','.png','.jpg']:kind=kind or 'Texture'
    else:kind=kind or 'Gaussian Model'
    key=(e,str(real),kind,stage if 'NPZ array ' in stage else '')
    if key in seen:return
    seen.add(key)
    if not source and e=='E13' and 'runs' in p.parts and kind!='Checkpoint':
        iteration=re.search(r'iteration_(\d+)',str(p))
        if iteration:
            own=next((ancestor for ancestor in p.parents if ancestor.name=='checkpoints'),None)
            candidate=own/f'chkpnt{iteration.group(1)}.pth' if own else None
            if candidate and candidate.is_file():source=str(candidate.resolve())
    if not source and e in HISTORY and kind!='Checkpoint' and not (e=='E13' and 'runs' in p.parts):
        iteration=re.search(r'iteration_(\d+)',str(p))
        iteration=iteration.group(1) if iteration else '30000'
        for candidate in [HISTORY[e]/f'checkpoints/chkpnt{iteration}.pth',HISTORY[e]/f'training/chkpnt{iteration}.pth']:
            if candidate.is_file():source=str(candidate.resolve());break
    if e in ['P1','P2','P3']:source=str((HISTORY['E10']/'checkpoints/chkpnt30000.pth').resolve())
    if e=='P4':source=str((HISTORY['E11']/'checkpoints/chkpnt30000.pth').resolve())
    if e=='E14' and (('iteration_7000' in str(real) and p.name=='point_cloud.ply') or (p.name=='before.ply' and 'cluster_pruning' in p.parts)):
        # Native Scene.save(7000) precedes DBSCAN; chkpnt7000 is saved after
        # pruning. The two represent different primitive sets.
        checkpoint=HISTORY['E14']/'cluster_pruning/before_pruning_checkpoint.pth'
        source=str(checkpoint.resolve()) if checkpoint.is_file() else ''
        stage=(stage or str(p.parent.relative_to(HISTORY[e])))+'; native7K snapshot BEFORE DBSCAN, distinct from post-prune chkpnt7000'
    if e=='E14' and 'iteration_7000' in str(real) and p.name.startswith('point_cloud_cluster-'):
        checkpoint=HISTORY['E14']/'cluster_pruning/before_pruning_checkpoint.pth'
        source=str(checkpoint.resolve()) if checkpoint.is_file() else ''
        stage=(stage or str(p.parent.relative_to(HISTORY[e])))+'; actual7K DBSCAN cluster subset exported from before-pruning native parameters; not the full before/pruned model'
    if e=='E14' and p.name in ['after.ply','removed.ply'] and 'cluster_pruning' in p.parts:
        checkpoint=HISTORY['E14']/'cluster_pruning'/('after_pruning_checkpoint.pth' if p.name=='after.ply' else 'before_pruning_checkpoint.pth')
        source=str(checkpoint.resolve()) if checkpoint.is_file() else ''
    if e=='E14' and p.name in ['before_pruning_checkpoint.pth','after_pruning_checkpoint.pth']:
        stage='Native7K '+('before' if p.name.startswith('before') else 'after')+' DBSCAN; before optimizer.step; distinct from training/chkpnt7000'
    if e=='E14' and 'stage_renders' in p.parts:
        render_stage=p.parts[p.parts.index('stage_renders')+1]
        checkpoint={'cluster_before':HISTORY[e]/'cluster_pruning/before_pruning_checkpoint.pth',
                    'cluster_after':HISTORY[e]/'cluster_pruning/after_pruning_checkpoint.pth',
                    'final_30000':HISTORY[e]/'training/chkpnt30000.pth'}.get(render_stage)
        source=str(checkpoint.resolve()) if checkpoint and checkpoint.is_file() else ''
    if e=='E14' and any(name in p.parts for name in ['estimated_dense_depth','estimated_dense_normal']):
        # These are upstream Metric3D predictions, never exports of the 2D30K checkpoint.
        source=''
    if e=='E14' and (HISTORY[e]/'refinement_complete.json').is_file():
        refinement=json.loads((HISTORY[e]/'refinement_complete.json').read_text())
        refined_paths={str(Path(refinement[k]).resolve()) for k in ['native_model','gaussian_ply','textured_obj','refined_raw_ply','refined_post_ply','refined_welded_raw_ply'] if refinement.get(k)}
        refined_paths.update(str(Path(k).resolve()) for k in refinement.get('texture_dependencies',[]))
        if str(real) in refined_paths or p.is_relative_to(HISTORY[e]/'stage_renders/refined') or p.is_relative_to(HISTORY[e]/'visual_views/refined'):
            native=Path(refinement['native_model']).resolve() if refinement.get('native_model') else None
            source=str(native) if native and native.is_file() and real!=native else ''
            stage=(stage+' / ' if stage else '')+f"Actual native SuGaR refinement {refinement.get('iterations','unrecorded')} iterations after2D30K; immediate source is refinement model"
        if p.is_relative_to(HISTORY[e]/'refinement/uv_seam_unwelded_post_attempt'):
            stage=(stage+' / ' if stage else '')+'Archived UV-seam import/post attempt; excluded from scientific comparison; canonical exact-weld raw/post in refinement/mesh_post'
            artifact_status='exists_archived_UV_seam_import_attempt' if nf else 'exists_empty_failed_Mesh_attempt_no_triangle_surface'
        if p.name=='refined_raw.ply':
            stage=(stage+' / ' if stage else '')+'Original geometry import with per-UV duplicated vertices; canonical exact-weld raw preserves triangle coordinates'
    if kind=='Point Cloud PLY' and (p.name in ['input.ply','points3D.ply'] or 'initialization' in p.parts):source=''
    if e=='P2' and p.parent==HISTORY['P2']:
        stage={'original_gaussians.ply':'Original E10 30K; actual symlink resolves to preserved source','cleaned_gaussians.ply':'Cleaned E10 30K; no retraining','removed_gaussians.ply':'Removed E10 primitives only; not a reconstructed model'}.get(p.name,stage)
    if e=='E12' and kind=='Gaussian PLY' and real==(HISTORY[e]/'checkpoints/point_cloud/iteration_30000/point_cloud.ply').resolve():
        stage='Actual E12 native GGGS+EO30K; extraction runtime aliases resolve to this same terminal file; EO matrices separate'
    if e=='E13' and ('iteration_7000' in str(p) or p.name=='chkpnt7000.pth'):stage=(stage or str(p.parent.relative_to(HISTORY[e])))+'; partial7K native QGS, not30K terminal'
    if e=='E13' and ('iteration_15000' in str(p) or p.name=='chkpnt15000.pth'):stage=(stage or str(p.parent.relative_to(HISTORY[e])))+'; partial15K native QGS, not30K terminal'
    if e=='E13' and 'runs' in p.parts:
        stage=(stage or str(p.parent.relative_to(HISTORY[e])))+'; ARCHIVED base preset, excluded from primary Mip result'
        artifact_status='exists_archived_base_preset_partial_or_failed'
    if e=='E13' and 'failure_exports' in p.parts:
        provenance=p.parent/'PROVENANCE.json'
        info=json.loads(provenance.read_text()) if provenance.is_file() else {}
        stage=f"Failed transient native QGS state at iteration {info.get('failed_iteration','unknown')}; not a completed formal model"
        artifact_status='exists_failed_native_state_not_terminal'
        source=str(Path(info['checkpoint']).resolve()) if info.get('checkpoint') else ''
        read_validation=p.parent/'NATIVE_CHECKPOINT_READ_VALIDATION.json'
        if read_validation.is_file() and json.loads(read_validation.read_text()).get('status')=='native_archive_read_failed':
            if kind=='Checkpoint':
                artifact_status='exists_failed_export_archive_unreadable_NOT_resume_checkpoint'
                viewer='Cannot load in native PyTorch; retained failed export for audit'
                stage+='; actual native ZIP reader failed, original capture metadata does not certify stored payload'
            elif kind=='Gaussian Model':
                stage+='; native PLY saved directly from failed live model; companion PTH is unreadable'
            source=''
        ply_validation=p.parent/'PLY_NUMERICAL_VALIDATION.json'
        if kind=='Gaussian Model' and ply_validation.is_file():
            actual=json.loads(ply_validation.read_text())
            stage+=f"; actual PLY retains {actual['raw_nonfinite_primitives']} nonfinite native primitives, full payload read and SHA recorded"
            artifact_status='exists_failed_native_PLY_with_recorded_nonfinite_parameters'
    native_export=p.parent/'NATIVE_EXPORT_PROVENANCE.json'
    if e=='E13' and native_export.is_file():
        actual=json.loads(native_export.read_text())
        if kind=='Checkpoint':nv=actual['primitives']
        stage=actual['model_stage']+f"; native {actual['native_preset']} preset; {actual['nonfinite_native_primitives']} nonfinite primitives retained"
        artifact_status='exists_bitwise_verified_partial_native_Mip_state_not_terminal'
        source=actual['source_checkpoint'] if kind!='Checkpoint' else ''
    if e=='E13' and p.name=='latest_native_state.pth':
        provenance=p.with_suffix('.provenance.json')
        info=json.loads(provenance.read_text()) if provenance.is_file() else {}
        stage=f"Rolling native resume state iteration {info.get('iteration','unknown')}; not the canonical delivered terminal Checkpoint"
        if 'runs' in p.parts:
            stage+='; archived base snapshot frozen after failed attempt'
            artifact_status='exists_archived_base_preset_checkpoint'
        else:
            stage+='; overwritten during active training'
            artifact_status='exists_rolling_partial_checkpoint'
        source=''
    if e=='E13' and kind=='Gaussian Model' and p.suffix=='.ply':
        reports=[HISTORY[e]/'checkpoint_provenance_validation.json',HISTORY[e]/'partial_checkpoint_provenance_validation.json']
        record=next((r for report in reports if report.is_file() for r in json.loads(report.read_text()).get('records',[]) if r.get('native_ply_path')==str(real)),None)
        if record and record.get('native_nonfinite_primitives',0):
            stage=(stage+' / ' if stage else '')+f"Original unfiltered native QGS; {record['native_nonfinite_primitives']} nonfinite primitives retained"
            artifact_status='exists_native_model_with_recorded_nonfinite_primitives'
        if real==(HISTORY[e]/'checkpoints/point_cloud/iteration_30000/point_cloud.ply').resolve():
            stage='Actual native QGS Mip30K; '+stage
    if e=='E13' and p.name=='quadratic_centers_rgb_point_cloud.ply':
        info=HISTORY[e]/'viewer_exports/INDEX.json'
        meta=json.loads(info.read_text()) if info.is_file() else {}
        stage=f"Finite XYZRGB centers only; {meta.get('nonfinite_centers_excluded',0)} nonfinite centers excluded; no quadratic surface representation"
    if kind in ['Mesh PLY','Mesh OBJ','Textured OBJ']:
        version={'recon_init.ply':'Raw marching tetrahedra','recon.ply':'Binary Search10','recon_post.ply':'Post','recon_post_vertex_color.ply':'Post vertex-colored; geometry hash unchanged','raw.ply':'Raw','post.ply':'Post','refined_raw.ply':'Refined raw','native_largest_cluster.ply':'Official largest-component post'}.get(p.name,p.stem)
        stage=(stage+' / ' if stage else '')+version
    if kind=='Point Cloud PLY' and p.name in ['input.ply','points3D.ply']:stage=(stage+' / ' if stage else '')+'Initialization Point Cloud, not Gaussian'
    gauge='VGGT-Omega' if e in ['E1','E2','E3','E4','E7','E8','M1'] else 'standard VGGT'
    # Keep the real repository lookup path through an existing data-disk
    # symlink; the absolute delivery path remains fully resolved.
    lookup=p.absolute() if p.absolute().is_relative_to(REPO) else real
    rows.append(dict(experiment_id=e,method=method,artifact_type=kind,stage=stage or str(p.parent.relative_to(HISTORY[e])) if e in HISTORY and p.is_relative_to(HISTORY[e]) else stage,format=p.suffix[1:],absolute_path=str(real),relative_path=os.path.relpath(lookup,REPO),file_size_bytes=real.stat().st_size,vertices_or_gaussians=nv,faces=nf,coordinate_system=artifact_coordinates(e,p,kind,stage,gauge),source_checkpoint=source,viewer=viewer,status=artifact_status or ('exists_header_verified' if p.suffix=='.ply' else 'exists')))
for e,root in HISTORY.items():
    method='QGS' if e=='E13' else '2D-SuGaR' if e=='E14' else '2DGS' if e in ['E6','E11','P4'] else 'GGGS'
    if not root.exists():continue
    if e in [f'E{x}' for x in range(1,12)]:
        dirs=[root/'checkpoints',root/'gaussian',root/'mesh']
    elif e=='E13':
        dirs=[root/n for n in ['checkpoints','native_geometry','viewer_exports','meshes','surface_inputs','failure_exports','recovery','runs','retained_native_states']]
    elif e=='E14':
        # Source/vendor fixtures (e.g. PyTorch3D's cow and teapot) are not
        # KongQueZhuYu experimental models, even though their files exist.
        dirs=[root/n for n in ['training','cluster_pruning','initialization','initial_mesh','refinement','input_colmap','stage_renders']]
    else:dirs=[root]
    for d in dirs:
        if not d.exists():continue
        for walk,sub,files in os.walk(d):
            sub[:]=[n for n in sub if n not in ['official','.venv','.git','source','__pycache__','submodules','weights','adapted_source','adapted'] and not n.endswith('_official')]
            for name in files:
                p=Path(walk)/name
                if e=='E13' and name in ['quadratic_primitive_parameters.npz','center_point_native_indices.npz']:
                    kind='Quadratic Primitive Parameters' if name=='quadratic_primitive_parameters.npz' else 'Point Cloud Index Mapping'
                    add(e,p,method=method,kind=kind,stage='Native six-scale geometry parameters (not a complete appearance model)' if kind=='Quadratic Primitive Parameters' else 'Finite-center viewer rows map to actual native QGS indices')
                if p.suffix in ['.ply','.obj','.mtl','.pth','.pt'] or name in ['gaussians.npz','exposure_matrices.npz']:
                    add(e,p,method=method)
                if p.suffix in ['.npy','.npz'] and any(n in p.parts for n in ['depth','normals','depth_prior','normal_prior','estimated_dense_depth','estimated_dense_normal']):
                    add(e,p,method=method,kind='Normal Map' if any('normal' in n for n in p.parts) else 'Depth Map')
    for p in root.glob('exposure_params/*.npz'):add(e,p,stage='per-view exposure',method=method)
for d in (MESH/'meshes').glob('*'):
    if d.name.startswith('M') and d.is_dir():
        for p in d.rglob('*.ply'):add(d.name.split('_')[0],p,stage=str(p.relative_to(d)),method=d.name)
maproots=[(e,MESH/'surface_inputs'/f'{e.lower()}_{"2dgs" if e=="E11" else "gggs"}_surface') for e in ['E10','E11']]
maproots += [('P2',HISTORY['P2']/'surface_inputs'),('E12',HISTORY['E12']/'surface_inputs'),('E13',HISTORY['E13']/'surface_inputs')]
maproots += [('E14',root) for root in (HISTORY['E14']/'stage_renders').glob('*') if root.is_dir()]
maproots += [(p.name,p) for p in (OUT/'comparison/history_uniform').glob('E*') if p.is_dir()]
for e,root in maproots:
    index=root/'render_index.json'
    if index.exists():
        for r in json.loads(index.read_text()):
            path=Path(r['path'])
            with zipfile.ZipFile(path) as archive:arrays={Path(name).stem for name in archive.namelist()}
            channels=[('Alpha Map','alpha'),('RGB Map','render_raw'),('RGB Map','render_corrected')]
            if e=='E14' and root.name=='refined':
                # Native 3DGS lacks surface depth/normal. Zero sentinel arrays
                # remain in the original NPZ but are never indexed as measurements.
                channels += [('Depth Map','depth_expected_center_aux'),('Normal Map','normal_smallest_axis_aux')]
            else:
                channels += [(kind,key) for kind,keys in [('Depth Map',['depth_surface','depth_expected','depth_median','depth_median_native_raw','depth_native_mesh']),('Normal Map',['native_normal','native_normal_world','normal_world','normal_camera','normal_unit_world','normal_unit_camera','depth_normal_world','depth_normal_camera','normal_native_mesh_camera']),('Alpha Map',['alpha_native_mesh'])] for key in keys]
            for kind,key in channels:
                if key not in arrays:continue
                auxiliary=key.endswith('_aux')
                stage=f'{root.name}; source {r["source_id"]}; NPZ array {key}'+('; AUX Gaussian center/axis, not native surface depth/normal' if auxiliary else '')
                add(e,path,stage=stage,method=e,kind=kind,artifact_status='exists_npz_array_verified_auxiliary' if auxiliary else 'exists_npz_array_verified')
for modeldir in [HISTORY['P3']/'stereo']:
    for p in modeldir.rglob('*.npy'):
        if p.name in ['depth.npy','confidence.npy','occlusion_mask.npy']:add('P3',p,method='DLNR',kind='Depth Map' if p.name=='depth.npy' else 'Stereo Confidence')
for e in ['E13','E14']:
    for p in (HISTORY[e]/'scripts').glob('*viewer*'):
        if p.is_file():add(e,p,method=e,kind='Native Viewer')
    for obj in (HISTORY[e]).rglob('*.obj'):
        if any(n=='official' or n.endswith('_official') or n in ['adapted','.venv','cpu_visual_validation'] for n in obj.parts):continue
        for line in obj.read_text(errors='replace').splitlines():
            if not line.startswith('mtllib '):continue
            mtl=obj.parent/line.split(maxsplit=1)[1].strip()
            if not mtl.is_file():continue
            add(e,mtl,method=e,kind='Material')
            for line in mtl.read_text(errors='replace').splitlines():
                if line.startswith('map_Kd '):add(e,mtl.parent/line.split(maxsplit=1)[1].strip(),method=e,kind='Texture')
writecsv(OUT/'artifact_index/ALL_ARTIFACTS.csv',rows,FIELDS)
g=[r for r in rows if r['artifact_type'] in ['Gaussian PLY','Gaussian Model','Checkpoint','Exposure Parameters','Point Cloud PLY','Quadratic Primitive Parameters','Point Cloud Index Mapping'] and '208 view' not in r['stage']]
m=[r for r in rows if r['artifact_type'] in ['Mesh PLY','Mesh OBJ','Textured OBJ','Material','Texture']]
for name,data,title in [('GAUSSIAN_PLY_INDEX.md',g,'Gaussian/model files'),('MESH_FILE_INDEX.md',m,'Mesh files')]:
    content=f'# {title} — actual inventory\n\nEvery path below was resolved and exists. PLY types determined from actual property headers; ordinary XYZ is never labelled Gaussian. Six-scale QGS PLY is a native quadratic model, not a standard3DGS import. This active inventory only includes produced files; pending stages are explicitly marked. EO matrices are external to PLY.\n\n|Experiment|Stage/type|Absolute file path|Bytes|Vertices/Gaussians|Faces|Viewer|\n|---|---|---|---:|---:|---:|---|\n'
    for r in data:
        r=dict(r,stage=r['stage']+' / '+r['artifact_type'])
        r['absolute_path']=f'[{r["absolute_path"]}](<{r["absolute_path"]}>)'
        content+='|'+ '|'.join(str(r[k]) for k in ['experiment_id','stage','absolute_path','file_size_bytes','vertices_or_gaussians','faces','viewer'])+'|\n'
    for e,root in HISTORY.items():
        if e in ['E12','E13','E14','P1','P2','P3','P4'] and not any(r['experiment_id']==e for r in data):
            if name=='GAUSSIAN_PLY_INDEX.md' and e in ['P1','P3','P4']:
                base_e='E11' if e=='P4' else 'E10'
                content+=f'|{e}|Reuses the actual {base_e} 30K Gaussian above; no new Gaussian training||||||\n'
            else:content+=f'|{e}|Pending execution — no corresponding delivered file||||||\n'
    (OUT/'artifact_index'/name).write_text(content)
metrics={}
for p in [DUAL/'reconstruction_comparison.csv',MESH/'comparisons/rendering_comparison.csv']:
    for r in csv.DictReader(p.open()):metrics[r['experiment']]=(r,str(p))
old=list(csv.DictReader((PLANT/'comparison.csv').open()))
gauss=[]
gf='experiment_id frontend geometry_rgb mask_stage camera_interface gaussian_method eo_enabled iterations num_gaussians psnr_raw psnr_corrected fg_psnr ssim lpips gradient_preservation background_alpha training_minutes peak_vram_gb checkpoint_path gaussian_ply_path metric_source evaluation_scope status training_views saved_camera_records foreground_training_constraints training_configuration_source'.split()
for e in [f'E{x}' for x in range(1,15)]:
    r,src=metrics.get(e,({},''));i=int(e[1:]);root=HISTORY[e]
    native=[a for a in rows if a['experiment_id']==e and (a['artifact_type']=='Gaussian PLY' or (a['artifact_type']=='Gaussian Model' and a['format']=='ply'))]
    if e=='E13':
        primary=(root/'checkpoints').resolve()
        native=[a for a in native if Path(a['absolute_path']).is_relative_to(primary)]
    final=[a for a in native if 'iteration_30000' in a['absolute_path']]
    staged=[a for a in native if re.search(r'iteration_\d+',a['absolute_path']) and 'failure_exports' not in a['absolute_path']]
    staged.sort(key=lambda a:int(re.search(r'iteration_(\d+)',a['absolute_path']).group(1)),reverse=True)
    selected=(final or staged or ([{}] if i>=12 else native or [{}]))[0]
    checkpoints=[a for a in rows if a['experiment_id']==e and a['artifact_type']=='Checkpoint' and '30000' in a['absolute_path']]
    if e=='E13':checkpoints=[a for a in checkpoints if Path(a['absolute_path']).is_relative_to((root/'checkpoints').resolve())]
    item=dict(experiment_id=e,frontend='VGGT-Omega' if i in [1,2,3,4,7,8] else 'standard VGGT',geometry_rgb='foreground-only historical' if i<=6 else 'full background context',mask_stage='after Point Map',camera_interface='corrected' if i in [8,10,11,12,13,14] else 'historical centered interface',gaussian_method='QGS' if i==13 else '2D-SuGaR' if i==14 else '2DGS' if i in [6,11] else 'GGGS',eo_enabled=i in [2,4,12],iterations=r.get('training_complete_iteration',int(re.search(r'iteration_(\d+)',selected.get('absolute_path','')).group(1)) if re.search(r'iteration_(\d+)',selected.get('absolute_path','')) else ''),num_gaussians=selected.get('vertices_or_gaussians',''),psnr_raw=r.get('full_psnr',''),fg_psnr=r.get('fg_mask_psnr',''),ssim=r.get('ssim',''),lpips=r.get('lpips_alex',''),gradient_preservation=r.get('fg_gradient_retention',''),background_alpha=r.get('background_alpha_mean',''),training_minutes=float(r.get('training_seconds') or r.get('training_wall_seconds') or 0)/60 if r else '',peak_vram_gb=r.get('peak_training_vram_gib') or r.get('peak_allocated_gib',''),checkpoint_path=(checkpoints or [{}])[0].get('absolute_path',''),gaussian_ply_path=selected.get('absolute_path',''),metric_source=src,evaluation_scope='208 view common evaluation implementation; historical camera/background conditions differ' if r else 'not yet uniformly evaluated',status='historical_model_verified' if selected else 'pending')
    if not item['checkpoint_path'] and selected.get('source_checkpoint'):
        item['checkpoint_path']=selected['source_checkpoint']
    # Historical geometry inference already saw foreground-only RGB before
    # its Point Map filter. Do not label that as equivalent to late masking
    # of full-context predictions in E7 onward.
    item['mask_stage']='foreground-only geometry RGB BEFORE inference; Point Map foreground filter (historical)' if i<=6 else 'full background geometry RGB; RAP foreground filter AFTER Point Map'
    cameras=root/'checkpoints/cameras.json'
    if cameras.is_file():
        saved_cameras=json.loads(cameras.read_text());item['saved_camera_records']=len(saved_cameras)
    actual=root/'actual_training_parameters.json'
    if actual.is_file():
        params=json.loads(actual.read_text());item['training_configuration_source']=str(actual.resolve())
        dataset=params.get('dataset',{});dataset=dataset if isinstance(dataset,dict) else {}
        if dataset.get('eval') is False:item['training_views']=item.get('saved_camera_records','')
    if i<=4:
        log=root/'train.log'
        if log.is_file():
            with log.open() as stream:first_line=stream.readline().strip()
            if first_line.startswith('EO enabled:'):
                item['foreground_training_constraints']=first_line
                item['training_configuration_source']=str(log.resolve())+' (actual header; complete old CLI not saved)'
    elif i in [5,6]:item['foreground_training_constraints']='historical native training configuration; additional flags retained in input-config audit'
    else:item['foreground_training_constraints']='no added foreground RGB loss, Mask loss or Gaussian pruning; E14 has native DBSCAN7K'
    if i<=4 and not r:
        legacy=old[i-1]
        item.update(psnr_raw=legacy['PSNR'],psnr_corrected=legacy.get('EO-corrected PSNR',''),fg_psnr=legacy['FG-PSNR'],ssim=legacy['SSIM'],lpips=legacy['LPIPS'],training_minutes=legacy['Training Time (min)'],peak_vram_gb=float(legacy['Peak VRAM (MiB)'])/1024,metric_source=str(PLANT/'comparison.csv'),evaluation_scope='legacy metrics; FG definition differs; pending unified rerender')
    if i>=12:
        item['status']='terminal_native_model_present_pending208_eval' if final else 'partial_native_model_verified' if selected else 'pending'
    gauss.append(item)
gaussian_table=OUT/'comparison/ALL_GAUSSIAN_EXPERIMENTS.csv'
previous=list(csv.DictReader(gaussian_table.open())) if gaussian_table.is_file() else []
verified={r['experiment_id']:r for r in previous if r.get('status')=='model_and208_metrics_verified'}
for i,item in enumerate(gauss):
    old_row=verified.get(item['experiment_id'])
    if old_row and old_row.get('gaussian_ply_path')==item.get('gaussian_ply_path') and old_row.get('num_gaussians')==str(item.get('num_gaussians')):
        gauss[i]=dict(item,**old_row)
known={r['experiment_id'] for r in gauss}
gauss += [r for r in verified.values() if r['experiment_id'] not in known and Path(r.get('gaussian_ply_path','')).is_file()]
writecsv(gaussian_table,gauss,list(dict.fromkeys(gf+[k for r in gauss for k in r])))
mf='mesh_id source_experiment source_gaussian mesh_algorithm depth_source mesh_stage vertices faces components boundary_edges nonmanifold_edges silhouette_iou boundary_f1 boundary_chamfer_px foreground_coverage background_geometry_ratio hole_area surface_roughness extraction_minutes raw_mesh_path post_mesh_path textured_mesh_path metric_source evaluation_scope'.split()
struct={(r['method'],r['stage']):r for r in csv.DictReader((MESH/'comparisons/mesh_structure_comparison.csv').open())}
meshes=[]
for r in csv.DictReader((MESH/'comparisons/mesh_reprojection_comparison.csv').open()):
    if r['reference']!='RAP predicted mask':continue
    s=struct.get((r['method'],r['stage']),{});eid=r['method'].split('_')[0];folder=MESH/'meshes'/r['method']
    raw=folder/'raw.ply';post=folder/'post.ply'
    meshes.append(dict(mesh_id=r['method'],source_experiment='E10' if eid in ['M3','M4'] else 'E11' if eid=='M5' else 'VGGT frontend',source_gaussian=HISTORY['E10' if eid in ['M3','M4'] else 'E11'].as_posix() if eid in ['M3','M4','M5'] else '',mesh_algorithm='TSDF' if 'TSDF' in r['method'] else 'NKSR',depth_source='GGGS median' if eid=='M4' else '2DGS expected depth_ratio=0' if eid=='M5' else 'Point Map / oriented surface observations',mesh_stage=r['stage'],**{k:s.get(k,'') for k in ['vertices','faces','components','boundary_edges','nonmanifold_edges']},silhouette_iou=r['silhouette_iou'],boundary_f1=r['boundary_fscore_2px'],boundary_chamfer_px=r['boundary_chamfer_px'],foreground_coverage=r['foreground_coverage'],background_geometry_ratio=r['background_geometry_fraction'],raw_mesh_path=str(raw.resolve()) if raw.exists() else '',post_mesh_path=str(post.resolve()) if post.exists() else '',metric_source=str(MESH/'comparisons/mesh_reprojection_comparison.csv'),evaluation_scope='historical unified208; hole area/roughness pending; boundary edges not holes'))
writecsv(OUT/'comparison/ALL_MESH_EXPERIMENTS.csv',meshes,mf)
(OUT/'artifact_index/verification.json').write_text(json.dumps(dict(status='active_inventory',files=len(rows),all_paths_exist=all(Path(r['absolute_path']).is_file() for r in rows),historical_gaussians={e:next((a['vertices_or_gaussians'] for a in rows if a['experiment_id']==e and a['artifact_type']=='Gaussian PLY' and '30000' in a['absolute_path']),None) for e in HISTORY}),indent=2))
print('Verified actual artifacts',len(rows),'Gaussian/model files',len(g),'meshes',len(m),flush=True)

