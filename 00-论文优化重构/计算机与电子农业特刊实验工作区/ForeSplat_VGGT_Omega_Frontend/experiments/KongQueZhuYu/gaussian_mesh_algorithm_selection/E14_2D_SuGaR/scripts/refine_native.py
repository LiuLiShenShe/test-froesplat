"""Run native 15K mesh-bound SuGaR refinement and native textured OBJ/Gaussian exports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['MPLBACKEND']='Agg'
import sys,json,time,hashlib,inspect
from pathlib import Path
from types import SimpleNamespace
import torch,open3d as o3d,numpy as np
ROOT=Path(__file__).resolve().parents[1];OFF=ROOT/'official';OUT=ROOT/'refinement';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(OFF));sys.path.append(str(OFF/'gaussian_splatting'))
from sugar_trainers.refine import refined_training
from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    if (ROOT/'refinement_complete.json').exists():return
    assert (ROOT/'initial_mesh_complete.json').exists()
    mesh_info=json.loads((ROOT/'initial_mesh_complete.json').read_text())
    if mesh_info.get('status')!='complete' or not mesh_info.get('post_mesh'):
        raise RuntimeError('Native initial Mesh has no nonempty post surface; refinement unavailable; inspect initial_mesh_complete.json')
    torch.manual_seed(0);np.random.seed(0)
    args=SimpleNamespace(scene_path=str(ROOT/'input_colmap'),checkpoint_path=str(ROOT/'training')+os.sep,mesh_path=str(ROOT/'initial_mesh/post.ply'),output_dir=str(OUT),iteration_to_load=30000,normal_consistency_factor=.1,gaussians_per_triangle=1,n_vertices_in_fg=1000000,refinement_iterations=15000,bboxmin=None,bboxmax=None,export_ply=True,eval=False,gpu=0,white_background=False)
    config_file=ROOT/'actual_refinement_parameters.json'
    if config_file.exists():assert json.loads(config_file.read_text())==vars(args), 'Refinement recovery parameters differ'
    write(config_file,vars(args))
    started=time.time();torch.cuda.reset_peak_memory_stats()
    retained=OUT/'sugarfine_post_normalconsistency01_gaussperface1/15000.pt'
    previous_wall=0.;recovered=False
    if retained.is_file():
        # This exact local terminal model predates the texture-loader failure.
        native=torch.load(retained,map_location='cpu',weights_only=False)
        assert native['iteration']==15000 and native['state_dict']['_surface_mesh_faces'].numel()>0
        del native
        path=str(retained);recovered=True
        completed_attempts=[]
        for line in (ROOT/'logs/actual_commands.jsonl').read_text().splitlines():
            event=json.loads(line)
            if event.get('stage')=='04_SuGaR15K_refinement_and_texture' and event.get('started_unix',float('inf'))<=retained.stat().st_mtime<=event.get('ended_unix',0):
                completed_attempts.append(event)
        assert completed_attempts, 'Missing measured original training execution'
        previous_wall=completed_attempts[-1]['ended_unix']-completed_attempts[-1]['started_unix']
        write(ROOT/'refinement_training_complete.json',dict(status='complete',iterations=15000,native_model=str(retained.resolve()),sha256=hashlib.sha256(retained.read_bytes()).hexdigest(),native_checkpoint_iteration_verified=True,original_attempt_seconds=previous_wall,recovery='Reuse exact completed native15K checkpoint; no additional optimization or retraining'))
        print('Reusing verified completed native15K SuGaR model; extracting texture only',flush=True)
    else:
        path=refined_training(args)
    # Native final PT contains mesh vertices, surfel binding and optimizer state; preserve it.
    assert Path(path).is_file()
    extraction=SimpleNamespace(scene_path=args.scene_path,iteration_to_load=30000,checkpoint_path=args.checkpoint_path,refined_model_path=path,mesh_output_dir=str(OUT),n_gaussians_per_surface_triangle=1,square_size=8,eval=False,gpu=0,postprocess_mesh=False,postprocess_density_threshold=.1,postprocess_iterations=5,mesh_path=args.mesh_path)
    obj=extract_mesh_and_texture_from_refined_sugar(extraction)
    assert Path(obj).is_file()
    # Preserve geometry-only export from the actual native textured OBJ, with no edit.
    mesh=o3d.io.read_triangle_mesh(str(obj));assert len(mesh.triangles)>0
    o3d.io.write_triangle_mesh(str(OUT/'refined_raw.ply'),mesh,write_ascii=False)
    import sys
    sys.path.append(str(ROOT.parent.parent/'geometry_mesh_final_selection/scripts'))
    from mesh_artifacts import save_meshes
    scale=json.loads((ROOT.parent.parent/'geometry_mesh_final_selection/configs/standard_spatial_scale.json').read_text())
    stats=save_meshes(mesh,OUT/'mesh_post',scale)
    ply=list((OUT/'refined_ply').glob('*.ply'));assert len(ply)==1
    deps=[str(x.resolve()) for x in OUT.glob('*.mtl')]+[str(x.resolve()) for x in OUT.glob('*.png')]
    torch.cuda.synchronize()
    write(ROOT/'refinement_complete.json',dict(status='complete',iterations=15000,native_model=str(Path(path).resolve()),gaussian_ply=str(ply[0].resolve()),textured_obj=str(Path(obj).resolve()),texture_dependencies=deps,refined_raw_ply=str(OUT/'refined_raw.ply'),refined_post_ply=str(OUT/'mesh_post/post.ply'),seed=0,views=208,extra_optimization='Native mesh normal consistency0.1,1 Gaussian per face; native vertex/appearance refinement on same208 views',elapsed_seconds=previous_wall+time.time()-started,recovery_export_seconds=time.time()-started,reused_native15K_checkpoint=recovered,peak_vram_scope='Recovery texture/Mesh export only; original15K training peak not independently recorded after loader failure' if recovered else 'Native15K training and texture/Mesh export',peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,**stats))
if __name__=='__main__':main()

