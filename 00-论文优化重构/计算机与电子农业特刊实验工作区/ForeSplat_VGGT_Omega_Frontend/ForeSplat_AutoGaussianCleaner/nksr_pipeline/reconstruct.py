"""Path/I/O adaptation of the validated official ks NKSR extraction script."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import open3d as o3d
import nksr
from nksr.configs import __configs__

from .common import (HISTORY, PacedFile, Stage, guard_memory, hash_file, save_npz,
    stamp, stage_args, write, write_mesh)
import mesh_artifacts


def run(args):
    out = args.output/'mesh'; surface = args.output/'surface'
    if (out/'complete.json').exists():
        return
    info = json.loads((surface/'complete.json').read_text())
    assert info['input_ply'] == stamp(args.input_ply)
    weight = HISTORY/'weights/nksr_ks.pth'
    compatibility = json.loads((HISTORY/'configs/nksr_compatibility_module.json').read_text())
    scale = info['scale']; timings = {}
    torch.set_num_threads(8); torch.manual_seed(0); np.random.seed(0)
    with Stage(out, gpu=True) as stage:
        guard_memory()
        meter = stage.meter
        native_hash = hash_file(compatibility['installed'], meter)
        assert native_hash == compatibility['new_sha256'], 'Existing verified NKSR native module changed'
        weight_hash = hash_file(weight, meter)
        assert weight_hash == json.loads((HISTORY/'configs/nksr_environment_ready.json').read_text())['weight_sha256']
        before = time.perf_counter()
        with PacedFile(surface/'nksr_input.npz', 'rb', meter) as stream:
            with np.load(stream, allow_pickle=False) as archive:
                xyz, normal, sensor = [archive[k].astype(np.float32, copy=False) for k in ['xyz','normal','sensor']]
                rgb = archive['rgb']
        timings['input_read_s'] = time.perf_counter()-before
        assert len(xyz) == info['nksr_points']
        for start in range(0, len(xyz), 262144):
            sl = slice(start, start+262144)
            assert np.isfinite(xyz[sl]).all() and np.isfinite(normal[sl]).all() and np.isfinite(sensor[sl]).all()
            lengths = np.linalg.norm(normal[sl],axis=1)
            assert (lengths>.99).all() and (lengths<1.01).all()
            assert (np.sum(normal[sl]*(sensor[sl]-xyz[sl]),axis=1)>=-1e-6).all()
        config = copy.deepcopy(__configs__['ks']); config['url'] = str(weight)
        original_load = torch.load
        def paced_weight_load(path, *positional, **keywords):
            if isinstance(path, (str, Path)) and Path(path).resolve() == weight.resolve():
                with PacedFile(path, 'rb', meter) as stream:
                    return original_load(stream, *positional, **keywords)
            return original_load(path, *positional, **keywords)
        print('NKSR input',info['plant'],len(xyz),'voxel',scale['nksr_voxel'],flush=True)
        torch.cuda.synchronize(); before = time.perf_counter()
        with patch.object(torch, 'load', paced_weight_load):
            reconstructor = nksr.Reconstructor(torch.device('cuda:0'), config=config)
        with torch.no_grad():
            xyz_gpu = torch.from_numpy(xyz).cuda()
            normal_gpu = torch.from_numpy(normal).cuda()
            sensor_gpu = torch.from_numpy(sensor).cuda()
            field = reconstructor.reconstruct(xyz_gpu, normal=normal_gpu, sensor=sensor_gpu,
                detail_level=None, voxel_size=scale['nksr_voxel'], solver_max_iter=2000,
                solver_tol=1e-5, approx_kernel_grad=False, fused_mode=True)
            assert field is not None
            torch.cuda.synchronize(); timings['model_and_reconstruct_s'] = time.perf_counter()-before
            before = time.perf_counter()
            extracted = field.extract_dual_mesh(mise_iter=1, trim=True, max_points=1000000)
            vertices, faces = extracted.v.cpu().numpy(), extracted.f.cpu().numpy()
            torch.cuda.synchronize(); timings['extract_dual_mesh_s'] = time.perf_counter()-before
        timings['nksr_compute_s'] = timings['model_and_reconstruct_s']+timings['extract_dual_mesh_s']
        print('NKSR extracted',len(vertices),'vertices',len(faces),'faces',flush=True)
        del extracted, field, reconstructor, xyz_gpu, normal_gpu, sensor_gpu, normal, sensor
        torch.cuda.empty_cache(); gc.collect(); guard_memory()
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
        del vertices, faces
        before = time.perf_counter()
        mesh = mesh_artifacts.colour_from_observations(mesh, xyz, rgb)
        # Reuse the exact historical post-processing implementation. Only its
        # artifact writers are paced; no geometry operation is replaced.
        def paced_mesh(path, value, **kwargs): return write_mesh(path, value, meter)
        def paced_archive(path, **arrays): return save_npz(path, meter, **arrays)
        with patch.object(mesh_artifacts.o3d.io, 'write_triangle_mesh', paced_mesh), \
             patch.object(mesh_artifacts.np, 'savez_compressed', paced_archive):
            stats = mesh_artifacts.save_meshes(mesh, out, scale)
        timings['colour_postprocess_export_s'] = time.perf_counter()-before
        guard_memory()
        report = dict(status='complete',plant=info['plant'],input_points=len(xyz),scale=scale,
            input_ply=stamp(args.input_ply),input_npz=stamp(surface/'nksr_input.npz'),
            raw_path=str((out/'raw.ply').resolve()),post_path=str((out/'post.ply').resolve()),
            protocol=info['protocol'],config=config,parameters=dict(detail_level=None,
                voxel_size=scale['nksr_voxel'],solver_max_iter=2000,solver_tol=1e-5,
                approx_kernel_grad=False,fused_mode=True,mise_iter=1,trim=True,max_points=1000000),
            weight_path=str(weight),weight_sha256=weight_hash,native_module_sha256=native_hash,
            nksr_commit=subprocess.check_output(['git','-C',str(HISTORY/'camera_interface/nksr_official'),'rev-parse','HEAD'],text=True).strip(),
            software=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,nksr=nksr.__version__,open3d=o3d.__version__),
            allocator=os.environ.get('PYTORCH_ALLOC_CONF','native default'),timing=timings,
            postprocess='Historical mesh_artifacts.save_meshes: remove components with <50 triangles only',
            **stats)
    report['resources'] = stage.result
    write(out/'mesh_report.json',report)
    write(out/'complete.json',dict(status='complete',plant=info['plant'],report='mesh_report.json'))
    print(info['plant'],'NKSR complete',stats,flush=True)


if __name__ == '__main__':
    run(stage_args(__doc__))
    # Same post-publication exit policy as the validated historical NKSR entry.
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
