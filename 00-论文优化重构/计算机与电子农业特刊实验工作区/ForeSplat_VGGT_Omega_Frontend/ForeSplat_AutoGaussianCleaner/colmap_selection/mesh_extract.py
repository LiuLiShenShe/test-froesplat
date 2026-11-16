"""Existing KS NKSR adapter, Open3D TSDF and Poisson on one frozen surface."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='8'
import argparse
import hashlib
import io
import json
import resource
import sys
import time
from pathlib import Path
from unittest.mock import patch
from runtime import OUT, FRONT, CLEANER, SOURCES, Meter, write
import numpy as np
import open3d as o3d


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('method',choices=['NKSR','TSDF','Poisson'])
    args=parser.parse_args();surface=OUT/'mesh'/f'{args.model}_surface'
    base_model=args.model.removesuffix('_Cleaned')
    info=json.loads((surface/'complete.json').read_text());scale=info['scale']
    output=OUT/'mesh'/args.method/args.model;output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():return
    meter=Meter();wall=time.perf_counter();cpu=time.process_time()
    input_digest=hashlib.sha256()
    with (surface/'nksr_input.npz').open('rb') as stream:
        while True:
            block=stream.read(256*1024)
            if not block:break
            meter.record(len(block),'read');input_digest.update(block)
    input_hash=input_digest.hexdigest()
    sys.path.insert(0,str(CLEANER));sys.path.insert(0,str(FRONT/'experiments/KongQueZhuYu/geometry_mesh_final_selection/scripts'))
    from nksr_pipeline.common import stamp,write_mesh
    import mesh_artifacts
    if args.method=='NKSR':
        import nksr_pipeline.reconstruct as existing
        import nksr
        import torch
        from types import SimpleNamespace
        from surface_protocol import PROTOCOL
        ply=OUT/'cleaning'/base_model/'cleaned_gaussians.ply' if args.model.endswith('_Cleaned') else Path(json.loads((OUT/'gaussian'/base_model/'complete.json').read_text())['native_ply'])
        source_info=dict(info,plant='KongQueZhuYu',input_ply=stamp(ply),nksr_points=info['points'],protocol=dict(PROTOCOL,version=2,
            confidence='New true COLMAP run: RAP masks and alpha>=0.5; no frontend predicted geometry',
            coordinate_policy='Unchanged historical true COLMAP world gauge for all five trainings; no VGGT, rigid alignment, scan fitting or Sim(3) fitting',
            extent='Current Gaussian native depth observations, 0.5/99.5 percentiles; no historical scene alignment'))
        write(surface/'complete.json',source_info)
        link=output/'surface'
        if not link.exists():link.symlink_to(surface,target_is_directory=True)
        # Official whole-scene KS inference exceeded48GB on this run. Use its
        # own split_into_chunks/reconstruct_by_chunk path at the SAME effective
        # voxel size. Normalize exactly as native reconstruct(voxel_size=...)
        # does first, because its convenience chunk_size branch drops voxel_size.
        original_reconstruct=nksr.Reconstructor.reconstruct
        chunk_record={}
        def chunked(reconstructor,xyz,normal=None,sensor=None,**kw):
            if kw.get('_inside_chunk',False):
                kw.pop('_inside_chunk');return original_reconstruct(reconstructor,xyz,normal=normal,sensor=sensor,**kw)
            world_scale=kw.pop('voxel_size')/reconstructor.hparams.voxel_size
            kw.pop('detail_level',None)
            normalized=xyz/world_scale;origins=sensor/world_scale if sensor is not None else None
            transforms,parts,features=nksr.split_into_chunks(normalized,12.8,.05,normal=normal,sensor=origins)
            chunk_record.update(normalized_chunk_size=12.8,world_chunk_size=12.8*world_scale,overlap_ratio=.05,
                chunks=len(parts),points_per_chunk=[len(x) for x in parts],world_scale=world_scale,
                effective_voxel=scale['nksr_voxel'],input_points=len(xyz),source='Official nksr.split_into_chunks and Reconstructor.reconstruct_by_chunk',
                reason='Whole-scene native UNet inference OOM; no point removal or resolution reduction',
                limit='Official overlapping local field solves can differ from one global solve; reported as chunked KS')
            write(output/'chunking.json',chunk_record)
            reconstructor.set_chunk_tmp_device('cpu')
            field=reconstructor.reconstruct_by_chunk(transforms,parts,normal_list=features['normal'],sensor_list=features['sensor'],
                detail_level=None,_inside_chunk=True,**kw)
            def extract(**options):
                result=field.extract_dual_mesh(**options);result.v*=world_scale;return result
            return SimpleNamespace(extract_dual_mesh=extract)
        with patch.object(nksr.Reconstructor,'reconstruct',chunked):
            existing.run(argparse.Namespace(scene=OUT/'common_input/colmap_source.json',input_ply=ply,output=output))
        report=json.loads((output/'mesh/mesh_report.json').read_text())
        for name in ['raw.ply','post.ply']:
            if not (output/name).exists():(output/name).symlink_to(output/'mesh'/name)
        write(output/'complete.json',dict(status='complete',method=args.method,model=args.model,
            source=str(surface),oriented_input_sha256=input_hash,chunking=chunk_record,wall_s=time.perf_counter()-wall,report=report))
        return
    with np.load(io.BytesIO(meter.read(surface/'nksr_input.npz')),allow_pickle=False) as data:
        xyz,normal,rgb=[data[k] for k in ['xyz','normal','rgb']]
    if args.method=='Poisson':
        pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        pcd.normals=o3d.utility.Vector3dVector(normal);pcd.colors=o3d.utility.Vector3dVector(rgb.astype(float)/255)
        mesh,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,depth=10,n_threads=8)
        params=dict(depth=10,n_threads=8,density_trimming=False,input_identical_to_nksr=str(surface/'nksr_input.npz'))
    else:
        import torch
        from types import SimpleNamespace
        sys.path.insert(0,str(SOURCES['2DGS']))
        from utils import mesh_utils as bounded
        train_names={Path(n).stem for n in (OUT/'common_input/train_views.txt').read_text().splitlines()}
        frames=sorted(p for p in (surface/'frames').glob('*.npz') if p.stem in train_names);assert len(frames)==183
        cameras=[];maximum=0.
        for frame in frames:
            with np.load(io.BytesIO(meter.read(frame)),allow_pickle=False) as data:
                z,K,c2w=[data[k] for k in ['depth','K','c2w']]
            h,w=z.shape;maximum=max(maximum,float(z.max()))
            c=o3d.camera.PinholeCameraParameters()
            c.intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2]-.5,K[1,2]-.5)
            c.extrinsic=np.linalg.inv(c2w);cameras.append(c)
        class FrameCache:
            index=None;value=None
            def get(self,index):
                if self.index!=index:
                    with np.load(io.BytesIO(meter.read(frames[index]))) as data:self.value={k:data[k] for k in ['depth','rgb']}
                    self.index=index
                return self.value
        cache=FrameCache()
        class Maps:
            def __init__(self,key):self.key=key
            def __getitem__(self,index):
                value=cache.get(index)[self.key]
                return torch.from_numpy(value).permute(2,0,1).float()/255 if self.key=='rgb' else torch.from_numpy(value[None])
        adapter=SimpleNamespace(viewpoint_stack=[SimpleNamespace(gt_alpha_mask=None) for _ in frames],rgbmaps=Maps('rgb'),depthmaps=Maps('depth'))
        # Call the existing official bounded TSDF method unchanged. The camera
        # adapter supplies the exact same half-pixel intrinsics as its own
        # to_cam_open3d(), while frame arrays are lazily loaded through the meter.
        with patch.object(bounded,'to_cam_open3d',lambda unused:cameras):
            mesh=bounded.GaussianExtractor.extract_mesh_bounded(adapter,voxel_size=scale['nksr_voxel'],
                sdf_trunc=scale['nksr_voxel']*5,depth_trunc=max(maximum*1.01,1e-3),mask_backgrond=False)
        params=dict(voxel_length=scale['nksr_voxel'],sdf_trunc=scale['nksr_voxel']*5,frames=183,depth_scale=1.,held_out_test_views=27,
            depth='Camera-Z; cached RAP+alpha validity, no additional mask inside fusion',
            implementation='Official 2DGS utils.mesh_utils.GaussianExtractor.extract_mesh_bounded, unmodified; camera and lazy frame adapters only',
            native_2dgs_bounded_equivalent=base_model=='2DGS',count_as_independent_native_method=False)
    mesh=mesh_artifacts.colour_from_observations(mesh,xyz,rgb)
    original_npz=np.savez_compressed
    def archive(path,**arrays):
        buffer=io.BytesIO();original_npz(buffer,**arrays);meter.save(path,buffer.getbuffer())
    def save(path,mesh,**kwargs):return write_mesh(path,mesh,meter)
    with patch.object(mesh_artifacts.o3d.io,'write_triangle_mesh',save),patch.object(mesh_artifacts.np,'savez_compressed',archive):
        stats=mesh_artifacts.save_meshes(mesh,output,scale)
    write(output/'complete.json',dict(status='complete',method=args.method,model=args.model,source=str(surface),parameters=params,
        oriented_input_sha256=input_hash,
        wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,io=meter.stats(),**stats))


if __name__=='__main__':
    main()
    # Match the existing NKSR publication policy after every artifact is closed.
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
