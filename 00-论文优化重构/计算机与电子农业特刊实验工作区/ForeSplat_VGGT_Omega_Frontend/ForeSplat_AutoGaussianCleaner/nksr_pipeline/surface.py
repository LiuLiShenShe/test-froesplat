"""Scene adaptation of historical render_surface + prepare_rendered_inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import gc
import json
from pathlib import Path
import numpy as np
import torch

from .common import (ROOT, Stage, StreamingObservations, PROTOCOL, camera_record,
    get_camera, guard_memory, load_gaussians, load_image, load_npy, make_scene,
    normals_from_grid, save_image, save_npz, stamp, stage_args, unproject, write)


def reference_scale(spec, scene, meter):
    """Same robust extent from existing original foreground depth observations."""
    meta = json.loads(Path(spec['geometry_cameras']).read_text())
    frames = {Path(row['image_name']).stem:row for row in meta['frames']}
    alignment = np.asarray(meta['scene_alignment_matrix'])
    saved = json.loads((Path(spec['dataset'])/'metadata/reconstruction_metadata.json').read_text())
    assert np.allclose(alignment, saved['scene_alignment_matrix'], atol=1e-9)
    threshold, minimum = saved['confidence_threshold_value'], saved['min_conf']
    samples, counts, maximum_pose_error = [], [], 0.
    for index, view in enumerate(scene.views):
        frame = frames[view['name']]
        raw = np.eye(4); raw[:3,:] = frame['extrinsic_model_w2c']
        aligned = raw @ np.linalg.inv(alignment)
        error = float(np.max(np.abs(aligned-view['E'])))
        maximum_pose_error = max(maximum_pose_error, error)
        if not np.allclose(aligned, view['E'], atol=2e-4, rtol=2e-4):
            raise ValueError(f'Existing geometry / GGGS gauge mismatch: {view["name"]}')
        base = Path(spec['geometry_dir']); stem = frame.get('stem', view['name'])
        depth = load_npy(base/f'depth_{stem}.npy', meter).squeeze()
        confidence = load_npy(base/f'confidence_{stem}.npy', meter).squeeze()
        mask = load_npy(base/f'model_mask_{stem}.npy', meter).astype(bool)
        intrinsic = np.asarray(frame['intrinsic_model'])
        xyz, _ = unproject(depth, dict(Fx=intrinsic[0,0], Fy=intrinsic[1,1], Cx=intrinsic[0,2],
            Cy=intrinsic[1,2], world_view_transform=raw.T))
        valid = mask & np.isfinite(depth) & (depth>0) & np.isfinite(confidence) & (confidence>=threshold) & (confidence>minimum) & np.isfinite(xyz).all(-1)
        points = xyz[valid]; counts.append(len(points))
        sample = points[::max(1, len(points)//4000)]
        samples.append(sample@alignment[:3,:3].T+alignment[:3,3])
        if index%25 == 0:
            print('scale', spec['plant'], index+1, '/', len(scene.views), flush=True)
        guard_memory(16)
    points = np.concatenate(samples)
    low, high = np.quantile(points, [.005, .995], axis=0)
    diagonal = float(np.linalg.norm(high-low))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError('Invalid robust scene extent')
    return dict(diagonal=diagonal, robust_bbox_min=low.tolist(), robust_bbox_max=high.tolist(),
        extent_quantiles=[.005,.995], origin=((low+high)/2).tolist(),
        sampling_voxel=diagonal/1024, nksr_voxel=diagonal/512,
        reference='Existing original Omega foreground depth, saved global confidence threshold, deterministic ~4000 samples/frame; scale only, no clipping',
        original_foreground_points=sum(counts), original_confidence_threshold=threshold,
        raw_to_export_alignment=alignment.tolist(), maximum_verified_pose_abs_error=maximum_pose_error,
        rendered_surface_alignment='Already exported GGGS world frame; no transform applied to rendered surface points',
        scale_fit_to_scan=False)


def run(args):
    out = args.output/'surface'
    if (out/'complete.json').exists():
        return
    spec = json.loads(args.scene.read_text())
    torch.set_num_threads(8)
    with Stage(out, gpu=True) as stage:
        meter = stage.meter
        scene = make_scene(spec)
        scale = reference_scale(spec, scene, meter)
        write(out/'spatial_scale.json', scale)
        model, count = load_gaussians(scene, args.input_ply, meter)
        sampler = StreamingObservations(scale)
        records, cameras = [], []
        representatives = scene.evidence_indices()
        previews = out/'previews'; previews.mkdir(exist_ok=True)
        for index, view in enumerate(scene.views):
            camera = get_camera(scene, index)
            record = camera_record(camera); cameras.append(record)
            mask = load_image(view['mask'], (view['width'], view['height']), meter, mask=True)
            with torch.no_grad():
                rendered = scene.render(camera, model)
            depth = rendered['median_depth'].detach().cpu().numpy().squeeze()
            alpha = rendered['mask'].detach().cpu().numpy().squeeze()
            xyz, sensor = unproject(depth, record)
            valid = mask & (alpha>=.5) & np.isfinite(depth) & (depth>0) & np.isfinite(xyz).all(-1)
            normal, good = normals_from_grid(xyz, depth, valid, sensor)
            raw_rgb = rendered['render'].detach().cpu().numpy().transpose(1,2,0)
            rgb = np.rint(np.clip(raw_rgb,0,1)*255).astype(np.uint8)
            selected = valid & good
            data = dict(xyz=xyz[selected], normal=normal[selected],
                sensor=np.broadcast_to(sensor,(int(selected.sum()),3)).copy(), rgb=rgb[selected])
            added = sampler.add(data)
            records.append(dict(index=index,name=view['name'],valid_alpha_depth_points=int(valid.sum()),
                valid_normal_points=int(selected.sum()),new_unique_observations=added,cumulative_points=sampler.count))
            if index in representatives:
                import PIL.Image
                def preview(name, array):
                    image = PIL.Image.fromarray(np.uint8(np.clip(array,0,1)*255)); image.thumbnail((540,960))
                    save_image(previews/f'{view["name"]}_{name}.png',np.asarray(image)/255.,meter)
                visible_depth = np.where(selected, np.clip(depth/scale['diagonal'],0,1), 0)
                preview('depth', visible_depth)
                preview('normal', np.where(selected[...,None], (normal+1)/2, 0))
                preview('alpha', alpha)
            print('surface',spec['plant'],index+1,'/',len(scene.views),'new',added,'total',sampler.count,flush=True)
            del rendered, depth, alpha, xyz, normal, raw_rgb, rgb, data, valid, selected, good, mask, camera
            guard_memory(16)
            if index%20 == 0: gc.collect()
        del model, scene
        torch.cuda.empty_cache(); gc.collect()
        arrays = sampler.finish()
        save_npz(out/'nksr_input.npz', meter, **arrays)
        write(out/'camera_runtime.json',dict(records=cameras,representative_indices=representatives))
        write(out/'complete.json',dict(status='complete',plant=spec['plant'],input_ply=stamp(args.input_ply),
            input_gaussians=count,nksr_points=len(arrays['xyz']),frames=records,used_camera_count=len(cameras),
            scale=scale,protocol=PROTOCOL,render_max_side=1920,training_resolution_divisor=2,
            depth='Native GGGS median_depth; foreground mask and alpha>=0.5',
            normal_strategy=PROTOCOL['normal_method'],
            coordinate_frame='Current exported corrected GGGS world gauge; original alignment verified, never applied twice',
            sampler='Exact historical voxel+dominant-normal-direction first observation selection; incremental sorted-key membership; no averaging',
            npz_fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()}))


if __name__ == '__main__':
    run(stage_args(__doc__))
