"""Independent AMB3R OFF/ON adapter; upstream network and weights stay untouched."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'third_party' / 'amb3r'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / 'thirdparty'))
from frontends.common import EXPERIMENT, as_numpy, export_bundle, json_write, load_inputs, sha256, write_ply

PREDICTION_KEYS = (
    'world_points', 'world_points_conf', 'depth', 'depth_conf', 'depth_metric',
    'pose', 'intrinsic', 'extrinsic', 'pts3d_by_unprojection', 'pose_enc',
    'median_metric_z', 'median_metric_z_log',
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def input_digest(inputs):
    digest = hashlib.sha256()
    for key in ('rgb', 'masks', 'valid_pixels'):
        digest.update(np.ascontiguousarray(inputs[key]).tobytes())
    digest.update(json.dumps(inputs['transforms'], sort_keys=True).encode())
    return digest.hexdigest()


def dependency_versions():
    result = {}
    for name in ('torch', 'torchvision', 'numpy', 'flash-attn', 'pytorch3d',
                 'torch-scatter', 'spconv-cu120', 'cumm-cu120', 'utils3d'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def prepare_model(checkpoint=None):
    """Use exactly AMB3R.load_weights with strict state-dict checking."""
    from amb3r.model import AMB3R
    checkpoint = Path(checkpoint or UPSTREAM / 'checkpoints' / 'amb3r.pt')
    model = AMB3R(device='cuda', metric_scale=True, voxel_resolutions=[0.01])
    model.load_weights(str(checkpoint), data_type='bf16', strict=True)
    model.eval().to('cuda')
    return model


def per_frame_predictions(result):
    """Copy all official geometry outputs; hidden decoder tensors are not predictions."""
    available = {key: as_numpy(result[key][0]) for key in PREDICTION_KEYS if key in result}
    count = available['world_points'].shape[0]
    predictions = []
    # This reproduces frontend.py's torch.median (lower median for even lengths).
    if 'depth_metric' in available:
        d = available['depth'].reshape(count, -1)
        md = available['median_metric_z'].reshape(count)
        lower = (d.shape[1] - 1) // 2
        per_frame_scale = md / (np.partition(d, lower, axis=1)[:, lower] + 1e-8)
        scene_scale = float(np.partition(per_frame_scale, (count - 1) // 2)[(count - 1) // 2])
    else:
        scene_scale = None
    for i in range(count):
        pred = {key: array[i] for key, array in available.items()}
        if scene_scale is not None:
            pred['scene_metric_scale_derived'] = np.asarray(scene_scale, np.float32)
            pred['frame_metric_scale_derived'] = np.asarray(per_frame_scale[i], np.float32)
            pred['world_points_metric_derived'] = pred['world_points'] * scene_scale
            pred['pts3d_by_unprojection_metric_derived'] = pred['pts3d_by_unprojection'] * scene_scale
            pose_metric = pred['pose'].copy()
            pose_metric[:3, 3] *= scene_scale
            pred['pose_metric_derived'] = pose_metric
        predictions.append(pred)
    return predictions, scene_scale


def camera_geometry_checks(predictions, inputs):
    """Check actual projection, SE(3) inversion and first-reference convention."""
    h, w = inputs['masks'].shape[-2:]
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack([xx, yy], -1)
    rows = []
    for i, pred in enumerate(predictions):
        c2w = pred['pose'].astype(np.float64)
        w2c = pred['extrinsic'].astype(np.float64)
        k = pred['intrinsic'].astype(np.float64)
        mask = inputs['masks'][i] & inputs['valid_pixels'][i]
        row = dict(name=inputs['rows'][i]['name'], status='direct_pose_head_prediction',
                   pnp_status='not_applicable', pnp_valid_points=None,
                   inverse_se3_max_error=float(np.abs(np.linalg.inv(c2w)[:3] - w2c).max()),
                   rotation_determinant=float(np.linalg.det(c2w[:3, :3])),
                   rotation_orthogonality_max_error=float(np.abs(c2w[:3, :3].T @ c2w[:3, :3] - np.eye(3)).max()))
        for key, label in (('world_points', 'direct_world_points'),
                           ('pts3d_by_unprojection', 'depth_unprojection')):
            pts = pred[key].astype(np.float64)
            camera = pts @ w2c[:, :3].T + w2c[:, 3]
            projected = camera @ k.T
            valid = mask & np.isfinite(projected).all(-1) & (camera[..., 2] > 1e-8)
            uv = projected[..., :2] / np.maximum(projected[..., 2:], 1e-8)
            errors = np.linalg.norm(uv - pixels, axis=-1)[valid]
            row[label + '_positive_depth_foreground_fraction'] = float(valid.sum() / max(mask.sum(), 1))
            row[label + '_reprojection_median_px'] = float(np.median(errors)) if len(errors) else None
            row[label + '_reprojection_p95_px'] = float(np.quantile(errors, .95)) if len(errors) else None
        rows.append(row)
    first = predictions[0]['pose']
    summary = dict(first_camera_c2w_identity_max_error=float(np.abs(first - np.eye(4)).max()),
                   pose_convention='official frontend: extrinsic world-to-camera; pose its inverse camera-to-world',
                   reference='first input camera is pose/depth reference; independent direct point head agreement is evaluated separately, not guaranteed',
                   unprojection_reprojection_check='algebraic convention check, not independent pose ground truth',
                   direct_pointmap_reprojection_check='pointmap vs pose/depth-head consistency diagnostic, not feature-track error')
    return rows, summary


def write_report(out, metadata, bundle=None, error=None):
    if error:
        report = '# AMB3R Base run failed\n\n' + str(error) + '\n\n```text\n' + metadata['traceback'] + '\n```\n'
    else:
        report = (f'# {metadata["method"]}\n\n'
                  f'Official `AMB3R.forward(frames, iters={metadata["backend_iterations"]})`, '
                  f'{metadata["frame_count"]} frames, scope `{metadata["scope"]}`. '
                  f'Input size: {metadata["input_size_hw"]}.\n\n'
                  f'Inference: {metadata["inference_seconds"]:.3f} s; '
                  f'peak allocated VRAM: {metadata["peak_allocated_gib"]:.3f} GiB; '
                  f'peak reserved VRAM: {metadata["peak_reserved_gib"]:.3f} GiB. Only physical GPU1 used.\n\n'
                  'Timing includes the official forward, geometry/camera heads, configured backend and CUDA synchronization; '
                  'it excludes model loading, input preparation, CPU diagnostics and output serialization.\n\n'
                  'Raw `world_points`, `pts3d_by_unprojection` and `pose` retain the official normalized scene scale. '
                  '`depth_metric` is the official predicted metric depth. Direct point-head metric derivatives '
                  'are conditional depth-scale transfers with gauge_validated=false, not guaranteed metres. '
                  'Depth-unprojected metric points and scaled camera translations use the same official scale '
                  'and are algebraically consistent, without establishing measured accuracy. '
                  'no scan registration or scale correction is applied. Native local pointmap head is absent.\n\n'
                  'Masks are external filters only; the original network does not consume masks. '
                  'All input frames retain source order; geometry uses aspect-preserving resize and padding, '
                  'with nearest-neighbor masks and saved pixel transforms.\n\n'
                  f'Checkpoint SHA256: `{metadata["weight_sha256"]}`. '
                  f'Official commit: `{metadata["git_commit"]}`.\n\n'
                  f'Foreground mask+confidence point count: {bundle["cloud_counts"]["mask_confidence"]}.\n\n'
                  '[Official source](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/model.py), '
                  '[paper](https://arxiv.org/abs/2511.20343).\n')
    (Path(out) / 'RUN_REPORT.md').write_text(report)


def run_stage(model, inputs, iters, out, common_meta, seed):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty, preserving existing run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(common_meta, method='AMB3R_Base_' + ('ON' if iters else 'OFF'), backend_iterations=iters)
    metadata['geometry_branch'] = 'official direct world_points head; primary Bundle pointmaps and foreground clouds'
    metadata['secondary_geometry_branch'] = 'official pts3d_by_unprojection; preserved separately, including explicitly derived predicted-scale export'
    seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    frames = {'images': inputs['images'][None].to('cuda')}
    started = time.perf_counter()
    try:
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            stages = model.forward(frames, iters=iters)
        torch.cuda.synchronize()
        metadata.update(inference_seconds=time.perf_counter() - started,
                        inference_time_scope='original AMB3R.forward including camera/intrinsic/depth/point heads and configured backend, plus CUDA synchronization; excludes model load, input preparation/H2D, CPU diagnostics and output serialization',
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                        official_stage_count=len(stages),
                        official_output_keys=list(stages[-1]))
        predictions, scene_scale = per_frame_predictions(stages[-1])
        metadata['scene_metric_scale_predicted'] = scene_scale
        metadata['frame_metric_scales_predicted'] = [float(p['frame_metric_scale_derived']) for p in predictions]
        metadata['metric_scale_notes'] = ('Official scale = median_t(exp(median_metric_z_log_t) / '
                                          '(median_pixels(depth_t) + 1e-8)); lower medians match torch.median. '
                                          'Metric derivatives scale original geometry/translation; masks do not influence scale.')
        metadata['world_points_metric_derived_gauge_validated'] = False
        metadata['global_mask_confidence_metric_derived_gauge_validated'] = False
        metadata['direct_pointmap_metric_transfer'] = ('Conditional transfer of the depth-head scene scale to the direct point head; '
                                                       'shared normalized gauge has not been established; no guaranteed metric coordinates.')
        metadata['depth_unprojection_metric_derivation'] = ('Official depth scale applied to normalized depth unprojection and camera translations '
                                                            'by the same scalar; algebraic consistency only, not measured metric accuracy.')
        if iters:
            # Keep the stage-0 prediction for an OFF/ON reproducibility check.
            before, _ = per_frame_predictions(stages[0])
            fingerprint = hashlib.sha256()
            for p in before:
                fingerprint.update(np.ascontiguousarray(p['world_points']).tobytes())
            metadata['stage0_world_points_sha256'] = fingerprint.hexdigest()
            del before
        else:
            fingerprint = hashlib.sha256()
            for p in predictions:
                fingerprint.update(np.ascontiguousarray(p['world_points']).tobytes())
            metadata['stage0_world_points_sha256'] = fingerprint.hexdigest()
        del stages, frames
        gc.collect()
        torch.cuda.empty_cache()
        status, checks = camera_geometry_checks(predictions, inputs)
        metadata['geometry_checks'] = checks
        json_write(out / 'run_parameters.json', metadata)
        json_write(out / 'status.json', dict(status='exporting', **metadata))
        bundle = export_bundle(out, inputs, predictions,
                               c2w=np.stack([p['pose'] for p in predictions]),
                               intrinsics=np.stack([p['intrinsic'] for p in predictions]),
                               metadata=metadata, pose_status=status)
        json_write(out / 'run_parameters.json', metadata)
        json_write(out / 'camera_geometry_checks.json', status)
        metric_pose = np.stack([p['pose_metric_derived'] for p in predictions])
        np.savez_compressed(out / 'cameras' / 'cameras_metric_derived.npz',
                            c2w=metric_pose, w2c=np.linalg.inv(metric_pose),
                            intrinsics_model=np.stack([p['intrinsic'] for p in predictions]),
                            scene_metric_scale=np.asarray(scene_scale),
                            names=np.array([r['name'] for r in inputs['rows']]))
        metric_points, metric_colors, unproject_metric_points, unproject_metric_colors = [], [], [], []
        for i, pred in enumerate(predictions):
            xyz = pred['world_points_metric_derived']
            conf = pred['world_points_conf'].reshape(inputs['masks'][i].shape)
            valid = (inputs['masks'][i] & inputs['valid_pixels'][i] & np.isfinite(xyz).all(-1)
                     & np.isfinite(conf) & (conf > 1))
            metric_points.append(xyz[valid])
            metric_colors.append(inputs['rgb'][i][valid])
            unproject = pred['pts3d_by_unprojection_metric_derived']
            unproject_valid = (inputs['masks'][i] & inputs['valid_pixels'][i] & np.isfinite(unproject).all(-1)
                               & np.isfinite(conf) & (conf > 1))
            unproject_metric_points.append(unproject[unproject_valid])
            unproject_metric_colors.append(inputs['rgb'][i][unproject_valid])
        write_ply(out / 'pointcloud' / 'global_mask_confidence_metric_derived.ply',
                  np.concatenate(metric_points), np.concatenate(metric_colors))
        write_ply(out / 'pointcloud' / 'global_depth_unprojection_mask_confidence_metric_predicted.ply',
                  np.concatenate(unproject_metric_points), np.concatenate(unproject_metric_colors))
        metadata['metric_geometry_exports'] = dict(
            direct_pointhead=dict(pointmap_field='world_points_metric_derived',
                                  ply='pointcloud/global_mask_confidence_metric_derived.ply',
                                  gauge_validated=False, units='conditional depth-scale transfer; not guaranteed metres'),
            depth_unprojection=dict(pointmap_field='pts3d_by_unprojection_metric_derived',
                                    ply='pointcloud/global_depth_unprojection_mask_confidence_metric_predicted.ply',
                                    camera_file='cameras/cameras_metric_derived.npz',
                                    depth_pose_scaling_consistent_by_construction=True, measured_metric_accuracy_validated=False,
                                    confidence_filter='same primary world_points_conf > 1 and unchanged foreground mask',
                                    point_count=sum(len(x) for x in unproject_metric_points)))
        bundle['metadata'].update(metadata)
        json_write(out / 'bundle.json', bundle)
        json_write(out / 'run_parameters.json', metadata)
        del metric_points, metric_colors, unproject_metric_points, unproject_metric_colors
        write_report(out, metadata, bundle)
        json_write(out / 'status.json', dict(status='complete', **metadata))
        del predictions
        gc.collect()
        print(json.dumps(dict(output=str(out), **metadata), default=str), flush=True)
        return metadata
    except Exception as exc:
        metadata.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                        elapsed_seconds=time.perf_counter() - started,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        json_write(out / 'FAILED.json', metadata)
        json_write(out / 'status.json', dict(status='failed', **metadata))
        write_report(out, metadata, error=exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=EXPERIMENT / 'manifest.csv')
    parser.add_argument('--checkpoint', type=Path, default=UPSTREAM / 'checkpoints' / 'amb3r.pt')
    parser.add_argument('--mode', choices=('off', 'on', 'both'), default='both')
    parser.add_argument('--representative', type=int, help='Explicit fixed evenly-spaced subset; never labels it full sequence')
    parser.add_argument('--indices', help='Comma-separated usable-frame indices, in source order')
    parser.add_argument('--long-edge', type=int, default=518)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    with open(args.manifest) as f:
        all_rows = [r for r in csv.DictReader(f) if r['status'] == 'ok']
    if args.indices and args.representative:
        parser.error('--indices and --representative are mutually exclusive')
    indices = [int(x) for x in args.indices.split(',')] if args.indices else None
    if args.representative:
        indices = np.linspace(0, len(all_rows) - 1, args.representative).round().astype(int).tolist()
    if indices is not None and (indices != sorted(set(indices)) or any(i < 0 or i >= len(all_rows) for i in indices)):
        parser.error('Indices must be unique, increasing and within usable rows')
    inputs = load_inputs(args.manifest, long_edge=args.long_edge, alignment=14, indices=indices)
    torch.set_num_threads(8)
    metadata = dict(scope='full_sequence' if indices is None else 'fixed_representative_subset',
                    full_sequence=indices is None, input_views=len(inputs['rows']),
                    frame_count=len(inputs['rows']), total_usable_frames=len(all_rows),
                    source_indices=[int(r['source_index']) for r in inputs['rows']],
                    usable_frame_indices=[int(r['frame_index']) for r in inputs['rows']],
                    seed=args.seed, gpu_physical_index=1, cuda_visible_devices='1',
                    precision='bf16 autocast; official heads fp32', mask_input_to_network=False,
                    long_edge=args.long_edge, size_alignment=14, input_size_hw=list(inputs['images'].shape[-2:]),
                    shared_input_sha256=input_digest(inputs), manifest_sha256=sha256(args.manifest),
                    weight_path=str(args.checkpoint), weight_sha256=sha256(args.checkpoint),
                    git_commit=subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
                    dependencies=dependency_versions(), command=sys.argv,
                    scale='normalized raw pointmaps and camera poses; separate predicted metric depth and explicit metric derivatives',
                    world_frame='first input camera reference',
                    local_pointmaps='no native local pointmap head; pts3d_by_unprojection remains world-frame geometry')
    print(json.dumps(dict(event='loading_model', **metadata)), flush=True)
    try:
        model = prepare_model(args.checkpoint)
    except Exception as exc:
        metadata.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        failure_path = EXPERIMENT / ('AMB3R_initialization_failure' + args.suffix + '.json')
        if failure_path.exists():
            failure_path = failure_path.with_name(failure_path.stem + '_' + str(time.time_ns()) + '.json')
        json_write(failure_path, metadata)
        raise
    metadata['checkpoint_backend_interp_v2'] = bool(model.backend.interp_v2.item())
    print('model loaded strict=True; ready for official forward', flush=True)
    results = []
    modes = [0, 1] if args.mode == 'both' else [int(args.mode == 'on')]
    for iters in modes:
        label = 'AMB3R_Base_' + ('ON' if iters else 'OFF') + args.suffix
        results.append(run_stage(model, inputs, iters, EXPERIMENT / label, metadata, args.seed))
    if len(results) == 2:
        json_write(EXPERIMENT / ('AMB3R_Base_OFF_ON_pair' + args.suffix + '.json'),
                   dict(input_identical=results[0]['shared_input_sha256'] == results[1]['shared_input_sha256'],
                        stage0_world_points_identical=results[0]['stage0_world_points_sha256'] == results[1]['stage0_world_points_sha256'],
                        off=results[0], on=results[1]))


if __name__ == '__main__':
    main()
