#!/usr/bin/env python3
"""Headless official SfM runner with observation-only provenance and bundle export.

The official AMB3R_SfM.run, clustering, registration, keyframe selection and
non-keyframe refinement execute unchanged. No segmentation/sky model or mesh.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import contextlib
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
AMB_ROOT = ROOT / 'third_party' / 'amb3r'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AMB_ROOT))
sys.path.insert(0, str(AMB_ROOT / 'thirdparty'))
from frontends.common import EXPERIMENT, load_inputs, export_bundle, json_write, sha256, as_numpy
from frontends.sfm.report import write_run_report


def tensor_signature(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def storage_key(tensor):
    return tensor.untyped_storage().data_ptr()


class RecordingModel(nn.Module):
    """Observe both official entry points, including direct initialization calls."""
    def __init__(self, model, images, output, local_compression='none', masks=None):
        super().__init__()
        self.model = model
        self.device = model.device
        self.name = model.name
        self.output = Path(output)
        self.local_compression = local_compression
        self.evaluation_masks = masks
        self.signatures = {tensor_signature(image): i for i, image in enumerate(images)}
        if len(self.signatures) != len(images):
            raise ValueError('Duplicate input image contents need explicit frame provenance; refusing ambiguous observer')
        self.calls = []
        self.current_call = None
        self.sources = {}
        self.frame_sources = {}
        self.active_frame_sources = {}
        self.phase = 'clustering'
        self.last_intrinsics = np.full((len(images), 3, 3), np.nan, dtype=np.float32)
        self.accepted_calls = [[] for _ in images]
        self.latest_frame_call = {}
        self.output.joinpath('local_predictions').mkdir(parents=True, exist_ok=True)

    def extract_amb3r_sfm_features(self, frames):
        return self.model.extract_amb3r_sfm_features(frames)

    def run_amb3r_sfm(self, frames, cfg, keyframe_memory=None, benchmark_conf0=None):
        indices = [self.signatures[tensor_signature(image)] for image in frames['images'][0]]
        call_id = len(self.calls)
        record = dict(call_id=call_id, phase=self.phase, frame_indices=indices,
                      local_reference_frame=indices[0], anchor_frame_indices=frames.get('kf_idx', []),
                      benchmark_early_stop_requested=benchmark_conf0 is not None, status='running')
        self.calls.append(record)
        self.current_call = call_id
        self.active_frame_sources = {index: call_id for index in indices}
        started = time.perf_counter()
        try:
            model_frames = frames
            if self.name == 'fast3r' and self.evaluation_masks is not None:
                # External masks are used only by independent projection checks.
                # Fast3R model views and official PnP arguments are unchanged.
                model_frames = dict(frames, masks=self.evaluation_masks[indices])
            result = self.model.run_amb3r_sfm(model_frames, cfg, keyframe_memory, benchmark_conf0)
            record['seconds'] = time.perf_counter() - started
            if result is None:
                record['status'] = 'official_benchmark_early_stop'
                return None
            record['status'] = 'predicted'
            export_started = time.perf_counter()
            # Feature tensors are internal activations, not geometric predictions.
            raw = {key: as_numpy(result[key]) for key in (
                'world_points', 'world_points_conf', 'pose', 'intrinsic', 'extrinsic',
                'depth', 'depth_metric', 'depth_conf', 'pts3d_by_unprojection',
                'median_metric_z', 'median_metric_z_log', 'benchmark_conf0',
                'local_points', 'local_confidence', 'focal', 'pose_enc') if key in result}
            raw['frame_indices'] = np.asarray(indices, dtype=np.int64)
            writer = np.savez_compressed if self.local_compression == 'deflate' else np.savez
            writer(self.output / 'local_predictions' / f'call_{call_id:05d}.npz', **raw)
            record['raw_export_seconds'] = time.perf_counter() - export_started
            record['raw_compression'] = self.local_compression
            self.sources[storage_key(result['world_points'])] = call_id
            for local_index, index in enumerate(indices):
                self.frame_sources[(index, tensor_signature(result['world_points'][0, local_index]))] = call_id
            for index in indices:
                self.latest_frame_call[index] = call_id
            k = raw.get('intrinsic')
            record['intrinsics_model'] = k[0].tolist() if k is not None else None
            record['confidence_mean'] = float(raw['world_points_conf'].mean())
            record['confidence_range'] = [float(raw['world_points_conf'].min()), float(raw['world_points_conf'].max())]
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'last_pose_status'):
                record['pnp_status'] = self.model.model.last_pose_status
                record['geometry_validation'] = self.model.model.last_geometry_validation
            json_write(self.output / 'local_predictions' / f'call_{call_id:05d}.json', record)
            return result
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error),
                          seconds=time.perf_counter() - started)
            json_write(self.output / 'local_predictions' / f'call_{call_id:05d}.json', record)
            raise

    def note_accepted(self, indices, call_id=None):
        for index in map(int, indices):
            source_id = self.active_frame_sources.get(index, self.current_call) if call_id is None else call_id
            if source_id is None:
                continue
            call = self.calls[source_id]
            if index not in call['frame_indices']:
                continue
            local_index = call['frame_indices'].index(index)
            if call['intrinsics_model'] is not None:
                self.last_intrinsics[index] = call['intrinsics_model'][local_index]
            self.accepted_calls[index].append(source_id)

    def resolve_frame_sources(self, points, indices):
        """Resolve copied/filtered candidate tensors and per-frame best selections.

        Official coarse registration filters points with Boolean indexing and
        replaces only improved frames. Storage identity alone loses provenance.
        Hashing unchanged per-frame coordinates handles these copies exactly.
        """
        fallback = self.sources.get(storage_key(points), self.current_call)
        return {int(index): self.frame_sources.get((int(index), tensor_signature(points[local_index])), fallback)
                for local_index, index in enumerate(indices)}


def make_observer_pipeline_class():
    from sfm.pipeline import AMB3R_SfM

    class ObservedSfM(AMB3R_SfM):
        """All overrides delegate directly to official implementations."""
        def __init__(self, model, cfg_path, output):
            super().__init__(model, cfg_path)
            self.output = Path(output)
            self.snapshots = []

        def snapshot(self, phase):
            memory = self.keyframe_memory
            entry = dict(phase=phase, keyframe_indices=memory.kf_idx,
                         keyframe_clusters=memory.kf_clusters,
                         official_unmapped_frames=sorted(memory.unmapped_frames),
                         initialized_frames=torch.nonzero(memory.iter > 0).flatten().tolist(),
                         frame_mean_sig_confidence=memory.conf.mean(dim=(1, 2)).tolist(),
                         update_counts=memory.iter.tolist())
            self.snapshots.append(entry)
            json_write(self.output / 'registration_stages.json', self.snapshots)
            print(f'SFM_STAGE {phase} initialized={len(entry["initialized_frames"])} '
                  f'keyframes={len(memory.kf_idx or [])} unmapped={len(memory.unmapped_frames)}', flush=True)

        def observe_memory(self):
            memory = self.keyframe_memory
            recorder = self.model
            original_fuse = memory.fuse_into_memory
            original_update = memory.update
            original_update_kf = memory.update_kf
            original_initialize = memory.initialize

            def fuse(idx, *args, **kwargs):
                result = original_fuse(idx, *args, **kwargs)
                recorder.note_accepted(idx)
                return result

            def update(res, *args, **kwargs):
                previous = recorder.current_call
                previous_frames = recorder.active_frame_sources
                recorder.current_call = recorder.sources.get(storage_key(res['pts']), previous)
                recorder.active_frame_sources = recorder.resolve_frame_sources(res['pts'], res['idx'])
                try:
                    return original_update(res, *args, **kwargs)
                finally:
                    recorder.current_call = previous
                    recorder.active_frame_sources = previous_frames

            def update_kf(res, *args, **kwargs):
                result = original_update_kf(res, *args, **kwargs)
                if result[0] is not None:
                    recorder.sources[storage_key(result[0]['pts'])] = recorder.current_call
                    for local_index, index in enumerate(result[0]['idx']):
                        source_id = recorder.active_frame_sources.get(int(index), recorder.current_call)
                        recorder.frame_sources[(int(index), tensor_signature(result[0]['pts'][local_index]))] = source_id
                return result

            def initialize(predictions, best):
                result = original_initialize(predictions, best)
                self.map_reference_usable_index = int(best)
                self.initial_map_reference_c2w = as_numpy(memory.poses[int(best)]).copy()
                # Native initialization averages multiple first-view predictions.
                for pred in predictions.values():
                    sources = recorder.resolve_frame_sources(pred['pred']['pts'], pred['idx'])
                    for index, call_id in sources.items():
                        recorder.note_accepted([index], call_id)
                return result

            memory.fuse_into_memory = fuse
            memory.update = update
            memory.update_kf = update_kf
            memory.initialize = initialize

        def extract_features(self, images, chunk_size=128):
            # Smaller encoder batches only; all image descriptors still clustered together.
            result = super().extract_features(images, chunk_size=8)
            np.save(self.output / 'image_descriptors.npy', as_numpy(result))
            json_write(self.output / 'descriptor_statistics.json',
                       dict(dtype=str(result.dtype), shape=list(result.shape),
                            original_model_dtype=self.model.model.last_descriptor_original_dtype,
                            external_descriptor_fp32=self.model.model.descriptor_fp32,
                            finite=bool(torch.isfinite(result).all()),
                            mean=float(result.float().mean()), std=float(result.float().std())))
            return result

        def run(self, *args, **kwargs):
            # Observe the actual distance computation before official clustering.
            # This catches invalid whitening instead of silently clustering NaNs.
            import sfm.pipeline as official
            original = official.get_distance_matrix
            def observe_distance(descriptors, *distance_args, **distance_kwargs):
                result = original(descriptors, *distance_args, **distance_kwargs)
                np.save(self.output / 'clustering_distance_matrix.npy', as_numpy(result))
                if not torch.isfinite(result).all():
                    raise FloatingPointError('Official descriptor whitening produced non-finite distances; '
                                             f'descriptor dtype={descriptors.dtype}, shape={tuple(descriptors.shape)}. '
                                             'Raw descriptors retained; no clusters or geometry fabricated.')
                return result
            official.get_distance_matrix = observe_distance
            try:
                return super().run(*args, **kwargs)
            finally:
                official.get_distance_matrix = original

        def process_clusters(self, images, clusters, distance_matrix=None):
            self.observe_memory()
            self.model.phase = 'initial_cluster_prediction'
            json_write(self.output / 'initial_clusters.json', clusters)
            if distance_matrix is not None:
                np.save(self.output / 'clustering_distance_matrix.npy', as_numpy(distance_matrix))
            result = super().process_clusters(images, clusters, distance_matrix)
            json_write(self.output / 'restructured_clusters.json',
                       dict(clusters=result[0], best_initial_keyframe=result[1]))
            return result

        def initialize_map(self, *args, **kwargs):
            self.model.phase = 'initialization'
            result = super().initialize_map(*args, **kwargs)
            self.snapshot('initialization')
            return result

        def coarse_registration(self, *args, **kwargs):
            self.model.phase = 'coarse_registration'
            result = super().coarse_registration(*args, **kwargs)
            self.snapshot('coarse_registration')
            return result

        def keyframe_mapping(self, *args, **kwargs):
            self.model.phase = 'keyframe_refinement'
            result = super().keyframe_mapping(*args, **kwargs)
            self.snapshot('keyframe_refinement')
            return result

        def non_keyframe_mapping(self, *args, **kwargs):
            self.model.phase = 'non_keyframe_refinement'
            result = super().non_keyframe_mapping(*args, **kwargs)
            self.snapshot('non_keyframe_refinement')
            return result

    return ObservedSfM


def validate_final_geometry(points, poses, intrinsics, masks, statuses):
    frames = []
    for index, (xyz, c2w, k, mask, status) in enumerate(zip(points, poses, intrinsics, masks, statuses)):
        if status['status'] != 'registered':
            frames.append(dict(frame_index=index, status=status['status'], measured_points=0))
            continue
        h, w = mask.shape
        yy, xx = np.indices((h, w))
        camera = xyz @ np.linalg.inv(c2w)[:3, :3].T + np.linalg.inv(c2w)[:3, 3]
        selection = mask & np.isfinite(camera).all(-1) & (camera[..., 2] > 1e-8)
        projected = camera[selection] @ k.T
        projected = projected[:, :2] / projected[:, 2:3]
        errors = np.linalg.norm(projected - np.stack([xx[selection], yy[selection]], -1), axis=-1)
        errors = errors[np.isfinite(errors)]
        frames.append(dict(frame_index=index, status=status['status'], measured_points=len(errors),
                           foreground_reprojection_median_px=float(np.median(errors)) if len(errors) else None,
                           foreground_reprojection_p90_px=float(np.quantile(errors, .9)) if len(errors) else None,
                           rotation_determinant=float(np.linalg.det(c2w[:3, :3])),
                           orthogonality_error=float(np.linalg.norm(c2w[:3, :3].T @ c2w[:3, :3] - np.eye(3))),
                           positive_foreground_depth_fraction=float((camera[..., 2][mask] > 0).mean())))
    return dict(convention='camera_to_world; inverse transforms exported world XYZ to camera',
                intrinsic_limitation='Official memory merges pointmaps/poses, but does not optimize/store intrinsics; '
                                     'K is from latest participating accepted local prediction. Residual includes this mismatch.',
                frames=frames)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['amb3r', 'fast3r'], required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path, default=EXPERIMENT / 'manifest.csv')
    parser.add_argument('--views', type=int, default=0, help='0 uses every valid manifest frame')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg', type=Path, default=AMB_ROOT / 'sfm' / 'sfm_config.yaml')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--head-views', type=int, default=2)
    parser.add_argument('--descriptor-fp32', action='store_true',
                        help='Losslessly cast pooled descriptors to FP32 before original whitening; record separate retry')
    parser.add_argument('--retry-of', type=Path, help='Previous failed attempt directory for explicit provenance')
    parser.add_argument('--persistent-output', type=Path,
                        help='Record intended final persistent directory when running on temporary SSD; does not copy files')
    parser.add_argument('--local-compression', choices=['none', 'deflate'], default='none',
                        help='Lossless local raw NPZ serialization; uncompressed avoids repeated GPU-idle deflate overhead')
    parser.add_argument('--force-low-confidence-mapping', action='store_true',
                        help='Reproduce official allow_unmapped_frames=false; default records failures honestly')
    args = parser.parse_args()
    from omegaconf import OmegaConf
    import cv2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setRNGSeed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    method = 'AMB3R_SfM' if args.model == 'amb3r' else 'Fast3R_AMB3R_SfM'
    out = args.output or EXPERIMENT / method
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Preserving existing run: {out}. Choose a new output directory.')
    out.mkdir(parents=True, exist_ok=True)
    with open(args.manifest) as f:
        available = sum(row['status'] == 'ok' for row in csv.DictReader(f))
    indices = list(range(args.views)) if 0 < args.views < available else None
    edge, alignment = (518, 14) if args.model == 'amb3r' else (512, 16)
    inputs = load_inputs(args.manifest, edge, alignment, indices)
    n = len(inputs['rows'])
    cfg = OmegaConf.load(args.cfg)
    cfg.device = 'cuda:0'
    cfg.allow_unmapped_frames = not args.force_low_confidence_mapping
    cfg_path = out / 'sfm_config.yaml'
    OmegaConf.save(cfg, cfg_path)
    meta = dict(method=method, seed=args.seed, full_sequence=n == available, input_views=n,
                available_views=available, gpu_physical=1, torch_version=torch.__version__,
                torch_cuda=torch.version.cuda, long_edge=edge, padding_alignment=alignment,
                inference_mode='official image clustering, coarse registration and global mapping',
                core_network_modified=False, official_sfm_modified=False,
                config_changes=dict(device='cuda:0 under CUDA_VISIBLE_DEVICES=1',
                                    allow_unmapped_frames=bool(cfg.allow_unmapped_frames)),
                official_default_allow_unmapped_frames=False,
                feature_extraction_chunk_size=8,
                descriptor_fp32=args.descriptor_fp32,
                retry_of=str(args.retry_of) if args.retry_of else None,
                execution_output_directory=str(out.resolve()),
                final_persistent_output_directory=str((args.persistent_output or out).resolve()),
                output_storage_policy=('temporary SSD output; every artifact must be copied and verified before canonical completion'
                                       if args.persistent_output else 'direct persistent output'),
                local_prediction_compression=args.local_compression,
                confidence_raw_semantics='1 + exp(logit); direct model values',
                confidence_memory_semantics='official mean of (raw_conf-1)/raw_conf',
                final_confidence='recovered as 1/(1-memory_sig); raw local confidences saved separately',
                pointcloud_processing='Official SfMemory confidence weighted fusion only; no additional frontend fusion',
                timing_scope=dict(pipeline='wall time of features/clustering/mapping including observation and raw export',
                                  local_model='wall time around official local model call; Fast3R includes CPU PnP',
                                  local_raw_export='D2H transfers, array conversion and NPZ serialization/writes'),
                intrinsic_provenance='last accepted local prediction participating in official memory; no global intrinsic optimizer',
                world_frame='official initialized map reference; transformed/registered clusters',
                scale='ambiguous normalized pointmap/pose scale; not verified metres')
    meta['amb3r_git_commit'] = subprocess.check_output(['git', '-C', str(AMB_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    json_write(out / 'run_parameters.json', meta)
    started = time.perf_counter()
    recorder = pipeline = None
    log_path = out / 'pipeline.log'
    try:
        if args.model == 'amb3r':
            from amb3r.model import AMB3R
            model = AMB3R()
            weights = args.weights or AMB_ROOT / 'checkpoints' / 'amb3r.pt'
            model.load_weights(str(weights), data_type='bf16', strict=True)
            meta['weight_source'] = 'https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel/view?usp=sharing'
            meta['backend_interp_v2'] = bool(model.backend.interp_v2.item())
            meta['metric_scale_note'] = 'Metric head depth_metric is saved per local call. Official SfM uses normalized world_points and pose and estimates intercluster scale; final map not certified metric.'
        else:
            from frontends.sfm.fast3r_adapter import Fast3RForAMB3RSfM
            model = Fast3RForAMB3RSfM(args.weights, head_parallel_views=args.head_views)
            weights = (args.weights or ROOT / 'weights' / 'fast3r') / 'model.safetensors'
            meta['fast3r_git_commit'] = subprocess.check_output(['git', '-C', str(ROOT / 'third_party' / 'fast3r'), 'rev-parse', 'HEAD'], text=True).strip()
            meta['feature_descriptor'] = 'mean pooled original Fast3R CroCo encoder patch tokens; original SfM whitening'
            meta['base_model_benchmark_early_stop'] = 'Not implemented for single-stage Fast3R; original SfM confidence-based candidate selection, best-frame replacement and mapping thresholds remain unchanged. Native AMB3R early-stop only avoids its backend.'
            meta['pnp_iterations'] = 100
            meta['pnp_focal_method'] = 'first_view_from_global_head'
            meta['weight_source'] = 'jedyang97/Fast3R_ViT_Large_512'
            meta['fast3r_license'] = 'FAIR Noncommercial Research License'
        meta['weight_sha256'] = sha256(weights)
        meta['weight_path'] = str(weights.resolve())
        meta['git_commit'] = meta['amb3r_git_commit'] if args.model == 'amb3r' else meta['fast3r_git_commit']
        meta['code_commits'] = dict(sfm=meta['amb3r_git_commit'], base_model=meta['git_commit'])
        model = model.to('cuda:0').eval()
        from frontends.sfm.fast3r_adapter import PipelineBridge
        recorder = RecordingModel(PipelineBridge(model, descriptor_fp32=args.descriptor_fp32), inputs['images'], out,
                                  local_compression=args.local_compression, masks=inputs['masks'])
        pipeline = make_observer_pipeline_class()(recorder, cfg_path, out)
        torch.cuda.reset_peak_memory_stats()
        json_write(out / 'run_parameters.json', meta)
        print(f'BEGIN {method} {n} views physical GPU1', flush=True)
        pipeline_started = time.perf_counter()
        with open(log_path, 'w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = pipeline.run(inputs['images'].unsqueeze(0))
        torch.cuda.synchronize()
        meta.update(inference_seconds=time.perf_counter() - pipeline_started,
                    elapsed_seconds_including_model_load=time.perf_counter() - started,
                    local_model_seconds=sum(call['seconds'] for call in recorder.calls),
                    local_raw_export_seconds=sum(call.get('raw_export_seconds', 0) for call in recorder.calls),
                    peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 2**30,
                    local_prediction_calls=len(recorder.calls))
        meta['local_call_phase_counts'] = {phase: sum(call['phase'] == phase for call in recorder.calls)
                                           for phase in sorted({call['phase'] for call in recorder.calls})}
        meta['official_benchmark_early_stop_calls'] = sum(
            call['status'] == 'official_benchmark_early_stop' for call in recorder.calls)
        meta['backend_frontend_fallback_count'] = log_path.read_text().count(
            'Unusual bad case detected in backend, skip it.')
        print(f'SFM_PIPELINE_DONE {method}; CPU geometry validation/export follows', flush=True)
        points = as_numpy(result.pts)
        sig = as_numpy(result.conf)
        poses_raw = as_numpy(result.poses)
        poses = poses_raw.copy()
        excluded = set(result.unmapped_frames)
        statuses = []
        for index in range(n):
            pose_valid = np.isfinite(poses_raw[index]).all() and abs(np.linalg.det(poses_raw[index, :3, :3])) > .1
            initialized = result.iter[index].item() > 0
            registered = initialized and pose_valid and index not in excluded
            status = dict(frame_index=index, source_index=int(inputs['rows'][index]['source_index']),
                          name=inputs['rows'][index]['name'], status='registered' if registered else 'unmapped',
                          reason=None if registered else ('official low confidence remapping rejection' if index in excluded else 'uninitialized or invalid official memory pose'),
                          mean_sig_confidence=float(sig[index].mean()), update_count=float(result.iter[index]),
                          accepted_local_call_ids=recorder.accepted_calls[index])
            pnp_contributors = []
            for call_id in recorder.accepted_calls[index]:
                call = recorder.calls[call_id]
                if call.get('pnp_status') is not None:
                    local_index = call['frame_indices'].index(index)
                    pnp_contributors.append(dict(call_id=call_id, **call['pnp_status'][local_index]))
            status['pnp_contributors'] = pnp_contributors
            status['last_local_pnp_status'] = pnp_contributors[-1]['status'] if pnp_contributors else 'not_applicable'
            status['failed_pnp_contributor_count'] = sum(row['status'] == 'failed_identity_fallback' for row in pnp_contributors)
            status['incomplete_pnp_observation_contributor_count'] = sum(
                row['status'] in ('not_called', 'started') for row in pnp_contributors)
            statuses.append(status)
            if not registered:
                excluded.add(index)
                poses[index] = np.nan
        json_write(out / 'registration_status.json', statuses)
        json_write(out / 'unmapped_frames.json', [statuses[index] for index in sorted(excluded)])
        json_write(out / 'keyframes.json', dict(indices=result.kf_idx, clusters=result.kf_clusters))
        json_write(out / 'local_call_index.json', recorder.calls)
        np.savez_compressed(out / 'official_memory_raw.npz', points=points, confidence_sig=sig,
                            c2w=poses_raw, iterations=as_numpy(result.iter),
                            keyframe_indices=np.asarray(result.kf_idx), unmapped_indices=np.asarray(sorted(excluded)))
        # Recover representation algebraically, never inflate or clamp confidence.
        with np.errstate(divide='ignore', invalid='ignore'):
            recovered = 1 / (1 - sig)
        recovered[~np.isfinite(recovered) | (sig < 0) | (sig >= 1)] = np.nan
        export_points = points.copy()
        for index in excluded:
            export_points[index] = np.nan
            recovered[index] = np.nan
        predictions = [dict(world_points=export_points[index], world_points_conf=recovered[index],
                            sfm_memory_points=points[index], sfm_memory_confidence_sig=sig[index],
                            sfm_memory_pose=poses_raw[index]) for index in range(n)]
        # Keep original per-camera local geometry and metric-head predictions
        # from the last accepted contributor, separately from fused world data.
        cached_call_id = None
        cached = None
        for index, prediction in enumerate(predictions):
            call_ids = recorder.accepted_calls[index]
            if not call_ids:
                continue
            call_id = call_ids[-1]
            if call_id != cached_call_id:
                with np.load(out / 'local_predictions' / f'call_{call_id:05d}.npz') as data:
                    cached = {key: data[key] for key in ('frame_indices', 'local_points', 'local_confidence',
                                                        'depth', 'depth_metric', 'depth_conf',
                                                        'median_metric_z', 'median_metric_z_log') if key in data}
                cached_call_id = call_id
            local_index = recorder.calls[call_id]['frame_indices'].index(index)
            for key, value in cached.items():
                if key == 'frame_indices':
                    continue
                label = key if key.startswith('local_') else 'last_local_' + key
                prediction[label] = value[0, local_index]
            prediction['last_accepted_local_call_id'] = np.asarray(call_id, dtype=np.int64)
        geometry_validation = validate_final_geometry(points, poses, recorder.last_intrinsics,
                                                     inputs['masks'], statuses)
        meta.update(mapped_frames=n - len(excluded), unmapped_frames=len(excluded))
        reference = getattr(pipeline, 'map_reference_usable_index', None)
        if reference is not None:
            reference_row = inputs['rows'][reference]
            meta.update(map_reference_usable_index=reference,
                        map_reference_source_index=int(reference_row['source_index']),
                        map_reference_frame_name=reference_row['name'],
                        initial_map_reference_c2w=pipeline.initial_map_reference_c2w,
                        initial_map_reference_identity_distance=float(np.linalg.norm(pipeline.initial_map_reference_c2w - np.eye(4))),
                        final_map_reference_c2w=poses[reference],
                        final_map_reference_identity_distance=float(np.linalg.norm(poses[reference] - np.eye(4))))
            meta['world_frame'] = f'official SfM initialized map reference: usable_index={reference}, source_index={reference_row["source_index"]}, name={reference_row["name"]}'
            json_write(out / 'map_reference.json', {key: value for key, value in meta.items()
                                                   if 'map_reference' in key})
            geometry_validation['map_reference'] = {key: value for key, value in meta.items()
                                                    if 'map_reference' in key}
        json_write(out / 'geometry_validation.json', geometry_validation)
        json_write(out / 'run_parameters.json', meta)
        # Release GPU networks before CPU serialization of the full scene.
        del model, pipeline
        pipeline = None
        recorder.model.cpu()
        torch.cuda.empty_cache()
        print(f'SFM_GPU_RELEASED {method}; writing CPU bundle', flush=True)
        bundle_started = time.perf_counter()
        export_bundle(out, inputs, predictions, poses, recorder.last_intrinsics, meta, statuses)
        meta['bundle_export_seconds'] = time.perf_counter() - bundle_started
        meta['post_pipeline_seconds'] = time.perf_counter() - pipeline_started - meta['inference_seconds']
        meta['elapsed_seconds_through_bundle_export'] = time.perf_counter() - started
        bundle = json.loads((out / 'bundle.json').read_text())
        bundle['metadata'] = meta
        json_write(out / 'bundle.json', bundle)
        json_write(out / 'run_parameters.json', meta)
        completed = dict(status='complete', mapped_frames=n - len(excluded),
                         unmapped_frames=len(excluded), input_views=n)
        json_write(out / 'status.json', completed)
        write_run_report(out, meta, completed)
        print(f'COMPLETE {method}: {n-len(excluded)}/{n} registered; bundle {out}', flush=True)
    except Exception as error:
        failure = dict(status='failed', error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.perf_counter() - started)
        if recorder is not None:
            json_write(out / 'local_call_index.json', recorder.calls)
            failure['failure_phase'] = recorder.phase
        if pipeline is not None and hasattr(pipeline, 'keyframe_memory'):
            memory = pipeline.keyframe_memory
            np.savez_compressed(out / 'failed_run_partial_memory.npz', points=as_numpy(memory.pts),
                                confidence_sig=as_numpy(memory.conf), pose=as_numpy(memory.poses),
                                iterations=as_numpy(memory.iter))
            failure['partial_initialized_frame_indices'] = torch.nonzero(memory.iter > 0).flatten().tolist()
            json_write(out / 'keyframes.json', dict(partial=True, indices=memory.kf_idx,
                                                    clusters=memory.kf_clusters))
            partial_status = [dict(frame_index=index, name=row['name'],
                                   source_index=int(row['source_index']),
                                   status='partial_initialized' if memory.iter[index] > 0 else 'not_registered_before_failure',
                                   official_unmapped=index in memory.unmapped_frames,
                                   reason='whole pipeline failed; final registration was not completed')
                              for index, row in enumerate(inputs['rows'])]
            json_write(out / 'registration_status.json', partial_status)
        json_write(out / 'status.json', failure)
        (out / 'error.txt').write_text(traceback.format_exc())
        write_run_report(out, meta, failure)
        raise


if __name__ == '__main__':
    main()
