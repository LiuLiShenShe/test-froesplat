#!/usr/bin/env python3
"""Persist a completed SfM attempt and verify the artifact set before aliasing.

CPU bookkeeping only. Geometry, weights and official prediction arrays are
never changed. Nonfinite JSON numbers become portable null; NPZ retains NaN.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import csv
import datetime
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from frontends.common import json_write, write_ply
from frontends.sfm.report import write_run_report


def artifact_sizes(directory):
    return {str(path.relative_to(directory)): path.stat().st_size
            for path in directory.rglob('*') if path.is_file()}


def preserve_unmapped_local_geometry(directory, metadata):
    """Retain valid local heads without claiming a rejected global registration."""
    if metadata['method'] != 'Fast3R_AMB3R_SfM':
        return
    registrations = json.loads((directory / 'registration_status.json').read_text())
    unmapped = [row for row in registrations if row['status'] == 'unmapped']
    if not unmapped:
        metadata['unmapped_local_geometry_supplement_count'] = 0
        return
    from PIL import Image
    calls = json.loads((directory / 'local_call_index.json').read_text())
    transforms = json.loads((directory / 'pixel_transforms.json').read_text())
    latest = {}
    for call in calls:
        if call['status'] == 'predicted':
            for index in call['frame_indices']:
                latest[index] = call
    saved = []
    for row in unmapped:
        index = row['frame_index']
        call = latest.get(index)
        if call is None:
            saved.append(dict(frame_index=index, status='no returned local prediction before failure'))
            continue
        local_index = call['frame_indices'].index(index)
        stem = Path(row['name']).stem
        raw_file = directory / 'local_predictions' / f'call_{call["call_id"]:05d}.npz'
        with np.load(raw_file) as raw:
            if 'local_points' not in raw:
                saved.append(dict(frame_index=index, status='model did not expose local head'))
                continue
            labels = dict(local_points='local_points', local_confidence='local_confidence',
                          world_points='cluster_world_points_unregistered',
                          world_points_conf='cluster_world_confidence', pose='cluster_c2w_unregistered',
                          intrinsic='intrinsics_model', focal='focal')
            values = {label: raw[key][0, local_index] for key, label in labels.items() if key in raw}
        values['source_local_call_id'] = np.asarray(call['call_id'], dtype=np.int64)
        values['source_cluster_reference_usable_index'] = np.asarray(call['local_reference_frame'], dtype=np.int64)
        supplementary = directory / 'local_unmapped_predictions'
        supplementary.mkdir(exist_ok=True)
        target = supplementary / f'{stem}.npz'
        if target.exists():
            with np.load(target) as previous:
                if not np.array_equal(previous['local_points'], values['local_points'], equal_nan=True):
                    raise FileExistsError(f'Preserving differing existing local supplement {target}')
        else:
            np.savez_compressed(target, **values)
        provenance = dict(frame_index=index, source_index=row['source_index'], name=row['name'],
                          global_registration_status='unmapped',
                          local_geometry_convention='individual camera frame; not globally registered',
                          cluster_geometry_convention='local first-view reference; explicitly unregistered',
                          source_local_call_id=call['call_id'], source_local_prediction=str(raw_file.relative_to(directory)),
                          source_cluster_reference_usable_index=call['local_reference_frame'],
                          source_cluster_frame_indices=call['frame_indices'],
                          pnp_status=(call.get('pnp_status') or [])[local_index] if call.get('pnp_status') else None,
                          global_bundle_points_and_camera_changed=False)
        json_write(supplementary / f'{stem}.json', provenance)
        rgb = np.asarray(Image.open(directory / 'rgb' / f'{stem}.png').convert('RGB'))
        mask = np.asarray(Image.open(directory / 'mask' / f'{stem}.png')) > 0
        valid = np.zeros(mask.shape, dtype=bool)
        left, top, _, _ = transforms[index]['padding_ltrb']
        width, height = transforms[index]['resize_size_wh']
        valid[top:top + height, left:left + width] = True
        xyz = values['local_points']
        conf = values['local_confidence'].reshape(mask.shape)
        cluster_conf = values['cluster_world_confidence'].reshape(mask.shape)
        sig = (cluster_conf - 1) / cluster_conf
        provenance.update(official_rejection_reason=row.get('reason'),
                          last_returned_raw_confidence_mean=float(cluster_conf.mean()),
                          last_returned_sig_confidence_mean=float(sig.mean()),
                          last_returned_foreground_sig_confidence_mean=float(sig[mask & valid].mean()) if (mask & valid).any() else None,
                          foreground_model_pixel_fraction=float((mask & valid).mean()),
                          selection_scope='official whole-model-image confidence selection; external mask only used for saved diagnostics',
                          official_configuration='sfm_config.yaml')
        json_write(supplementary / f'{stem}.json', provenance)
        finite = np.isfinite(xyz).all(-1) & valid
        confident = finite & np.isfinite(conf) & (conf > 1)
        for label, selected in dict(raw=finite, confidence=confident, mask=finite & mask,
                                    mask_confidence=confident & mask).items():
            ply = directory / 'pointcloud' / 'local_unmapped_per_frame' / f'{stem}_{label}.ply'
            if not ply.exists():
                write_ply(ply, xyz[selected], rgb[selected])
        saved.append(provenance)
    json_write(directory / 'unmapped_local_geometry_index.json', saved)
    metadata['unmapped_local_geometry_supplement_count'] = sum('source_local_call_id' in row for row in saved)
    metadata['unmapped_local_geometry_supplement'] = 'local_unmapped_predictions/{frame_stem}.npz; four PLYs under pointcloud/local_unmapped_per_frame; latest returned local call, not globally aligned'


def validate_bundle(directory, status):
    bundle = json.loads((directory / 'bundle.json').read_text())
    n = bundle['frame_count']
    names = bundle['input_names']
    registrations = json.loads((directory / 'registration_status.json').read_text())
    unmapped = json.loads((directory / 'unmapped_frames.json').read_text())
    if len(names) != n or len(registrations) != n:
        raise ValueError('Per-frame names or registration records do not match bundle frame count')
    registered = sum(record['status'] == 'registered' for record in registrations)
    if registered != status['mapped_frames'] or len(unmapped) != status['unmapped_frames']:
        raise ValueError('Final mapped/unmapped counts disagree between exported records')
    if registered + len(unmapped) != n:
        raise ValueError('Final records silently omit one or more input frames')
    with np.load(directory / 'cameras' / 'cameras.npz') as cameras:
        if cameras['c2w'].shape != (n, 4, 4) or cameras['intrinsics_model'].shape != (n, 3, 3):
            raise ValueError('Camera arrays do not include all input frames')
        camera_names = cameras['names'].tolist()
        if camera_names != names:
            raise ValueError('Camera names/order disagree with bundle inputs')
    if len(json.loads((directory / 'pixel_transforms.json').read_text())) != n:
        raise ValueError('Missing original-to-model pixel transforms')
    for name in names:
        stem = Path(name).stem
        for subdir in ('raw_predictions', 'pointmaps', 'confidence'):
            prediction = directory / subdir / f'{stem}.npz'
            with zipfile.ZipFile(prediction) as archive:
                if not archive.namelist():
                    raise ValueError(f'Empty prediction archive {prediction}')
        for subdir, suffix in (('cameras', '.json'), ('rgb', '.png'), ('mask', '.png')):
            if not (directory / subdir / (stem + suffix)).is_file():
                raise FileNotFoundError(f'Missing {subdir} for frame {name}')
    with np.load(directory / 'official_memory_raw.npz') as memory:
        if memory['c2w'].shape != (n, 4, 4) or memory['iterations'].shape != (n,):
            raise ValueError('Official raw memory lacks the full sequence')
    return dict(frame_count=n, registered_frames=registered, unmapped_frames=len(unmapped),
                camera_names_and_order_verified=True, all_prediction_archives_present=True,
                all_rgb_mask_transform_records_present=True, official_memory_frame_count_verified=True)


def backfill(directory, metadata, status, persistence):
    amb_commit = metadata['amb3r_git_commit']
    metadata.setdefault('git_commit', metadata.get('fast3r_git_commit', amb_commit))
    metadata.setdefault('code_commits', dict(sfm=amb_commit, base_model=metadata['git_commit']))
    if metadata['method'] == 'AMB3R_SfM':
        metadata.setdefault('weight_path', str(ROOT / 'third_party' / 'amb3r' / 'checkpoints' / 'amb3r.pt'))
        metadata.setdefault('weight_source', 'https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel/view?usp=sharing')
        metadata['predicted_geometry_field'] = 'Direct normalized world_points/world_points_conf head, registered by original SfMemory; depth-unprojection points are separate local diagnostics'
        metadata['base_model_benchmark_early_stop'] = 'Official Frontend confidence comparison before Backend; returned predictions also retain official Backend-to-Frontend low-confidence fallback'
    else:
        metadata['predicted_geometry_field'] = 'Fast3R local-cluster first-view global point maps, registered by original SfMemory; camera-local maps saved separately'
        metadata['base_model_benchmark_early_stop'] = 'Single-stage Fast3R returns full prediction; original SfM best-frame confidence selection/mapping thresholds are unchanged; native early-stop only avoids its Backend'
    metadata.setdefault('execution_output_directory', str(directory.resolve()))
    metadata['final_persistent_output_directory'] = str(directory.resolve())
    metadata.setdefault('output_storage_policy', 'direct persistent /data HDD output')
    metadata['persistence'] = persistence
    manifest = ROOT / 'experiments' / 'KongQueZhuYu' / 'manifest.csv'
    metadata['manifest_path'] = str(manifest)
    metadata['manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    metadata['manifest_provenance'] = 'Associated unchanged shared manifest; original RGB/mask hashes recorded by Stage0 audit. No original images are altered or reread by finalization.'
    with manifest.open() as stream:
        manifest_rows = {row['name']: row for row in csv.DictReader(stream) if row['status'] == 'ok'}
    frame_records = json.loads((directory / 'registration_status.json').read_text())
    ordered_pairs = [[row['name'], manifest_rows[row['name']]['rgb_sha256'],
                      manifest_rows[row['name']]['mask_sha256']] for row in frame_records]
    metadata['input_pair_order_sha256'] = hashlib.sha256(
        json.dumps(ordered_pairs, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    metadata['input_pair_order_hash_scope'] = 'Ordered original filenames plus original RGB and mask SHA256 strings from manifest; includes every actual input view'
    metadata.setdefault('timing_scope', dict(
        pipeline='wall time of features/clustering/mapping including observation and raw export',
        local_model='wall time around official local model call; Fast3R includes CPU PnP',
        local_raw_export='D2H transfers, array conversion and NPZ serialization/writes'))
    metadata['registered_frames'] = status.get('mapped_frames')
    metadata['inference_time_scope'] = metadata['timing_scope']['pipeline']
    if 'elapsed_seconds_through_bundle_export' in metadata:
        metadata['total_frontend_seconds'] = metadata['elapsed_seconds_through_bundle_export']
        metadata['total_frontend_time_scope'] = 'Measured from model loading after inputs were prepared through CPU bundle export; includes official pipeline and observation/raw export; excludes original input reading/preparation and final SSD-to-HDD persistence copy'
    if 'bundle_export_seconds' not in metadata:
        metadata['timing_measurement_limitations'] = 'This older live runner did not time final CPU bundle export or complete end-to-end frontend execution. Missing values are not inferred from file timestamps.'
    metadata['host_io_context'] = dict(
        input_and_environment_storage='/data ext4 /dev/sda1 HDD',
        temporary_ssd='/tmp /dev/nvme0n1p2',
        observed_pressure='shared /data I/O pressure during native run: some avg60 approximately 70%, full avg60 approximately 66%',
        timing_comparison_limitation='Native output writes directly to congested HDD; hybrid execution stages output on SSD. Pipeline wall time is not solely model compute and cannot isolate a model speed difference.')
    log = directory / 'pipeline.log'
    if log.exists():
        metadata['backend_frontend_fallback_count'] = log.read_text().count('Unusual bad case detected in backend, skip it.')
    calls_file = directory / 'local_call_index.json'
    if calls_file.exists():
        calls = json.loads(calls_file.read_text())
        phases = sorted({call['phase'] for call in calls})
        metadata['local_call_phase_counts'] = {phase: sum(call['phase'] == phase for call in calls) for phase in phases}
        metadata['official_benchmark_early_stop_calls'] = sum(call['status'] == 'official_benchmark_early_stop' for call in calls)
        pnp = [record for call in calls for record in (call.get('pnp_status') or [])]
        if pnp:
            counts = {state: sum(record['status'] == state for record in pnp)
                      for state in sorted({record['status'] for record in pnp})}
            metadata['observed_pnp_status_counts_over_all_local_calls'] = counts
            metadata['pnp_instrumentation_incomplete_views'] = sum(
                record['status'] in ('not_called', 'started') for record in pnp)
    json_write(directory / 'run_parameters.json', metadata)
    bundle_path = directory / 'bundle.json'
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text())
        bundle['metadata'] = metadata
        bundle['pointcloud_processing'] = metadata['pointcloud_processing']
        json_write(bundle_path, bundle)
    # Normalize JSON serialization only; numerical predictions remain untouched.
    normalized = 0
    for path in directory.rglob('*.json'):
        nonfinite = []
        value = json.loads(path.read_text(), parse_constant=lambda value: nonfinite.append(value))
        if nonfinite:
            json_write(path, value)
            normalized += 1
    write_run_report(directory, metadata, status)
    with (directory / 'RUN_REPORT.md').open('a') as report:
        report.write('\n存储与时间：' + metadata['output_storage_policy'] + '。')
        report.write('持久化复制 ' + str(persistence.get('copy_seconds', 0)) + ' s；')
        report.write('文件集合/尺寸已核验。')
        if status['status'] == 'complete':
            report.write('全帧 camera、RGB、Mask、预测存档已经核验。')
        else:
            report.write('保留失败日志与已产生的部分输出；未宣称全量 bundle 已完成。')
        report.write('共享 HDD 拥塞及 SSD/HDD 输出位置差异使 pipeline wall time 不能直接解释为模型速度差异。\n')
    shutil.copy2(directory / 'RUN_REPORT.md', directory / 'MODEL_RUN_REPORT.md')
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--canonical', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = (args.destination or source).absolute()
    status = json.loads((source / 'status.json').read_text())
    metadata = json.loads((source / 'run_parameters.json').read_text())
    if args.canonical and status['status'] != 'complete':
        raise ValueError('A failed or partial pipeline cannot become the canonical complete bundle')
    if status['status'] == 'complete':
        validate_bundle(source, status)
        preserve_unmapped_local_geometry(source, metadata)
    sizes = artifact_sizes(source)
    print(f'ARTIFACT_SET {metadata["method"]}: {len(sizes)} files, {sum(sizes.values())} bytes', flush=True)
    copied = source != destination.resolve()
    copy_seconds = 0.0
    if copied:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Preserving existing attempt: {destination}')
        temporary = destination.with_name(destination.name + f'.copying_{os.getpid()}')
        if temporary.exists():
            raise FileExistsError(temporary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        print(f'PERSISTENCE_COPY_BEGIN {source} -> {temporary}', flush=True)
        shutil.copytree(source, temporary)
        copied_sizes = artifact_sizes(temporary)
        if sizes != copied_sizes:
            raise ValueError(f'Persistent copy file set or sizes differ; retained at {temporary}')
        if status['status'] == 'complete':
            validate_bundle(temporary, status)
            # Spot-check actual arrays, not only archive size: first/reference/last.
            names = json.loads((source / 'bundle.json').read_text())['input_names']
            checks = {0, len(names) - 1, int(metadata.get('map_reference_usable_index', 0))}
            for index in checks:
                name = Path(names[index]).stem + '.npz'
                with np.load(source / 'raw_predictions' / name) as original, np.load(temporary / 'raw_predictions' / name) as clone:
                    for key in ('world_points', 'world_points_conf'):
                        if not np.array_equal(original[key], clone[key], equal_nan=True):
                            raise ValueError(f'Copied prediction differs: frame {index}, {key}')
        copy_seconds = time.perf_counter() - started
        temporary.rename(destination)
        print(f'PERSISTENCE_COPY_VERIFIED {copy_seconds:.3f}s; persistent bundle {destination}', flush=True)
    persistence = dict(state='complete persistent artifact set', execution_directory=str(source),
                       persistent_directory=str(destination), copied_from_temporary_storage=copied,
                       copy_seconds=copy_seconds, copied_file_count=len(sizes),
                       original_total_bytes=sum(sizes.values()), file_set_and_sizes_verified=True,
                       prediction_array_spot_checks='first, map reference, last' if copied and status['status']=='complete' else 'not applicable to same-directory finalization')
    normalized = backfill(destination, metadata, status, persistence)
    if status['status'] == 'complete':
        persistence['bundle_validation'] = validate_bundle(destination, status)
    json_write(destination / 'persistence_verification.json', persistence)
    if args.canonical:
        canonical = args.canonical.absolute()
        if canonical.is_dir() and not canonical.is_symlink() and not any(canonical.iterdir()):
            # Stage0 created empty placeholders; no experiment is overwritten.
            canonical.rmdir()
        if canonical.exists() or canonical.is_symlink():
            if not canonical.is_symlink() or canonical.resolve() != destination.resolve():
                raise FileExistsError(f'Preserving existing canonical result: {canonical}')
        else:
            canonical.symlink_to(os.path.relpath(destination, canonical.parent), target_is_directory=True)
    registry_path = ROOT / 'experiments' / 'KongQueZhuYu' / 'run_registry.json'
    registry = json.loads(registry_path.read_text())
    for record in registry['records']:
        if record['method'] == metadata['method']:
            record.update(state=status['status'], attempt_path=str(destination),
                          original_execution_path=str(source), persistence_state='complete persistent artifact set',
                          input_views=metadata['input_views'], full_sequence=metadata['full_sequence'],
                          registered_frames=status.get('mapped_frames'), unmapped_frames=status.get('unmapped_frames'),
                          keyframe_count=len(json.loads((destination / 'keyframes.json').read_text()).get('indices', [])),
                          copy_seconds=copy_seconds,
                          last_observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if args.canonical:
                record['persistent_bundle'] = str(args.canonical.absolute())
    json_write(registry_path, registry)
    print(json.dumps(dict(method=metadata['method'], status=status['status'], persistent_output=str(destination),
                          copy_seconds=copy_seconds, files=len(sizes), json_normalized=normalized,
                          registration=status.get('mapped_frames'), unmapped=status.get('unmapped_frames'))))


if __name__ == '__main__':
    main()
