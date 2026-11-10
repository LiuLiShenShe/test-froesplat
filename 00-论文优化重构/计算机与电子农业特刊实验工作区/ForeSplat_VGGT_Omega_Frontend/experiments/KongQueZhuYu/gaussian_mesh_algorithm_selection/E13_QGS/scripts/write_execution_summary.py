"""Summarize completed native E13 outputs using actual records only."""
from pathlib import Path
import json, csv, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def read(name):
    path=ROOT/name
    assert path.is_file(), f'Actual completion record missing: {path}'
    return json.loads(path.read_text())

def main():
    # The already running parent queue loaded its stage list before this
    # optional CPU audit was added. Generate it here if needed without redoing
    # training or launching any GPU work.
    if not (ROOT/'native_loss_log_audit.json').is_file():
        from audit_native_loss_log import main as audit_losses
        audit_losses()
    training=read('training_complete.json')
    params=read('actual_training_parameters.json');cfg=params['config'];preset=params.get('native_preset','base')
    opt=cfg['optimizer'];pipe=cfg['pipeline']
    archived=ROOT/'runs/base_expandable_oom_13301/training_execution_records.json'
    base_records=json.loads(archived.read_text()) if archived.is_file() else []
    archived_base_seconds=sum(float(r.get('duration_seconds',0)) for r in base_records)
    evaluation=read('evaluation_complete.json')
    mesh=read('meshes/native_TSDF/complete.json')
    checkpoint=read('checkpoint_provenance_validation.json')
    loss_audit=read('native_loss_log_audit.json')
    kernel=read('kernel_validation.json')
    native=read('native_geometry/statistics.json')
    viewer=read('native_viewer/smoke/complete.json')
    cameras=read('runtime_cameras.json')
    assert training['iterations']==30000 and evaluation['views']==208
    assert checkpoint['status']=='passed' and kernel['status']=='passed' and loss_audit['status']=='passed'
    assert cameras['all208_GT_hashes_identical_to_E10']
    assert viewer['actual_server_terminated'] and viewer['native_HTTP_rendering_verified']
    terminal_provenance=next(r for r in checkpoint['records'] if r['iteration']==30000)
    metrics=list(csv.DictReader((ROOT/'per_view_metrics.csv').open()))
    assert len(metrics)==208
    native_ply=Path(training['native_quadratic_ply']).resolve()
    artifact_paths=[native_ply,Path(training['checkpoint']).resolve(),
        ROOT/'meshes/native_TSDF/raw.ply',ROOT/'meshes/native_TSDF/post.ply',
        ROOT/'meshes/native_TSDF/official_top10_post.ply',
        ROOT/'viewer_exports/quadratic_centers_rgb_point_cloud.ply',
        Path(native['parameter_archive']).resolve()]
    artifacts=[]
    for path in artifact_paths:
        assert path.is_file(),f'Requested native artifact does not exist: {path}'
        artifacts.append(dict(path=str(path.resolve()),size_bytes=path.stat().st_size))
    keys=['full_psnr','fg_mask_psnr','ssim','lpips_alex','fg_gradient_retention',
          'foreground_alpha_mean','background_alpha_mean','alpha_mask05_iou']
    summary={k:float(np.mean([float(r[k]) for r in metrics])) for k in keys}
    lines=['# E13 native QGS actual execution summary','',
        f"Actual official native training completed at {training['iterations']:,} iterations, "
        f"seed 0, physical GPU1, producing {training['quadratic_primitives']:,} quadratic primitives. "
        'All 208 corrected cameras and training RGB hashes match the frozen E10 reference.', '',
        '| Unified 208-view metric | Mean |','|---|---:|']
    lines += [f'| {k} | {v:.9g} |' for k,v in summary.items()]
    lines += ['', 'RGB metrics use unclipped native float renders saved to NPZ, then clip RGB to [0,1] '
        'for PSNR, SSIM, LPIPS and gradients. Raw means no EO or appearance correction. '
        'SSIM and Alex LPIPS use exactly the historical E10/E11 evaluation recipe.', '',
        f"Recorded training wall time across attempts: {training['training_wall_seconds']/60:.3f} minutes. "
        f"Saved prefix plus successful resumed training: {training.get('successful_model_training_wall_seconds',training['training_wall_seconds'])/60:.3f} minutes. "
        f"Successful segment: {training.get('successful_segment_wall_seconds',training['training_wall_seconds'])/60:.3f} minutes. "
        f"Successful-process peak allocated VRAM: {training['peak_allocated_gib']:.3f} GiB; "
        f"peak reserved: {training['peak_reserved_gib']:.3f} GiB.", '',
        f"Primary native preset: {preset}; resume checkpoint: {training.get('resume_checkpoint')}. "
        f"Archived base-preset attempts cost {archived_base_seconds/60:.3f} actual training minutes. "
        'Two actual base attempts failed at11700 and13301; failed native models and original logs '
        'are separately retained. The13301 failed PTH is unreadable by Python ZIP and the native ' 
        'PyTorch archive reader; its separate native PLY has the complete expected payload length. ' 
        'The failed PTH is not a resume checkpoint. A fresh Mip run uses the same VGGT initialization, cameras and '
        'training RGB; no base checkpoint is promoted or resumed across different native presets. '
        'Native resumed captures, if any, lack RNG/viewpoint-stack state and precede the optimizer '
        'step; uninterrupted trajectory equivalence is never inferred.', '',
        f"Native terminal parameter audit: {terminal_provenance['native_nonfinite_primitives']:,} "
        f"nonfinite primitives ({terminal_provenance['native_nonfinite_fraction']:.9g} of the model). "
        'The original native checkpoint and PLY retain them unchanged; exact float32 bit equality '
        'validates provenance, while any nonfinite count is a native numerical-stability defect '
        'and is not reported as an all-fields-finite success. Actual rendered outputs and metrics '
        'are validated separately. Derived shape categories exclude undefined parameters.', '',
        'Wall time includes the frozen original-RGB adapter and upstream iterable get_mask IO. '
        'The separate native_mask_IO_diagnostic.json records the exact CPU mask getter on one '
        'representative source view (no GPU); this overhead is not intrinsic quadratic primitive '
        'compute and is not a pure-GPU algorithm speed measurement.', '',
        f"Actual native {preset}: depth_ratio={pipe['depth_ratio']}, lambda_dist={opt['lambda_dist']}, "
        f"lambda_normal={opt['lambda_normal']}, lambda_dssim={opt['lambda_dssim']}; "
        f"densify threshold={opt['densify_grad_threshold']}, percent_dense={opt['percent_dense']}, "
        f"interval={opt['densification_interval']}, from={opt['densify_from_iter']}, "
        f"until={opt['densify_until_iter']}, opacity reset={opt['opacity_reset_interval']}. "
        f"Strict geometry/NCC branch: {opt['multi_view_weight_from_iter']}<iteration<"
        f"{opt['multi_view_weight_until_iter']}. No extra depth/normal prior or EO.", '',
        'The raw training CSV preserves upstream conditional geo/NCC Python variables, which can '
        'hold prior-sample values when a branch contributes nothing. native_loss_log_audit.json '
        'records this distinction; terminal iteration30000 has zero actual geo/NCC contribution '
        'even if those raw logged variables remain nonzero. Actual total_loss is authoritative.', '',
        'The native PLY has scale_0..5 and retains quadratic primitive intersections. '
        'The decoder is s=exp(scale0..2)*tanh(scale3..5); the third signed scale controls local '
        'quadratic curvature rather than ordinary 3DGS thickness. Native Gaussian/mean curvature '
        'diagnostics are saved separately; they do not establish ground-truth geometric accuracy.', '',
        'Native bounded TSDF uses official SH0/kernel_size=-999/stop_z_gradient=True extraction '
        'maps, the corrected cameras, original RAP foreground fusion constraint and shared voxel/truncation. '
        'Original SH0 float RGB remains in NPZ; mesh RGB8 input is clipped to[0,1] to prevent '
        'out-of-range SH colors wrapping under the official uint8 cast. Depth and TSDF fusion are unchanged. '
        'The raw mesh, official top10-component post mesh, and comparable >=50-face-only post mesh '
        'are preserved. No smoothing or hole filling is applied. Exact counts and extraction time '
        'are in meshes/native_TSDF/complete.json.', '',
        '| Actual artifact | Bytes |','|---|---:|']
    lines += [f"| `{r['path']}` | {r['size_bytes']} |" for r in artifacts]
    lines += ['', 'The center XYZRGB PLY is a Point Cloud PLY for center inspection; it is not a '
        'Gaussian PLY or quadratic surface approximation. Inspect the native six-scale PLY with '
        'scripts/native_qgs_viewer.py using the original quadratic CUDA renderer. The actual '
        'terminal model-backed HTTP smoke verified RGB, alpha, unit camera-normal, camera-Z depth, '
        'gray primitive, orbit, zoom and 180° view-only responses. The test server was terminated '
        'before the following experiment. Optional 180° display transforms never modify original geometry.', '',
        'This execution record makes no geometric superiority claim from PSNR alone. Final ordering '
        'must use the common 208-view silhouette/coverage/roughness/hole diagnostics and actual '
        'fixed/free visual comparisons with E10/E11/E12/E14. No same-plant scan reference exists.', '',
        'Source: [official QGS pinned commit](https://github.com/will-zzy/QGS/tree/'
        '74d05c945e99fcaef7afe5a8831903be71ad9b55), local actual completion records and model files. '
        'Bocha search was unavailable with HTTP403; its reason was not established; no training-data-only claim is used.','']
    report=ROOT/'E13_RUN_SUMMARY.md'
    report.write_text('\n'.join(lines))
    record=dict(status='complete',report=str(report.resolve()),summary=summary,artifacts=artifacts,
                training_cost=dict(active_preset=preset,archived_base_attempt_seconds=archived_base_seconds,all_preset_actual_attempt_seconds=training['training_wall_seconds']+archived_base_seconds,all_attempts_seconds=training['training_wall_seconds'],
                  successful_prefix_plus_segment_seconds=training.get('successful_model_training_wall_seconds',training['training_wall_seconds']),
                  successful_segment_seconds=training.get('successful_segment_wall_seconds',training['training_wall_seconds'])),
                native_model_distinct_from_center_point_cloud=True,viewer_smoke_actual=True,
                report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    (ROOT/'execution_summary_complete.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Actual native E13 execution summary written',report,flush=True)

if __name__=='__main__':main()
