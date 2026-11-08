"""CPU-only inspection of saved heads; no prediction alignment or modification."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    'world_pointnorm_median', 'depth_unprojection_worldnorm_median',
    'median_worldnorm_pointwise_ratio', 'ratio_of_worldnorm_medians',
    'median_camera_norm_ratio', 'median_pointhead_camera_z_over_depth',
    'direct_pointhead_positive_camera_z_fraction',
    'direct_pointhead_reprojection_median_px', 'unprojection_reprojection_median_px',
)


def diagnose(bundle_dir):
    path = Path(bundle_dir)
    bundle = json.loads((path / 'bundle.json').read_text())
    checks = json.loads((path / 'camera_geometry_checks.json').read_text())
    rows = []
    for i, name in enumerate(bundle['input_names']):
        stem = Path(name).stem
        mask = np.asarray(Image.open(path / 'mask' / (stem + '.png'))) > 0
        with np.load(path / 'raw_predictions' / (stem + '.npz')) as f:
            xyz = f['world_points'].astype(np.float64)
            up = f['pts3d_by_unprojection'].astype(np.float64)
            confidence = f['world_points_conf'].reshape(mask.shape)
            depth = f['depth'].reshape(mask.shape).astype(np.float64)
            w2c = f['extrinsic'].astype(np.float64)
            pose = f['pose'].astype(np.float64)
        norms = np.linalg.norm(xyz, axis=-1)
        up_norms = np.linalg.norm(up, axis=-1)
        valid = mask & np.isfinite(xyz).all(-1) & np.isfinite(up).all(-1) & (up_norms > 1e-12)
        if not valid.any():
            rows.append(dict(name=name, status='empty_finite_foreground'))
            continue
        point_camera = xyz @ w2c[:, :3].T + w2c[:, 3]
        up_camera = up @ w2c[:, :3].T + w2c[:, 3]
        depth_valid = valid & np.isfinite(depth) & (depth > 1e-12) & (point_camera[..., 2] > 0)
        camera_norm = np.linalg.norm(point_camera, axis=-1)
        up_camera_norm = np.linalg.norm(up_camera, axis=-1)
        ratio = norms[valid] / up_norms[valid]
        row = dict(
            name=name, foreground_points=int(valid.sum()),
            world_pointnorm_median=float(np.median(norms[valid])),
            depth_unprojection_worldnorm_median=float(np.median(up_norms[valid])),
            median_worldnorm_pointwise_ratio=float(np.median(ratio)),
            worldnorm_ratio_iqr=np.quantile(ratio, [.25, .75]).tolist(),
            ratio_of_worldnorm_medians=float(np.median(norms[valid]) / np.median(up_norms[valid])),
            median_camera_norm_ratio=float(np.median(camera_norm[valid] / np.maximum(up_camera_norm[valid], 1e-12))),
            median_pointhead_camera_z_over_depth=float(np.median(point_camera[..., 2][depth_valid] / depth[depth_valid])) if depth_valid.any() else None,
            direct_pointhead_positive_camera_z_fraction=float(depth_valid.sum() / valid.sum()),
            pointhead_world_xyz_median=np.median(xyz[valid], axis=0).tolist(),
            unprojection_world_xyz_median=np.median(up[valid], axis=0).tolist(),
            camera_translation_norm=float(np.linalg.norm(pose[:3, 3])),
            camera_c2w_identity_max_error=float(np.abs(pose - np.eye(4)).max()),
            direct_pointhead_reprojection_median_px=checks[i]['direct_world_points_reprojection_median_px'],
            unprojection_reprojection_median_px=checks[i]['depth_unprojection_reprojection_median_px'],
            foreground_confidence_mean=float(confidence[valid].mean()),
            foreground_confidence_floor1_fraction=float((confidence[valid] == 1).mean()),
        )
        rows.append(row)
    def med(records, key):
        values = [row[key] for row in records if row.get(key) is not None]
        return float(np.median(values)) if values else None
    result = dict(
        method=bundle['method'], frame_count=len(rows),
        operation='CPU-only inspection of saved original predictions; no alignment, fitting, rescaling, or output modification',
        units='original normalized world pointmaps and camera poses; no metric/GT scaling applied',
        norm_ratio_definition='per foreground pixel norm(world_points)/norm(pts3d_by_unprojection), then median; both remain world-frame pointmaps; norm ratios are not fitted scene scale estimates',
        projection_definition='camera head extrinsic maps world to camera; positive-camera-z foreground; head disagreement is not ground-truth camera error',
        first_frame=rows[0],
        other_frames_view_medians={key: med(rows[1:], key) for key in FIELDS},
        all_frames_view_medians={key: med(rows, key) for key in FIELDS},
        per_frame=rows,
    )
    (path / 'head_consistency_diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle_dirs', nargs='*', type=Path)
    args = parser.parse_args()
    paths = args.bundle_dirs or [ROOT / 'experiments' / 'KongQueZhuYu' / ('AMB3R_Base_' + mode) for mode in ('OFF', 'ON')]
    for path in paths:
        result = diagnose(path)
        print(json.dumps(dict(path=str(path), summary=result['all_frames_view_medians']), ensure_ascii=False))


if __name__ == '__main__':
    main()
