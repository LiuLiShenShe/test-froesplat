from __future__ import annotations

from collections import Counter, defaultdict
import json

import numpy as np
from PIL import Image

from common import ALG, DATA, P12, P17A, ROOT, S20, V00, dimensions, git_origin, load_module, mask, read_csv, relative, report, sha256, write_csv


def p6_sources():
    roots = [DATA / '03-分割Mask/05-RAP-FSAM3掩膜', P12, V00, P17A]
    results = []
    for root in roots:
        for config in sorted(root.rglob('参数.json')):
            values = json.loads(config.read_text())
            if values.get('prompt_list') != 'P6' or values.get('use_prompt_ensemble') or values.get('prompt_selection_mode') != 'single':
                continue
            if values.get('use_cross_view_consensus') or values.get('use_memory_propagation'):
                if 'P6带盆整株' not in config.parent.name:
                    continue
            results.append((config, values))
    return results


def compare_masks(old, new):
    if old.shape != new.shape:
        raise ValueError('Mask dimensions differ; no resizing')
    intersection = int((old & new).sum())
    old_count, new_count = int(old.sum()), int(new.sum())
    union = old_count + new_count - intersection
    equal = np.array_equal(old, new)
    iou = intersection / union if union else 1.0
    return {'pixelwise_equal': equal, 'iou': iou, 'dice': 2 * intersection / (old_count + new_count) if old_count + new_count else 1.0, 'changed_pixel_ratio': float((old != new).mean()), 'foreground_area_ratio_difference': (new_count - old_count) / old.size, 'classification': 'EXACT_MATCH' if equal else 'SIMILAR_BUT_DIFFERENT' if iou >= 0.95 else 'DIFFERENT'}


def model_assets(sample):
    old = ROOT / '05-2DGS-new' / sample
    modern = DATA / '06-实验输出' / sample
    configs = [path for path in (old / 'cfg_args',) if path.exists()]
    configs += list(modern.glob('*/config.json'))
    commands = list(modern.glob('*/command.txt'))
    models = list(old.glob('point_cloud/iteration_*/point_cloud.ply')) + list(modern.glob('*/point_cloud/iteration_*/point_cloud.ply'))
    meshes = list((ROOT / '06-MESH-new' / sample).glob('*.ply')) + [path for path in modern.glob('*/*.ply') if 'mesh' in path.name or 'fuse' in path.name]
    metrics = list(modern.glob('*/foreground_object_results.json')) + list(old.glob('results*.json'))
    return configs, commands, models, meshes, metrics


def main():
    old_root = DATA / '03-分割Mask/02-sam_masks'
    samples = sorted(path.name for path in old_root.iterdir() if path.is_dir())
    loader = load_module(ROOT / '2d-gaussian-splatting-main/scene/colmap_loader.py', 'phase20b_colmap_readonly')
    colmap_names = {}
    for sample in samples:
        scene = DATA / '02-位姿COLMAP/03-final_locked' / sample
        binary = scene / 'sparse/0/images.bin'
        colmap_names[sample] = {image.name for image in loader.read_extrinsics_binary(str(binary)).values()} if binary.exists() else set()
    configs = {sample: model_assets(sample)[0] for sample in samples}
    rows = []
    for config, values in p6_sources():
        index = config.parent / '图像索引.csv'
        if not index.exists():
            continue
        for entry in read_csv(index):
            image_name = entry['图像名']
            stem = image_name.rsplit('.', 1)[0]
            if '_' in stem:
                sample, frame = stem.rsplit('_', 1)
            else:
                sample = config.parent.name.split('_')[1] if config.parent.name.startswith('E1_') else 'UNKNOWN'
                frame = stem
            p6 = config.parent / '最终掩膜' / f'mask_{stem}.png'
            original_rgb = ROOT / '02-FFT' / sample / f'{frame}.jpg'
            indexed_rgb = ROOT / entry['图像路径'] if not entry['图像路径'].startswith('/') else __import__('pathlib').Path(entry['图像路径'])
            old_path = old_root / sample / f'mask_{frame}.png'
            crop_path = ROOT / '03-SAM' / sample / f'crop_{frame}.png'
            row = {'sample_id': sample, 'frame_id': frame, 'p6_run': relative(config.parent), 'old_mask_path': relative(old_path), 'p6_mask_path': relative(p6), 'source_rgb_path': relative(indexed_rgb), 'original_rgb_path': relative(original_rgb), 'p6_config_path': relative(config), 'training_config_paths': ';'.join(relative(path) for path in configs.get(sample, [])) or 'UNKNOWN', 'source_commit': git_origin(config), 'classification': 'UNMATCHED', 'reason': 'NO_OLD_SAMPLE' if sample not in samples else 'MISSING_MASK', 'pixelwise_equal': 'UNKNOWN', 'colmap_image_name': 'UNKNOWN', 'training_coordinate_identity': 'NOT VERIFIED'}
            if sample not in samples or not old_path.exists() or not p6.exists():
                rows.append(row)
                continue
            try:
                old_mask, new_mask = mask(old_path), mask(p6)
                row.update({'old_mask_width': old_mask.shape[1], 'old_mask_height': old_mask.shape[0], 'p6_mask_width': new_mask.shape[1], 'p6_mask_height': new_mask.shape[0]})
                row['source_dimensions'] = 'x'.join(str(value) for value in dimensions(indexed_rgb))
                if old_mask.shape != new_mask.shape:
                    row.update(classification='NOT_VERIFIED', reason='MISMATCHED_DIMENSIONS')
                else:
                    with Image.open(indexed_rgb) as image:
                        rgb = np.array(image.convert('RGB'))
                    with Image.open(original_rgb) as image:
                        original = np.array(image.convert('RGB'))
                    with Image.open(crop_path) as image:
                        cropped = np.array(image.convert('RGB'))
                    identity = np.array_equal(rgb, original) and cropped.shape == rgb.shape and np.array_equal(cropped[old_mask], original[old_mask])
                    row['rgb_identity_evidence'] = 'indexed RGB exact decoded equality to FFT; old crop foreground exact equality' if identity else 'NOT VERIFIED'
                    if not identity:
                        row.update(classification='NOT_VERIFIED', reason='RGB_IDENTITY_FAILED')
                    else:
                        row.update(compare_masks(old_mask, new_mask))
                        row['reason'] = 'SAME_RAW_RGB_AND_COORDINATES'
                        row.update(old_mask_sha256=sha256(old_path), p6_mask_sha256=sha256(p6))
                        pot_path = S20 / 'GT口径拆分审计/gt_masks_split_v2' / sample / f'mask_pot_{frame}.png'
                        row['pot_region_coverage_difference'] = 'UNKNOWN'
                        if pot_path.exists():
                            pot = mask(pot_path)
                            if pot.shape == old_mask.shape and pot.any():
                                row['pot_region_coverage_difference'] = float(((new_mask & pot).sum() - (old_mask & pot).sum()) / pot.sum())
                colmap_name = next((name for name in colmap_names.get(sample, set()) if name.rsplit('.', 1)[0] in [frame, f'crop_{frame}']), None)
                if colmap_name:
                    row['colmap_image_name'] = colmap_name
                    scene = DATA / '02-位姿COLMAP/03-final_locked' / sample
                    colmap_rgb = scene / 'images' / colmap_name
                    row['colmap_image_dimensions'] = 'x'.join(str(value) for value in dimensions(colmap_rgb)) if colmap_rgb.exists() else 'UNKNOWN'
                    row['training_coordinate_note'] = 'COLMAP undistorted images may differ from raw; raw mask equality does not certify trained camera/input identity'
            except (ValueError, OSError, IndexError) as error:
                row.update(classification='NOT_VERIFIED', reason='CORRUPTED_OR_MISSING_RGB: ' + str(error))
            rows.append(row)
        print('mask comparison', config.parent.name, flush=True)
    write_csv('MASK_VERSION_COMPARISON.csv', rows)
    coverage, decisions = [], []
    for sample in samples:
        selected = [row for row in rows if row['sample_id'] == sample]
        valid = [row for row in selected if row['classification'] in ['EXACT_MATCH', 'SIMILAR_BUT_DIFFERENT', 'DIFFERENT']]
        old_frames = {path.stem.removeprefix('mask_') for path in (old_root / sample).glob('mask_*.png') if path.stem.removeprefix('mask_').isdigit()}
        matched_frames = {row['frame_id'] for row in valid}
        categories = Counter(row['classification'] for row in selected)
        row = {'sample_id': sample, 'old_mask_frames': len(old_frames), 'p6_version_frame_pairs': len(selected), 'verified_pairs': len(valid), 'unique_matched_frames': len(matched_frames), 'coverage_old_frames': len(matched_frames) / len(old_frames) if old_frames else 0, 'unmatched_old_frames': len(old_frames - matched_frames), 'exact_pair_count': categories['EXACT_MATCH'], 'similar_pair_count': categories['SIMILAR_BUT_DIFFERENT'], 'different_pair_count': categories['DIFFERENT'], 'not_verified_count': categories['NOT_VERIFIED'], 'missing_pair_count': categories['UNMATCHED'], 'p6_versions': ';'.join(sorted({entry['p6_run'] for entry in selected})) or 'UNKNOWN'}
        coverage.append(row)
        training_configs, commands, models, meshes, evaluations = model_assets(sample)
        decision = 'TARGETED_RERUN_REQUIRED' if valid and any(entry['classification'] != 'EXACT_MATCH' for entry in valid) and models else 'REUSE_AS_LEGACY_BASELINE' if models and training_configs else 'NOT_VERIFIED'
        decisions.append({'sample_id': sample, 'decision': decision, 'has_historical_model': bool(models), 'model_count': len(models), 'model_paths': ';'.join(relative(path) for path in models) or 'UNKNOWN', 'has_mesh': bool(meshes), 'mesh_paths': ';'.join(relative(path) for path in meshes) or 'UNKNOWN', 'training_command_paths': ';'.join(relative(path) for path in commands) or 'UNKNOWN', 'training_config_paths': ';'.join(relative(path) for path in training_configs) or 'UNKNOWN', 'mask_dir_trace': 'modern run configs use 02-sam_masks; old Windows cfg_args uses images_rgba, no explicit mask_dir', 'evaluation_paths': ';'.join(relative(path) for path in evaluations) or 'UNKNOWN', 'unique_p6_matched_frames': len(matched_frames), 'full_training_inputs_verified': False, 'direct_reuse_of_P6_R1': False, 'reason': 'Final P6+R1 masks absent; incomplete frame coverage and undistortion/version provenance prevent equivalence certification'})
    write_csv('MASK_MATCH_COVERAGE.csv', coverage)
    write_csv('RECONSTRUCTION_REUSE_DECISION.csv', decisions)
    valid = [row for row in rows if row['classification'] in ['EXACT_MATCH', 'SIMILAR_BUT_DIFFERENT', 'DIFFERENT']]
    matched_samples = sum(row['unique_matched_frames'] > 0 for row in coverage)
    unique_count = len({(row['sample_id'], row['frame_id']) for row in valid})
    exact = sum(row['classification'] == 'EXACT_MATCH' for row in valid)
    counts = Counter(row['classification'] for row in rows)
    report('MASK_INPUT_IDENTITY_AUDIT.md', f'''# 旧2DGS/P6输入身份审计

旧掩膜目录样本数 {len(samples)}；有身份验证的P6匹配样本 {matched_samples}/{len(samples)}；唯一帧 {unique_count}；
版本-帧对 {len(rows)}，有效 {len(valid)}，EXACT_MATCH {exact}。分布：{dict(counts)}。
19是SAM掩膜样本数量，不等于19个已证实训练完成的2DGS样本；BaiZhang不在该旧掩膜目录，单独UNMATCHED。
真实图像索引→原FFT RGB逐像素相同；old crop在mask内与RGB完全相同，核实mask/RGB配对，不按排序索引匹配。
两份掩膜必须同尺寸、同raw RGB坐标系才比较；没有任何缩放。
0.95只是SIMILAR_BUT_DIFFERENT描述阈值，不是3D等价阈值；IoU接近1也不能直接继承新方法结果。
图像尺寸、已注册COLMAP名和训练配置逐帧记录于CSV。COLMAP images是undistorted域，例如CaoMei1从2160×3840变2120×3800；
原掩膜比较不证明训练相机域一致。没有生成新P6+R1最终掩膜，本阶段的P6版本全部为旧运行。
Pot coverage只用已有split_v2真实pot标注，缺失写UNKNOWN。
''')
    report('RECONSTRUCTION_REUSE_REPORT.md', f'''# 重建继承结论

没有任何样本获得REUSE_DIRECTLY（P6+R1）。有P6匹配 {matched_samples}/19；EXACT_MATCH版本-帧对 {exact}/{len(valid)}。
逐样本决策见 `tables/RECONSTRUCTION_REUSE_DECISION.csv`；覆盖率见 `tables/MASK_MATCH_COVERAGE.csv`。
TARGETED_RERUN_REQUIRED是对新最终掩膜与已完成模型存在差异的有限样本建议，需Phase20C先产出新掩膜并再选受影响样本。
不得以此要求20序列全量重跑。旧模型仅能以其原输入/版本保留为legacy baseline，不能更名成R1结果。
即使个别帧EXACT_MATCH，也不能因局部覆盖或相同文件名判断整套训练输入/模型版本完全一致。
''')
    print('mask final', len(samples), matched_samples, unique_count, len(valid), exact, flush=True)


if __name__ == '__main__':
    main()
