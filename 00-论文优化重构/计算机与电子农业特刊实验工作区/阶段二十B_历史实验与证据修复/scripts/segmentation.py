from __future__ import annotations

from collections import defaultdict
import json

import cv2
import numpy as np

from common import ALG, BENCH, DATA, P12, P16, ROOT, S20, WORK, dimensions, git_origin, load_module, markdown_table, mask, read_csv, relative, report, rgb_equal, sha256, write_csv

METRICS = ['f1', 'iou_fg', 'miou', 'hd95_px', 'boundary_f1', 'leakage_energy']
LEGACY_PATH = WORK / '03-扩展实验后续执行/实验二_四样本提示词鲁棒性汇总.py'
LEGACY = load_module(LEGACY_PATH, 'phase20b_legacy_metrics')
PROMPTS = {'P1': 'P1_绿色植物', 'P2': 'P2_整株去花盆', 'P3': 'P3_叶和茎', 'P4': 'P4_作物幼苗', 'P5': 'P5_去背景植物体'}


def evaluate_arrays(ground_truth, prediction):
    if ground_truth.shape != prediction.shape:
        raise ValueError('Dimension mismatch; resampling is forbidden')
    counts = LEGACY.confusion(ground_truth, prediction)
    result = LEGACY.pixel_metrics(*counts)
    result.update(dict(zip(['tp', 'fp', 'fn', 'tn'], counts)))
    result['hd95_px'] = LEGACY.hd95(ground_truth, prediction)
    result['boundary_f1'] = LEGACY.boundary_f1(ground_truth, prediction, 3)
    result['leakage_energy'] = counts[1] / ground_truth.size
    return result


def verify_gt():
    processor = load_module(P12 / 'GT_QA/处理GT_v2.py', 'phase20b_gt_readonly')
    audits = []
    for path in sorted((P12 / 'GT_potted_clean').glob('*/mask_potted_clean_*.png')):
        sample = path.parent.name
        frame = path.stem.rsplit('_', 1)[1]
        annotation = ROOT / '03-GT-区分' / sample / f'{frame}.json'
        image = annotation.with_suffix('.jpg')
        stored = mask(path)
        generated = processor.process_frame(annotation, image)['masks']['potted_clean']
        split = S20 / 'GT口径拆分审计/gt_masks_split_v2' / sample
        potted = mask(split / f'mask_potted_plant_{frame}.png')
        cube = mask(split / f'mask_blue_cube_{frame}.png')
        data = json.loads(annotation.read_text())
        valid = np.array_equal(stored, generated) and np.array_equal(stored, potted & ~cube)
        valid = valid and stored.shape == (data['imageHeight'], data['imageWidth']) and dimensions(image) == (stored.shape[1], stored.shape[0])
        audits.append({'sample_id': sample, 'frame_id': frame, 'gt_path': relative(path), 'labelme_path': relative(annotation), 'potted_path': relative(split / f'mask_potted_plant_{frame}.png'), 'cube_path': relative(split / f'mask_blue_cube_{frame}.png'), 'formula_equal': valid, 'status': 'VERIFIED' if valid else 'NOT VERIFIED', 'sha256': sha256(path), 'source_commit': git_origin(annotation)})
    challenge_processor = load_module(P16 / '脚本/process_gt_challenge.py', 'phase20b_challenge_gt')
    for path in sorted((P16 / 'GT_potted_clean_challenge').glob('*/mask_potted_clean_*.png')):
        sample = path.parent.name
        frame = path.stem.rsplit('_', 1)[1]
        annotation = ROOT / '03-GT-区分_challenge' / sample / f'{frame}.json'
        data = json.loads(annotation.read_text())
        stored = mask(path)
        combined = np.zeros(stored.shape, dtype=np.uint8)
        for shape in data['shapes']:
            if shape['label'].strip().lower().replace('_', ' ') == 'potted plant':
                points = np.rint(np.array(shape['points'], dtype=np.float32)).astype(np.int32)
                points[:, 0] = np.clip(points[:, 0], 0, stored.shape[1] - 1)
                points[:, 1] = np.clip(points[:, 1], 0, stored.shape[0] - 1)
                cv2.fillPoly(combined, [points], 255)
        valid = np.array_equal(stored, combined > 0)
        audits.append({'sample_id': sample, 'frame_id': frame, 'gt_path': relative(path), 'labelme_path': relative(annotation), 'formula_equal': valid, 'status': 'VERIFIED' if valid else 'NOT VERIFIED', 'sha256': sha256(path), 'source_commit': git_origin(annotation), 'note': 'potted plant only; no separate pot/cube annotation; cannot compute pot coverage'})
    write_csv('GT_DEFINITION_AUDIT.csv', audits)
    return audits


def prediction_sources(record):
    sample, frame = record['sample'], record['frame']
    stem = f'{sample}_{frame}'
    rap = BENCH / 'rap_runs/RAP-FSAM3-v2' / sample
    sources = []
    for method in ['CLIPSeg_potted', 'GroundedSAM1_potted', 'SAM3_P2', 'RAP-FSAM3-v2']:
        configuration = rap / '参数.json' if method == 'RAP-FSAM3-v2' else BENCH / 'rap_runs/SAM3_P2' / sample / '参数.json' if method == 'SAM3_P2' else BENCH / 'README.md'
        sources.append(('3', method, BENCH / 'method_masks' / method / f'mask_{stem}.png', configuration, 'LEGACY_BASELINE'))
    for prompt, directory in PROMPTS.items():
        sources.append(('4', prompt, rap / '候选掩膜' / directory / f'mask_{frame}.png', rap / '参数.json', 'LEGACY_PROMPT_CANDIDATE'))
    mappings = {'Fixed_prompt_P2': ('候选掩膜/P2_整株去花盆', 'PIPELINE_OUTPUT'), 'Geometry_only_selector': ('REPLAY', 'OFFLINE_REPLAY'), 'Semantic_gated_selected': ('选择后掩膜', 'PIPELINE_OUTPUT'), 'Without_GCFC': ('残差修复掩膜', 'INTERMEDIATE_OUTPUT_NOT_INDEPENDENT_RUN'), 'RAP_FSAM3_full': ('最终掩膜', 'PIPELINE_OUTPUT')}
    for method, (directory, status) in mappings.items():
        path = BENCH / 'e3_multisample_ablation_masks/A1_old_replayed' / f'mask_{stem}.png' if directory == 'REPLAY' else rap / directory / f'mask_{frame}.png'
        sources.append(('5', method, path, rap / '参数.json', status))
    return sources


def aggregate(rows):
    valid = [row for row in rows if row['status'] == 'VERIFIED']
    grouped = defaultdict(list)
    for row in valid:
        grouped[row['sample_id']].append(row)
    result = []
    for sample, group in sorted(grouped.items()):
        result.append({'aggregation': 'per_sample_mean', 'sample_id': sample, 'valid_frames': len(group), 'valid_samples': 1, **{metric: float(np.mean([row[metric] for row in group])) for metric in METRICS}})
    sample_means = result.copy()
    reductions = [('per_frame_mean', np.mean), ('per_frame_median', np.median), ('per_frame_std_population_ddof0', np.std)]
    for label, function in reductions:
        result.append({'aggregation': label, 'sample_id': 'ALL', 'valid_frames': len(valid), 'valid_samples': len(grouped), **{metric: float(function([row[metric] for row in valid])) if valid else 'UNKNOWN' for metric in METRICS}})
    for label, function in [('sample_equal_mean', np.mean), ('per_sample_median', np.median), ('per_sample_std_population_ddof0', np.std)]:
        result.append({'aggregation': label, 'sample_id': 'ALL', 'valid_frames': len(valid), 'valid_samples': len(grouped), **{metric: float(function([row[metric] for row in sample_means])) if valid else 'UNKNOWN' for metric in METRICS}})
    return result


def main():
    audits = verify_gt()
    verified_gt = {(row['sample_id'], row['frame_id']) for row in audits if row['status'] == 'VERIFIED'}
    records = read_csv(BENCH / 'manifest.csv')
    frame_rows, provenance = [], []
    for record in records:
        sample, frame = record['sample'], record['frame']
        ground_path = P12 / 'GT_potted_clean' / sample / f'mask_potted_clean_{frame}.png'
        annotated_rgb = ROOT / '03-GT-区分' / sample / f'{frame}.jpg'
        bench_rgb = BENCH / 'selected_frames' / f'{sample}_{frame}.jpg'
        rgb_identity = rgb_equal(annotated_rgb, bench_rgb)
        ground = mask(ground_path)
        cache = {}
        for table, method, path, configuration, kind in prediction_sources(record):
            row = {'table': table, 'method': method, 'sample_id': sample, 'frame_id': frame, 'gt_definition': 'potted_plant AND NOT blue_cube', 'status': 'VERIFIED', 'evidence_kind': kind, 'prediction_path': relative(path), 'gt_path': relative(ground_path), 'configuration_path': relative(configuration), 'rgb_identity_verified': rgb_identity, 'source_commit': git_origin(path), 'metric_implementation': relative(LEGACY_PATH), 'boundary_tolerance_px': 3}
            try:
                prediction = mask(path)
                row.update({'width': prediction.shape[1], 'height': prediction.shape[0], 'prediction_sha256': sha256(path)})
                if not rgb_identity or (sample, frame) not in verified_gt or prediction.shape != ground.shape:
                    raise ValueError('RGB identity, GT verification or dimensions failed')
                key = row['prediction_sha256']
                if key not in cache:
                    cache[key] = evaluate_arrays(ground, prediction)
                row.update(cache[key])
            except (OSError, ValueError) as error:
                row['status'] = 'UNVERIFIED'
                row['reason'] = str(error)
            frame_rows.append(row)
            provenance.append({key: value for key, value in row.items() if key not in METRICS + ['tp', 'fp', 'fn', 'tn', 'precision', 'recall']})
        print('segmentation', sample, frame, flush=True)
    write_csv('SEGMENTATION_PER_FRAME_METRICS.csv', frame_rows)
    write_csv('SEGMENTATION_SOURCE_PROVENANCE.csv', provenance)
    summary_rows = []
    for table in ['3', '4', '5']:
        grouped = defaultdict(list)
        for row in frame_rows:
            if row['table'] == table:
                grouped[row['method']].append(row)
        results = []
        for method, group in sorted(grouped.items()):
            for row in aggregate(group):
                row.update({'table': table, 'method': method, 'gt_definition': 'potted_clean', 'evidence_kind': group[0]['evidence_kind'], 'source_index': 'tables/SEGMENTATION_SOURCE_PROVENANCE.csv', 'source_commit': group[0]['source_commit'], 'status': 'VERIFIED' if all(entry['status'] == 'VERIFIED' for entry in group) else 'PARTIAL'})
                results.append(row)
                if row['aggregation'] == 'sample_equal_mean':
                    summary_rows.append(row)
        write_csv(f'TABLE{table}_RECALCULATED.csv', results)
    passed = sum(row['status'] == 'VERIFIED' for row in audits)
    report('GT_DEFINITION_AUDIT.md', f'''# GT 定义审计

验证现有 GT：{passed}/{len(audits)}。不写入或重新保存任何 GT。
Easy21 逐像素核对 Phase12 clean、split_v2 的 potted∧¬cube、真实 LabelMe 多边形（只在内存核对）。
Challenge10 核对现有含盆标注；无拆分 pot，因此盆区指标不能由该集恢复。
逐帧来源、哈希和提交见 `tables/GT_DEFINITION_AUDIT.csv`。
原 E1 manifest 指向早期 union GT；不得用当前脚本注释倒推历史执行。
Table3–5 统一到同一 Easy20：4 序列×5 帧；不是20序列，也不是独立测试集。
v1 启发式 GT 不参与任何统计。不缩放 GT 或预测。
''')
    report('SEGMENTATION_METRIC_SPEC.md', '''# 分割指标规范

主统计口径为 sample_equal_mean：先对每个序列的有效帧指标取算术平均，再对序列均值等权平均。
Per-frame mean、per-sample mean、median、population SD(ddof=0) 分列为独立 aggregation 行。
4序列帧数相同，因此全有效时 sample_equal_mean=per_frame_mean；不把 pooled confusion F1 当帧均值。

F1=2TP/(2TP+FP+FN)；foreground IoU=TP/(TP+FP+FN)；mIoU=(foreground IoU+background IoU)/2。
分母为0时值为0，保持旧 safe_div 约定。
边界=mask减3×3方形腐蚀（OpenCV默认边界行为）；Boundary-F1 对双方边界用半径3 px、7×7椭圆膨胀判匹配，再取 precision/recall 调和平均。
双方空边界 BF1=1，单方空 BF1=0。
HD95 合并双向边界距离后取95百分位；cv2.distanceTransform(DIST_L2,3)，属于3×3近似欧氏距离，不换精确EDT。
单位为已有标注坐标系的原图像素；双方空边界0，单方空取图像对角线。
Leakage energy=FP/(H×W)，是旧脚本的掩膜泄漏比例，不是RGB辐射能量；新文稿应据实命名。
不与三维渲染脚本的RGB leakage energy混合。

权威旧实现：`03-扩展实验后续执行/实验二_四样本提示词鲁棒性汇总.py` 的 pixel_metrics/boundary_f1/hd95；
E1旧实现定义相同，逐函数回归测试验证。使用原函数，不执行其 main。
''')
    display = [{**row, **{metric: f'{row[metric]:.8f}' if isinstance(row[metric], float) else row[metric] for metric in METRICS}} for row in summary_rows]
    report('TABLE_3_4_5_RECALCULATION_REPORT.md', '# Table 3/4/5 纠正报告\n\n主统计：sample_equal_mean；GT=potted_clean；20帧/4序列；所有数值由真实预测重算。\n\n' + markdown_table(display, ['table', 'method', 'valid_frames', 'valid_samples'] + METRICS) + '''

Table5 Without_GCFC 使用本次同运行的残差修复中间掩膜（GCFC输入），只可作为中间阶段比较，不声称重新运行独立消融。
Geometry-only selector 为 OFFLINE_REPLAY，不是实际 pipeline 运行；后续测试核对原候选来源。
原稿 BF1=0.4543 与 Leakage=0.001935 无可追溯预测绑定，禁止沿用。
Table3 full F1=0.9706/mIoU=0.9641 的旧汇总混有 pooled confusion 与逐帧指标，不能作为统一GT结果沿用。
Table4 P2真实文本是 entire plant excluding pot，不是 entire potted plant；本次仅评价真实归档候选，不更改提示词。
本次数字是旧方法统一GT后的历史基线，不是P6+R1结果。完整聚合、逐帧配置与哈希见CSV。
''')


if __name__ == '__main__':
    main()
