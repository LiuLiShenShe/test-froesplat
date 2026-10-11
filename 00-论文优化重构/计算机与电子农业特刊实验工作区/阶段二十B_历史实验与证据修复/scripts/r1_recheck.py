from __future__ import annotations

from collections import defaultdict
import json
import math

import numpy as np

from common import P17A, P17C, S20, V00, load_module, mask, read_csv, relative, report, sha256, write_csv


def recompute(row, raw_mask, config):
    if not raw_mask.any():
        return {'base_total': 0.0, 'risk_penalty': 0.0, 'total_score': 0.0, 'primary_score': 0.0, 'tie_break_score': 0.0}
    quality = sum(float(row[f'q_{name}']) for name in ['area', 'comp', 'edge', 'temp', 'contrast'])
    quality += 0.5 / (1 + math.exp(-6 * (float(row['sam_score']) - 0.5)))
    base = quality / 5.5
    vertical = 1 - int(np.where(raw_mask.any(axis=1))[0].min()) / raw_mask.shape[0]
    penalty = 0.25 if vertical < config['vertical_coverage_min_ratio'] else 0.0
    return {'base_total': base, 'risk_penalty': penalty, 'total_score': base - penalty, 'primary_score': base - float(row['q_contrast']) / 5.5 - penalty, 'tie_break_score': 0.0, 'vertical_coverage': vertical}


def main():
    policy = load_module(S20 / '脚本/ranking_policy.py', 'phase20b_ranking_policy')
    offline = load_module(P17C / '脚本/phase17c_common.py', 'phase20b_offline_policy')
    sources = {'DEV10': P17A / '02_trackA_variants/P00_Control', 'Easy21': V00}
    configs = {name: json.loads((path / '参数.json').read_text()) for name, path in sources.items()}
    archives = {name: read_csv(path / '提示词评分.csv') for name, path in sources.items()}
    results = []
    for original in read_csv(P17C / 'Phase17C_ranking_benchmark.csv'):
        row = dict(original)
        source = sources[row['source']]
        stem = f"{row['sample']}_{row['frame']}"
        raw_path = source / '候选掩膜/raw_instance_P6' / f"mask_{stem}_{int(row['instance_id']):02d}.png"
        raw = mask(raw_path)
        actual = recompute(row, raw, configs[row['source']])
        production_primary, production_tie = policy.primary_and_tiebreak(actual['base_total'], float(row['q_contrast']), float(row['q_area']), 1.0, 5.5, 0.0, actual['risk_penalty'], 'r1')
        numerical_row = {key: float(value) if key in ['sam_score', 'total_score'] or key.startswith('q_') else value for key, value in row.items()}
        archive_matches = [entry for entry in archives[row['source']] if entry['图像'].rsplit('.', 1)[0] == stem and abs(float(entry['面积比例']) - float(row['area_ratio'])) < 1e-8]
        origin_match = len(archive_matches) == 1 and abs(float(archive_matches[0]['总分']) - float(row['total_score'])) < 1e-12
        contrast_match = origin_match and float(archive_matches[0]['前背景对比得分']) == float(row['q_contrast'])
        archival_primary = offline.primary_score(numerical_row, 'R1', offline.compute_penalty(numerical_row))
        error_total = abs(actual['total_score'] - float(row['total_score']))
        error_primary = abs(production_primary - actual['primary_score'])
        error_archive = abs(archival_primary - actual['primary_score'])
        valid = origin_match and contrast_match and max(error_total, error_primary, error_archive) < 1e-6 and production_tie == 0
        result = {'source': row['source'], 'sample_id': row['sample'], 'frame_id': row['frame'], 'candidate_id': row['candidate_id'], 'instance_id': row['instance_id'], 'q_contrast': row['q_contrast'], **actual, 'archived_total_score': row['total_score'], 'production_primary_score': production_primary, 'offline_primary_score': archival_primary, 'error_total': error_total, 'error_primary': error_primary, 'error_offline_primary': error_archive, 'q_contrast_source_match': contrast_match, 'q_temp': row['q_temp'], 'risk_penalty_source': 'raw mask vertical coverage, NOT inferred from total', 'candidate_path': relative(raw_path), 'candidate_sha256': sha256(raw_path), 'config_path': relative(source / '参数.json'), 'status': 'PASS' if valid else 'FAIL'}
        results.append(result)
    write_csv('R1_FORMULA_RECHECK.csv', results)
    passed = sum(row['status'] == 'PASS' for row in results)
    penalized = [row for row in results if row['risk_penalty'] != 0]
    body = f'''# R1 182 条候选最终核查

{passed}/{len(results)} {'PASS' if passed == 182 and len(results) == 182 else 'FAIL'}。
风险惩罚独立从原始实例掩膜纵向覆盖重算，不通过 base−archived_total 反推，避免循环验证。
含惩罚候选数：{len(penalized)}。base_total 与 total_score 最大差 {max(row['risk_penalty'] for row in results):.8f}。
total_score 最大误差 {max(row['error_total'] for row in results):.12g}；primary 最大误差 {max(row['error_primary'] for row in results):.12g}；离线 primary 最大误差 {max(row['error_offline_primary'] for row in results):.12g}。
冻结分母5.5，weights=1,1,1,1,1,0.5；semantic_gate关闭，semantic_total=0；无track_mask实例，0.15分支不执行。
R1=base_total−q_contrast/5.5−risk_penalty，tie_break=0；q_temp未配准为0.5（CSV逐条保存）。
总分与原始提示词评分按图像身份+面积验证唯一来源；raw instance ID以文件名和候选明细映射，不靠目录排序。
数值容差1e-6处理归档SAM分数六位小数，不更改归档数据、参数或公式。
Phase20A所述0.25差值是比较未扣风险项的base_total与已扣项的total_score，见下表。

'''
    body += '| candidate_id | base_total | risk_penalty | total_score | error_total |\n|---|---|---|---|---|\n'
    for row in penalized:
        body += f"| {row['candidate_id']} | {row['base_total']:.12f} | {row['risk_penalty']} | {row['total_score']:.12f} | {row['error_total']:.12g} |\n"
    body += '\n逐条判定与真实路径、哈希：`tables/R1_FORMULA_RECHECK.csv`；未运行SAM3、未修改ranking_policy.py。\n'
    report('R1_182_CANDIDATE_RECHECK.md', body)
    print(f'R1 {passed}/{len(results)}', flush=True)


if __name__ == '__main__':
    main()
