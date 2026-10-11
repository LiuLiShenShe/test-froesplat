from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from common import DATA, ROOT, WORK, markdown_table, read_csv, relative, report, sha256, write_csv

PAPER = WORK / 'ForeSplat__Foreground_Object_2D_Gaussian_Splatting_via_Semantic_Mask_Guided_Task_Reformulation_for_Plant/ForeSplat.tex'
M2M3 = ROOT / '00-论文优化重构/m2m3实验/实验输出'
OUTPUT = DATA / '06-实验输出'
COLMAP_CHECK = ROOT / '08-Check'


def parse_paper():
    text = PAPER.read_text(encoding='utf-8')
    values = {}
    rows = {
        'COLMAP': (13.63, 0.8745, 0.1072, 599.5, 78.0),
        '3DGS_with_VFM': (30.17, 0.9587, 0.0386, 5413.5, 642.0),
        'Standard_2DGS': (29.58, 0.9574, 0.0487, 12913.7, 157.9),
        'ForeSplat': (31.09, 0.9711, 0.0365, 5044.5, 55.0),
    }
    for method, numbers in rows.items():
        values[method] = {'method': method, 'paper_psnr': numbers[0], 'paper_ssim': numbers[1], 'paper_lpips': numbers[2], 'paper_train_time_sec': numbers[3], 'paper_mesh_time_sec': numbers[4], 'paper_claim': 'averaged over all 20 RGB sequences', 'paper_source': relative(PAPER), 'paper_source_sha256': sha256(PAPER), 'paper_line_hint': 'Table reconstruction near lines 740-753'}
    return values


def json_time_keys(path):
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return []
    found = []
    def visit(value, prefix=''):
        if isinstance(value, dict):
            for key, item in value.items():
                current = f'{prefix}.{key}' if prefix else key
                if re.search(r'time|elapsed|duration|runtime', key, re.I):
                    found.append((current, item))
                visit(item, current)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f'{prefix}[{index}]')
    visit(payload)
    return found


def actual_sources():
    rows = []
    old_models = sorted((ROOT / '05-2DGS-new').glob('*/'))
    old_models = [path for path in old_models if (path / 'cfg_args').exists()]
    old_meshes = sorted((ROOT / '06-MESH-new').glob('*_mesh*.log'))
    colmap_dirs = sorted(path for path in (ROOT / '04-COLMAP').iterdir() if path.is_dir()) if (ROOT / '04-COLMAP').exists() else []
    check_metrics = list(COLMAP_CHECK.rglob('*.csv')) + list(COLMAP_CHECK.rglob('*.json'))
    for path in old_models:
        rows.append({'method': 'Standard_2DGS', 'sample_id': path.name, 'artifact_type': 'checkpoint_or_model_directory', 'artifact_path': relative(path), 'metric_path': relative(path / 'results.json') if (path / 'results.json').exists() else 'UNKNOWN', 'runtime_path': 'UNKNOWN', 'views_or_frames': 'UNKNOWN', 'status': 'PARTIAL', 'reusable_role': 'REUSE_AS_LEGACY_BASELINE', 'note': 'cfg_args points to historical model; training command/runtime and common evaluation split not fully recoverable'})
    for path in old_meshes:
        text = path.read_text(encoding='utf-8', errors='replace')
        time_hits = [line.strip() for line in text.splitlines() if re.search(r'time|elapsed|second|秒|duration', line, re.I)]
        rows.append({'method': 'ForeSplat_or_2DGS_mesh', 'sample_id': path.name.removesuffix('_mesh.log').removesuffix(' (1)'), 'artifact_type': 'mesh_log', 'artifact_path': relative(path), 'metric_path': 'UNKNOWN', 'runtime_path': relative(path), 'views_or_frames': 'UNKNOWN', 'status': 'PARTIAL', 'reusable_role': 'REUSE_AS_LEGACY_BASELINE', 'note': 'time fields found: ' + ('; '.join(time_hits[:3]) if time_hits else 'NONE')})
    for path in check_metrics:
        text = path.read_text(encoding='utf-8', errors='replace')
        if re.search(r'psnr|ssim|lpips', text, re.I):
            rows.append({'method': 'COLMAP_or_render_check', 'sample_id': path.parent.name, 'artifact_type': path.suffix[1:], 'artifact_path': relative(path), 'metric_path': relative(path), 'runtime_path': 'UNKNOWN', 'views_or_frames': 'UNKNOWN', 'status': 'PARTIAL', 'reusable_role': 'REUSE_AS_DIAGNOSTIC', 'note': 'check output, not proven to be the Table7 20-sequence aggregate'})
    for path in sorted((OUTPUT).glob('*/*/foreground_object_results.json')):
        sample = path.parts[-2]
        method = path.parts[-1]
        payload = json.loads(path.read_text())
        result = payload.get('ours_30000', {})
        rows.append({'method': method, 'sample_id': sample, 'artifact_type': 'foreground_object_results', 'artifact_path': relative(path), 'metric_path': relative(path), 'runtime_path': relative(path.parent / 'logs') if (path.parent / 'logs').exists() else 'UNKNOWN', 'views_or_frames': result.get('num_images', 'UNKNOWN'), 'psnr': result.get('PSNR_fg', 'UNKNOWN'), 'ssim': result.get('SSIM_fg', 'UNKNOWN'), 'lpips': result.get('LPIPS_fg_black_bg', 'UNKNOWN'), 'status': 'PARTIAL', 'reusable_role': 'REUSE_AS_LEGACY_BASELINE', 'note': 'per-sample foreground metric; method/config and GT domain require same-protocol review'})
    for summary in sorted((M2M3 / 'reports').glob('full_evaluation_summary_*.csv')):
        for row in read_csv(summary):
            rows.append({'method': row['method'], 'sample_id': row['scene'], 'artifact_type': 'M2M3_full_evaluation_row', 'artifact_path': relative(summary), 'metric_path': relative(summary), 'runtime_path': relative(M2M3 / 'logs'), 'views_or_frames': row['test_gt_count'], 'psnr': row['psnr'], 'ssim': row['ssim'], 'lpips': row['lpips'], 'train_time_sec': row['train_wall_elapsed_sec'], 'render_time_sec': row['render_elapsed_sec'], 'metrics_time_sec': row['metrics_elapsed_sec'], 'status': 'VERIFIED_AS_PILOT', 'reusable_role': 'REUSE_AS_M2M3_PILOT_ONLY', 'note': '3-scene pilot, not 20-sequence Table7 aggregate; duplicate run tags are kept separate'})
    return rows, {'old_2dgs_models': len(old_models), 'mesh_logs': len(old_meshes), 'colmap_check_files': len(check_metrics), 'm2m3_summary_files': len(list((M2M3 / 'reports').glob('full_evaluation_summary_*.csv')))}


def main():
    paper = parse_paper()
    actual, counts = actual_sources()
    fields = list(dict.fromkeys(key for row in actual for key in row))
    write_csv('TABLE7_RECONSTRUCTION_SOURCE.csv', actual, fields)
    reusable = [row for row in actual if row['reusable_role'] in ['REUSE_AS_LEGACY_BASELINE', 'REUSE_AS_M2M3_PILOT_ONLY', 'REUSE_AS_DIAGNOSTIC']]
    write_csv('TABLE7_REUSABLE_RESULTS.csv', reusable, fields)
    claim_rows = []
    for method, claim in paper.items():
        evidence = [row for row in actual if method.lower().replace('_', '') in row['method'].lower().replace('_', '') or (method == 'ForeSplat' and 'RAP' in row['method']) or (method == 'Standard_2DGS' and row['method'] == 'Standard_2DGS')]
        claim_rows.append({'method': method, **claim, 'matching_artifact_rows': len(evidence), 'matching_samples': len({row['sample_id'] for row in evidence}), 'status': 'UNVERIFIED' if len(evidence) < 20 else 'PARTIAL', 'reason': 'No source set proves same 20 sequence/sample split, view split, metric ROI, and runtime protocol'})
    rows_by_method = Counter(row['method'] for row in actual)
    report('TABLE7_PROVENANCE_AUDIT.md', f'''# Table 7 证据审计

论文 Table 7 声称 20 序列平均，但本地证据只显示：历史 2DGS `cfg_args` 模型 {counts['old_2dgs_models']} 个，Mesh 日志 {counts['mesh_logs']} 个，M2M3 full summary {counts['m2m3_summary_files']} 个文件且每个是 3 场景 pilot，不能拼成20序列同协议平均。
逐行实际产物在 `tables/TABLE7_RECONSTRUCTION_SOURCE.csv`；可作为 legacy/pilot 的行在 `TABLE7_REUSABLE_RESULTS.csv`。
论文四行的 PSNR/SSIM/LPIPS/训练时间/网格时间被解析并保存在 source CSV；没有任何一行获得20序列可复核证据。
M2M3 pilot 有真实训练、render、metrics elapsed 日志，故时间可作为 pilot 结果复用；不能替换论文四方法 Table7 时间。
COLMAP、3DGS+VFM、Standard 2DGS、ForeSplat 的 common view split、评价区域、失败样本处理和 wall-clock 证据未同时闭合。
''')
    report('TABLE7_UNVERIFIED_CLAIMS.md', '# Table 7 未验证声明\n\n' + markdown_table(claim_rows, ['method', 'paper_psnr', 'paper_ssim', 'paper_lpips', 'paper_train_time_sec', 'paper_mesh_time_sec', 'matching_artifact_rows', 'matching_samples', 'status', 'reason']) + '''

禁止沿用“20序列平均”作为 VERIFIED；论文四行旧数字最多标记 `PAPER_REPORTED_UNVERIFIED`。
训练时间需在统一机器、迭代数、数据划分和计时边界下重测；网格时间需说明是否含后处理、写盘和失败重试。
''')
    write_csv('TABLE7_CLAIM_STATUS.csv', claim_rows)
    print('table7 inventory complete', len(actual), flush=True)


if __name__ == '__main__':
    main()
