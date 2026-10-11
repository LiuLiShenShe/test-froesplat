from __future__ import annotations

import json
from pathlib import Path

from common import DATA, P12, ROOT, S20, STAGE, dimensions, read_csv, relative, report, sha256, write_csv

SAMPLES = ['CaoMei1', 'ChangShouHua2', 'DouBanLv1', 'KongQueZhuYu', 'XianKeLai1']
GROUPS = {
    'V0': {'ranking_policy': 'base', 'refinement': 'OFF'},
    'V1': {'ranking_policy': 'r1', 'refinement': 'OFF'},
    'V2': {'ranking_policy': 'base', 'refinement': 'ON'},
    'V3': {'ranking_policy': 'r1', 'refinement': 'ON'},
}


def frames_for(sample):
    source = P12 / 'P6_raw_baseline/图像索引.csv'
    return [row['图像名'].replace('.jpg', '') for row in read_csv(source) if row['图像名'].startswith(sample + '_')]


def config(group, sample):
    info = GROUPS[group]
    base = {
        'phase': '20C_prepared_only',
        'group': group,
        'refinement': info['refinement'],
        'sample_id': sample,
        'path_mode': 'workspace_relative',
        'input_dir': relative(STAGE / f'phase20c_inputs/{sample}'),
        'output_dir': relative(STAGE / f'phase20c_pending/{group}/{sample}'),
        'sam3_repo': relative(ROOT / '第三方源码/SAM3-latest'),
        'sam3_checkpoint': relative(ROOT / 'sam3/sam3.pt'),
        'device': 'cuda:0',
        'prompt_list': 'P6',
        'default_prompt_id': 'P6',
        'candidate_mode': 'per_instance',
        'use_prompt_ensemble': False,
        'prompt_selection_mode': 'single',
        'ranking_policy': info['ranking_policy'],
        'score_weights': 'area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5',
        'use_temporal_alignment': False,
        'reuse_existing_candidates': True,
        'candidate_source_dir': relative(STAGE / f'phase20c_candidate_alias/{sample}'),
        'use_spnp_refinement': info['refinement'] == 'ON',
        'spnp_backend': 'sam3_if_supported' if info['refinement'] == 'ON' else 'none',
        'spnp_positive_points': 5,
        'spnp_negative_points': 8,
        'spnp_ring_radius': 15,
        'spnp_min_refined_iou': 0.65,
        'use_residual_repair': info['refinement'] == 'ON',
        'opening_kernel': 3,
        'thin_repair_mode': 'adjacent' if info['refinement'] == 'ON' else 'none',
        'use_corrective_geometry': info['refinement'] == 'ON',
        'corrective_geometry_backend': 'colmap_tracks' if info['refinement'] == 'ON' else 'none',
        'colmap_dir': relative(DATA / f'02-位姿COLMAP/03-final_locked/{sample}') if info['refinement'] == 'ON' else None,
        'geometry_positive_min_points': 12,
        'geometry_negative_min_area_ratio': 0.012,
        'geometry_correct_max_delta_ratio': 0.08,
        'geometry_correct_min_iou': 0.75,
        'geometry_track_dilation': 19,
        'geometry_unsupported_dilation': 29,
        'limit': 0,
        'gpu_inference_started': False,
        'config_provenance': relative(P12 / 'P6_raw_baseline/参数.json'),
    }
    return base


def main():
    config_dir = STAGE / 'reports/configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows, matrix_rows, commands = [], [], []
    script = relative(S20 / '脚本/生成RAP-FSAM3掩膜.py')
    for group in GROUPS:
        for sample in SAMPLES:
            cfg = config(group, sample)
            input_alias = STAGE / 'phase20c_inputs' / sample
            candidate_alias = STAGE / 'phase20c_candidate_alias' / sample / '候选掩膜/raw_instance_P6'
            input_alias.mkdir(parents=True, exist_ok=True)
            candidate_alias.mkdir(parents=True, exist_ok=True)
            frames = frames_for(sample)
            for stem in frames:
                source_image = P12 / 'frame_inputs' / f'{stem}.jpg'
                target_image = input_alias / f'{stem.rsplit("_", 1)[1]}.jpg'
                if not target_image.exists():
                    target_image.symlink_to(source_image)
            original_candidate_root = P12 / 'P6_raw_baseline/候选掩膜/raw_instance_P6'
            for source_candidate in sorted(original_candidate_root.glob(f'mask_{sample}_*.png')):
                suffix = source_candidate.name.removeprefix(f'mask_{sample}_')
                target_candidate = candidate_alias / f'mask_{suffix}'
                if not target_candidate.exists():
                    target_candidate.symlink_to(source_candidate)
            (config_dir / f'{group}_{sample}.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            input_dir = ROOT / cfg['input_dir']
            candidate_root = candidate_alias
            input_files = sorted(input_dir.glob('*.jpg')) if input_dir.exists() else []
            candidates = sorted(candidate_root.glob('mask_*.png'))
            stems = {path.stem.rsplit('_', 1)[0].removeprefix('mask_') for path in candidates}
            local_frame_stems = [stem.rsplit('_', 1)[1] for stem in frames]
            missing = sorted(set(local_frame_stems) - stems)
            colmap = ROOT / f"{cfg['colmap_dir']}" if cfg['colmap_dir'] else DATA / f'02-位姿COLMAP/03-final_locked/{sample}'
            checks = {
                'input_dir_exists': input_dir.is_dir(),
                'input_frame_count': len(input_files),
                'manifest_frame_count': len(local_frame_stems),
                'candidate_source_exists': candidate_root.is_dir(),
                'candidate_file_count': len(candidates),
                'candidate_frame_count': len(stems),
                'missing_candidate_frames': ';'.join(missing) or 'NONE',
                'colmap_dir_exists': colmap.is_dir(),
                'colmap_sparse_exists': (colmap / 'sparse/0').is_dir(),
                'sam3_repo_exists': (ROOT / '第三方源码/SAM3-latest').is_dir(),
                'sam3_checkpoint_exists': (ROOT / 'sam3/sam3.pt').is_file(),
            }
            status = 'READY_FOR_PHASE20C' if all(value is True for key, value in checks.items() if key.endswith('_exists')) and not missing else 'BLOCKED'
            manifest_rows.append({'group': group, 'sample_id': sample, **checks, 'formal_candidate_reuse': 'YES', 'candidate_generation_gpu_required': 'NO', 'refinement_gpu_required': 'YES' if cfg['use_spnp_refinement'] else 'NO', 'planned_config': relative(config_dir / f'{group}_{sample}.json'), 'planned_output': cfg['output_dir'], 'status': status})
            matrix_rows.append({'group': group, 'ranking_policy': cfg['ranking_policy'], 'refinement': cfg['refinement'], 'prompt_list': cfg['prompt_list'], 'default_prompt_id': cfg['default_prompt_id'], 'candidate_mode': cfg['candidate_mode'], 'use_prompt_ensemble': cfg['use_prompt_ensemble'], 'prompt_selection_mode': cfg['prompt_selection_mode'], 'use_spnp_refinement': cfg['use_spnp_refinement'], 'spnp_backend': cfg['spnp_backend'], 'use_residual_repair': cfg['use_residual_repair'], 'thin_repair_mode': cfg['thin_repair_mode'], 'use_corrective_geometry': cfg['use_corrective_geometry'], 'corrective_geometry_backend': cfg['corrective_geometry_backend'], 'candidate_cache': cfg['candidate_source_dir'], 'same_non_factor_parameters': 'YES'})
            flags = [f"--input_dir ${{REPO_ROOT}}/{cfg['input_dir']}", f"--output_dir ${{REPO_ROOT}}/{cfg['output_dir']}", f"--sam3_repo ${{REPO_ROOT}}/{cfg['sam3_repo']}", f"--sam3_checkpoint ${{REPO_ROOT}}/{cfg['sam3_checkpoint']}", '--device cuda:0', '--candidate_mode per_instance', '--prompt_list P6', '--default_prompt_id P6', '--prompt_selection_mode single', '--score_weights area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5', f"--ranking_policy {cfg['ranking_policy']}", '--reuse_existing_candidates', f"--candidate_source_dir ${{REPO_ROOT}}/{cfg['candidate_source_dir']}", '--force']
            if cfg['refinement'] == 'ON':
                flags += ['--use_spnp_refinement', '--spnp_backend sam3_if_supported', '--use_residual_repair', '--thin_repair_mode adjacent', '--use_corrective_geometry', '--corrective_geometry_backend colmap_tracks', f"--colmap_dir ${{REPO_ROOT}}/{cfg['colmap_dir']}"]
            commands.append(f"{group}/{sample}: python ${{REPO_ROOT}}/{script} " + ' '.join(flags))
    write_csv('V0_V3_CONFIG_MATRIX.csv', matrix_rows)
    write_csv('V0_V3_INPUT_MANIFEST.csv', manifest_rows)
    report('V0_V3_COMMANDS.md', '# V0–V3 未执行命令清单\n\n以下命令只准备，不执行；`${REPO_ROOT}` 是运行者显式设置的仓库根目录。\n\n' + '\n'.join(f'```bash\n{command}\n```' for command in commands) + '''

V2/V3 打开 `spnp_backend=sam3_if_supported`，即使复用候选也会按源码加载 SAM3；V0/V1 复用候选时不需生成阶段 SAM3 推理。所有组显式设置 P6、per_instance、single、无 prompt ensemble 和相同权重。
''')
    ready = sum(row['status'] == 'READY_FOR_PHASE20C' for row in manifest_rows)
    report('V0_V3_PREFLIGHT_REPORT.md', f'''# V0–V3 预检

配置数：{len(matrix_rows)}（4组×5样本）；资产行 READY：{ready}/{len(manifest_rows)}。
候选缓存来自真实 Phase 12 `P6_raw_baseline/候选掩膜/raw_instance_P6`，不会覆盖历史运行；候选缓存不能证明 R1 已正式执行，只能减少候选生成成本。
输入、候选帧覆盖、COLMAP sparse、SAM3 repo/checkpoint 均逐样本记录在 `tables/V0_V3_INPUT_MANIFEST.csv`。
V0/V1 的候选选择/离线可在无GPU环境验证；若按正式生产流程执行，仍应保留运行日志和参数 JSON。
V2/V3 的 SPNP backend 会加载 SAM3，需要 GPU；本阶段未启动任何生成或细化。
若 manifest status 为 BLOCKED，不能进入 Phase20C；当前配置只表达可执行条件，不代表实验结果已经存在。
''')
    report('PHASE20C_EXECUTION_PLAN.md', '''# Phase 20C 执行计划（只准备）

1. 在 V0–V3 配置 JSON 和命令锁定后，先运行 CPU dry-run/参数回显，不运行 SAM3。
2. 固定同一候选缓存、P6 prompt、输入帧、COLMAP 资产和输出命名；每组保存参数 JSON、运行日志、候选来源哈希。
3. 先执行 V0/V1，比较 base 与 R1 的 selected mask、primary score、oracle regret、空掩膜和候选选择变化。
4. 再执行 V2/V3，记录 SPNP、residual repair、corrective geometry 是否实际接受以及每阶段 mask 哈希。
5. 用 `potted_clean` GT 评价 DEV/安全集，不把 TEST32 无GT帧计算成准确率；报告 paired frame、sample_equal_mean、median、SD 和有效数。
6. 只有 V0–V3 完成且输入/版本一致后，才决定最小 2DGS/M2M3-GO GPU 重跑样本；不默认全量20序列。

停止条件：任何资产缺失、mask坐标域未闭合、refinement 参数未写入、运行中途失败或输出混入历史目录时，标记 BLOCKED 而非填数字。
''')
    print('v0-v3 prepared', ready, '/', len(manifest_rows), flush=True)


if __name__ == '__main__':
    main()
