"""CPU-only human-readable SfM run report; no network or model imports."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def write_run_report(output, metadata, status, failure=None):
    outcome = '完成' if status['status'] == 'complete' else '失败；不代表完成全量注册'
    audit_relative = os.path.relpath(ROOT / 'docs' / 'SFM_SOURCE_AUDIT.md', output)
    report = [f'# {metadata["method"]} RUN REPORT', '',
              f'状态：{outcome}。实际输入 {metadata["input_views"]} / {metadata["available_views"]} 个可用视图；'
              f'全量标志 {metadata["full_sequence"]}。物理 GPU1。', '',
              f'代码 AMB3R commit：`{metadata["amb3r_git_commit"]}`。',
              f'权重 SHA256：`{metadata.get("weight_sha256", "未完成权重加载")}`。', '',
              f'长边 {metadata["long_edge"]}，padding 对齐 {metadata["padding_alignment"]}；'
              '完整视野，RGB/Mask 使用相同几何变换，Mask 最近邻，未腐蚀、未曝光处理。', '',
              '执行官方聚类、初始化、coarse registration、keyframe BFS 与 non-keyframe refinement。'
              '仅增加观测与接口 shape bridge，没有修改网络或新建点云融合算法。', '',
              f'官方 allow_unmapped_frames 默认 false，本次 {metadata["config_changes"]["allow_unmapped_frames"]}。', '']
    if status['status'] == 'complete':
        report.extend([f'成功注册 {status["mapped_frames"]}；未映射 {status["unmapped_frames"]}。'
                       '每帧状态、原始输入编号与原因见 registration_status.json 和 unmapped_frames.json。', '',
                       f'pipeline 总时间 {metadata["inference_seconds"]:.3f} s；'
                       f'局部模型推理合计 {metadata["local_model_seconds"]:.3f} s；'
                       f'峰值 allocated VRAM {metadata["peak_vram_gib"]:.3f} GiB。', ''])
        if 'fast3r_git_commit' in metadata:
            report.extend([f'Fast3R commit：`{metadata["fast3r_git_commit"]}`。', ''])
        if 'peak_reserved_vram_gib' in metadata:
            report.extend([f'峰值 reserved VRAM {metadata["peak_reserved_vram_gib"]:.3f} GiB；'
                           f'局部调用 {metadata.get("local_prediction_calls")} 次。', ''])
        if 'official_benchmark_early_stop_calls' in metadata:
            report.extend([f'官方 benchmark 提前返回 None：{metadata["official_benchmark_early_stop_calls"]} 次；'
                           f'原生 Backend confidence 下降后退回 Frontend：{metadata.get("backend_frontend_fallback_count", 0)} 次。'
                           '这些是原始模型逻辑，None 调用没有伪造返回几何。', ''])
        if 'observed_pnp_status_counts_over_all_local_calls' in metadata:
            report.extend([f'全部已返回局部调用的实际 PnP observer 计数：'
                           f'{metadata["observed_pnp_status_counts_over_all_local_calls"]}。'
                           'not_called/started 为观测不完整；solver success 不代表几何精度。', ''])
        if 'bundle_export_seconds' in metadata:
            report.extend([f'最终 CPU bundle 导出 {metadata["bundle_export_seconds"]:.3f} s。', ''])
        else:
            report.extend(['此旧 runner 未计时最终 CPU bundle 导出与完整端到端耗时；未用文件时间推算缺失值。', ''])
        report.extend([f'局部原始预测序列化采用 {metadata.get("local_prediction_compression", "deflate")}；'
                       f'该阶段累计 {metadata.get("local_raw_export_seconds", float("nan")):.3f} s。'
                       '序列化策略只改变文件存储与运行开销，原始预测值不变。', ''])
        if 'map_reference_usable_index' in metadata:
            report.extend([f'地图参考帧：usable index {metadata["map_reference_usable_index"]}，'
                           f'原始 source index {metadata["map_reference_source_index"]}，'
                           f'{metadata["map_reference_frame_name"]}。'
                           f'初始/最终 C2W 距 identity：{metadata["initial_map_reference_identity_distance"]:.6g} / '
                           f'{metadata["final_map_reference_identity_distance"]:.6g}。', ''])
    else:
        report.extend([f'失败位置：{status.get("failure_phase", "模型加载或输入阶段")}。',
                       f'错误：`{status["error_type"]}: {status["error"]}`。',
                       '完整 traceback 见 error.txt；运行日志见 pipeline.log；已有局部预测与 partial memory 保留。', ''])
    report.extend(['原始每次网络 geometry/confidence/K/metric depth 保存为 local_predictions/call_*.npz。'
                   'memory confidence 为官方融合的 (conf-1)/conf；统一 confidence 仅代数恢复表示，不是原始网络输出。', '',
                   '最终地图为归一化/尺度不确定的 pointmap/pose；AMB3R metric depth 单独保留，'
                   '没有把 SfM 地图标为真实公制。未使用扫描点云尺度矫正。', '',
                   '官方 memory 不优化内参。最终 K 来自最近实际参与该帧的局部推理，'
                   'accepted call ids 可追溯；geometry_validation.json 的残差包含融合几何和局部 K 的差异。', '',
                   '未进行 GGGS/2DGS/3DGS/mesh/曝光训练，没有伪造 COLMAP 特征 tracks。', '',
                   f'[源码审计]({audit_relative})；'
                   '[官方 SfM pipeline](https://github.com/HengyiWang/amb3r/blob/'
                   f'{metadata["amb3r_git_commit"]}/sfm/pipeline.py)；'
                   '[AMB3R 论文](https://arxiv.org/abs/2511.20343)。'])
    Path(output, 'RUN_REPORT.md').write_text('\n'.join(report) + '\n')

