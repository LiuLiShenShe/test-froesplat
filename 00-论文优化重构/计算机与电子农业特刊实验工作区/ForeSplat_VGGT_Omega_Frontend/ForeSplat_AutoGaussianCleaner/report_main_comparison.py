#!/usr/bin/env python3
"""Summarize completed same-input C0--C5 measurements without selecting a global method."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mean(report):
    return report.get('metrics', report.get('final'))['mean']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--main-root', type=Path, required=True)
    args = parser.parse_args()
    base = args.comparison.resolve()
    main_root = args.main_root.resolve()
    names = ['C0', 'C1_raw', 'C1_adaptive', 'C2_raw', 'C2_adaptive', 'C3_pre_marker', 'C3', 'C4']
    outputs = {name: base/'baselines'/name for name in names}
    outputs.update({name: main_root/name for name in ['C5-A', 'C5-B', 'C5-C', 'C5-D']})
    original = read(outputs['C0']/'cleanup_metrics.json')
    baseline = mean(original)
    original_sha = read(outputs['C0']/'COMPLETE.json')['input_sha256']
    records = []
    trials = []
    for name, output in outputs.items():
        marker = read(output/'COMPLETE.json')
        report = read(output/'cleanup_metrics.json')
        metrics = report.get('metrics', report.get('final'))
        if (marker['input_sha256'] != original_sha or report['original_gaussians'] != 576396
                or len(metrics['views']) != 208 or not metrics.get('metrics_complete', True)):
            raise ValueError('Incomplete or different original GGGS comparison: '+name)
        native = output/'cleaned_gaussians.ply'
        if sha(native) != marker['native_sha256']:
            raise ValueError('Native model changed: '+name)
        if name.startswith('C5-'):
            queue = read(main_root/'SERIAL_MAIN_QUEUE_STATUS.json')['methods'][name]
            if queue['exit_code'] != 0 or queue['common_evaluation_exit_code'] != 0:
                raise ValueError('Main CLI or common evaluation failed: '+name)
        m = mean(report)
        gain = (baseline['bg_alpha_mass_per_fg']-m['bg_alpha_mass_per_fg'])/baseline['bg_alpha_mass_per_fg']
        record = dict(method=name, original_gaussians=576396, cleaned_gaussians=report['cleaned_gaussians'],
            removed_gaussians=report['removed_gaussians'], refined_gaussians=report.get('refined_gaussians', 0),
            background_alpha_mass_reduction=gain, fg_psnr=m['fg_psnr'], fg_psnr_delta=m['fg_psnr']-baseline['fg_psnr'],
            foreground_mse_relative_change=m['fg_mse']/baseline['fg_mse']-1,
            dark_foreground_mse_relative_change=m['dark_foreground_mse']/baseline['dark_foreground_mse']-1,
            ssim=m['fg_ssim'], masked_lpips=m.get('fg_masked_lpips'),
            boundary_f1=m['boundary_f1'], foreground_local_boundary_f1=m.get('fg_local_boundary_f1'),
            thin_coverage=m['thin_coverage'], depth_changed_fraction=m['depth_changed_fraction'],
            normal_angle_p95_deg=m['normal_angle_p95_deg'], seconds=report['seconds'],
            peak_allocated_mib=report['peak_allocated_mib'], input_sha256=original_sha,
            native_sha256=marker['native_sha256'], native_ply=str(native),
            supersplat_ply=str(output/'cleaned_gaussians_supersplat.ply'),
            removed_native_ply=str(output/'removed_gaussians.ply'),
            removed_supersplat_ply=str(output/'removed_gaussians_supersplat.ply'),
            original_ids=str(output/'original_gaussian_ids.npy'),
            removed_ids=str(output/'removed_original_gaussian_ids.npy'),
            report=str(output/'cleanup_metrics.json'),
            timing_scope=('Reused historical output: common evaluation/export only' if name.startswith('C3') or name=='C4'
                else 'Original baseline evaluation/export; no cleaning' if name=='C0'
                else 'Automatic cleaning including analysis/trials/evaluation/export'))
        records.append(record)
        manifest = output/'edit_manifest.json'
        if name.startswith('C5-') and manifest.exists():
            edits = read(manifest)['edits']
            for action in sorted({item['action'] for item in edits}):
                items = [item for item in edits if item['action']==action]
                frequency = Counter()
                for item in items:
                    for reason in item.get('reasons', []):
                        for rule in reason.split(':', 1)[-1].split(','):
                            frequency[rule.strip()] += 1
                accepted = [item for item in items if item['accepted']]
                if any(not item.get('metrics_complete', True) for item in accepted):
                    raise ValueError('Accepted partial evaluation: '+name)
                trials.append(dict(method=name, action=action, trials=len(items), accepted=len(accepted),
                    rejected=len(items)-len(accepted), rejection_occurrences=dict(frequency),
                    scope='Occurrences of actual recorded decision rules; checkpoints and overlapping trials are not additive Gaussian counts'))
    destination = base/'comparisons'
    with (destination/'MAIN_C0_C5_MEASURED_RESULTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    by_name = {r['method']: r for r in records}
    c, d = by_name['C5-C'], by_name['C5-D']
    comparison = dict(same_native_model=c['native_sha256']==d['native_sha256'],
        additional_seconds=d['seconds']-c['seconds'], relative_time_increase=d['seconds']/c['seconds']-1,
        additional_removed=d['removed_gaussians']-c['removed_gaussians'], accepted_final_refinements=d['refined_gaussians'],
        scope='Completed matched main-plant ablation only; not a global cross-plant selection')
    (destination/'MAIN_C5_TRIAL_AND_REFINEMENT_AUDIT.json').write_text(json.dumps(dict(
        input_sha256=original_sha, trials=trials, covariance_increment=comparison), ensure_ascii=False, indent=2)+'\n')
    text = '# 已完成的 C0–C5 同源实测对照\n\n'
    text += '全部独立起点为 E10 原始 GGGS 30K 的 576,396 点；相同 208 相机、RGB、Mask、原生滤波 Renderer 和最大边 960。此报告是主开发样本对照，尚未选定跨植物正式版本。\n\n'
    text += '| 方法 | 删除点 | 背景 Alpha mass 减少 | FG-PSNR | FG-MSE 变化 | 暗前景 MSE 变化 | 时间（秒） |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in records:
        text += f"| {r['method']} | {r['removed_gaussians']:,} | {r['background_alpha_mass_reduction']:.3%} | {r['fg_psnr']:.5f} | {r['foreground_mse_relative_change']:+.3%} | {r['dark_foreground_mse_relative_change']:+.3%} | {r['seconds']:.1f} |\n"
    text += '\nMSE 正变化表示变差。C3/C4 的时间只含复用后的共同评价与导出，不是历史清理耗时；完整时间范围见 CSV。\n\n'
    text += ('旧 V2/V3 可显著清除黑云，同时造成前景照片损失，不能作为直接生产剪枝规则。KNN 新增候选须经过实际贡献与质量验证；现有主样本证据没有支持其安全增益。'
        'C3 依靠原生贡献归因与删除后重新分析有效；最终版另含 3,592 个历史人工标记排除点，必须与 pre-marker 版分别报告。\n\n')
    text += ('C4 发现了候选，但全量候选有局部暗叶片/几何损伤，整批撤销；50% 候选通过质量检查，却仅因 0.19146076% 的收益低于 0.2% 门槛被拒绝。'
        '新 C5-C 的自动空间分组、失败组细分和多轮归因已产生真实修改，并保留累计质量检查。\n\n')
    text += (f"C5-C 删除 {c['removed_gaussians']:,} 点，背景 Alpha mass 减少 {c['background_alpha_mass_reduction']:.3%}，FG-PSNR 改善 {c['fg_psnr_delta']:.5f} dB。"
        'Thin Coverage 不变；全图 Boundary F1 的变化与植物边界邻域 F1 分开记录。直接查看原生照片/自由视角后，俯视、斜视及部分侧视仍可见大黑云、叶旁黑条和碎点。不能宣称全部黑点清除，也不能称全面优于历史 C3。\n\n')
    text += (f"C5-D 与 C5-C 的原生模型字节相同：{comparison['same_native_model']}。混合协方差优化没有最终接受的修改，额外耗时 {comparison['additional_seconds']:.1f} 秒（+{comparison['relative_time_increase']:.1%}）。"
        '详细实际拒绝规则见 MAIN_C5_TRIAL_AND_REFINEMENT_AUDIT.json。当前主样本不支持为这项零增益优化付出额外成本；开发集仍使用固定 C 版本。\n\n')
    text += '每组可直接打开的模型与报告：\n\n'
    for r in records:
        text += f"- **{r['method']}**：[原生 GGGS]({r['native_ply']}) · [SuperSplat]({r['supersplat_ply']}) · [Removed]({r['removed_native_ply']}) · [报告]({r['report']})\n"
    text += '\n跨植物效果、平均耗时/显存、统一参数选择及独立十三株结果仍须实际完成后报告。全部 19 株原生输入的最新格式核验另见 NATIVE_GGGS_INPUT_INTEGRITY.json；格式准备完成不代表清理验证完成。\n\n'
    text += '来源：本地完整模型与 208 相机实测、实际直接图像查看、原始拒绝记录。博查搜索返回 HTTP 403，未用训练数据替代实验依据。\n'
    (destination/'MAIN_C0_C5_MEASURED_REPORT.md').write_text(text)
    print(destination/'MAIN_C0_C5_MEASURED_REPORT.md')


if __name__ == '__main__':
    main()
