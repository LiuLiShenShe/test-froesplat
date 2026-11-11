#!/usr/bin/env python3
"""Summarize completed measurements after the batch and legacy audit, with no method changes."""
import argparse
import collections
import json
import statistics
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait-pid', type=int, required=True)
    p.add_argument('--batch-root', type=Path, required=True)
    a = p.parse_args()
    while Path(f'/proc/{a.wait_pid}/stat').exists():
        try:
            state = Path(f'/proc/{a.wait_pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == 'Z':
            break
        time.sleep(30)
    module = Path(__file__).resolve().parents[1]
    manifest = json.loads((module/'configs/plant_manifest.json').read_text())
    reports = a.batch_root/'reports'
    reports.mkdir(parents=True, exist_ok=True)
    measurements, rejections, refinement, paired = {}, collections.Counter(), {}, []
    for job in manifest['plants']:
        folder = a.batch_root/job['group']/job['plant']
        if not (folder/'C/COMPLETE.json').exists():
            continue
        data = json.loads((folder/'C/cleanup_metrics.json').read_text())
        measurements[job['plant']] = (job, data)
        edits = json.loads((folder/'C/edit_manifest.json').read_text())['edits']
        refinement[job['plant']] = dict(attempts=sum(e['action'] == 'refine' for e in edits),
            accepted=sum(e['action'] == 'refine' and e['accepted'] for e in edits))
        for edit in edits:
            if not edit['accepted']:
                for reason in edit['reasons']:
                    reasons = reason.split(':', 1)[1].split(',') if reason.startswith('crop_') else [reason.split(':', 1)[0]]
                    rejections.update(reasons)
        if (folder/'B/COMPLETE.json').exists():
            b = json.loads((folder/'B/cleanup_metrics.json').read_text())
            paired.append(dict(plant=job['plant'], group=job['group'],
                extra_artifact_reduction=data['artifact_reduction']-b['artifact_reduction'],
                fg_psnr_difference=data['final']['mean']['fg_psnr']-b['final']['mean']['fg_psnr'],
                seconds_difference=data['seconds']-b['seconds']))
    summary = {}
    for group in ['development', 'heldout_validation']:
        data = [m for j, m in measurements.values() if j['group'] == group]
        summary[group] = dict(measured=len(data), improved=sum(m['artifact_reduction'] > .002 for m in data),
            no_feasible_gain=[m['plant'] for m in data if m['outcome'] == 'no_feasible_gain'],
            mean_artifact_reduction=statistics.mean(m['artifact_reduction'] for m in data) if data else None,
            median_artifact_reduction=statistics.median(m['artifact_reduction'] for m in data) if data else None,
            mean_fg_psnr_delta=statistics.mean(m['final']['mean']['fg_psnr']-m['initial']['mean']['fg_psnr'] for m in data) if data else None)
    analysis = dict(completed_c=len(measurements), groups=summary, paired_c_minus_b=paired,
        refinement=refinement, common_rejection_reasons=dict(rejections.most_common()),
        method_frozen=(a.batch_root/'METHOD_FREEZE.json').exists(),
        semantic_certification='Binary masks only: pot/petiole/leaf-tip semantic errors remain independently unverified',
        production_method='C, global configs/default.yaml; no test-specific adjustment made by this summarizer')
    (reports/'MEASURED_GENERALIZATION_SUMMARY.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    text = '# Cross-plant measured generalization\n\n'
    text += f'Completed C models: {len(measurements)}/19. Development and heldout are reported separately.\n\n'
    for group, stats in summary.items():
        text += f'{group}: {stats["measured"]} measured; {stats["improved"]} improved; mean artifact reduction {stats["mean_artifact_reduction"]}; median {stats["median_artifact_reduction"]}; mean FG-PSNR change {stats["mean_fg_psnr_delta"]}.\n\n'
        text += f'No feasible gain: {", ".join(stats["no_feasible_gain"]) or "none among measured models"}.\n\n'
    text += '| Plant | Set | C minus B artifact reduction | C minus B FG-PSNR | C minus B seconds |\n|---|---|---:|---:|---:|\n'
    for item in paired:
        text += f'| {item["plant"]} | {item["group"]} | {item["extra_artifact_reduction"]:.6f} | {item["fg_psnr_difference"]:.6f} | {item["seconds_difference"]:.1f} |\n'
    text += '\nCommon actual rejection causes:\n\n'
    text += '\n'.join(f'- {key}: {count}' for key, count in rejections.most_common(12))
    text += '\n\nThese are quality-constraint rejections, not proof of segmentation ground truth. Binary masks do not independently certify every leaf tip, petiole or pot edge. Free-view point projection remains a proxy.\n'
    (reports/'GENERALIZATION_ANALYSIS.md').write_text(text)
    default = '# Final method decision\n\nProduction candidate is C with the frozen global default.yaml.\n\n'
    if paired:
        median = statistics.median(x['extra_artifact_reduction'] for x in paired)
        default += f'Paired C/B plants: {len(paired)}; median extra background reduction from C: {median:.6f}. '
        default += 'Refinement benefit is supported on these measured pairs.\n\n' if median > 0 else 'A repeatable extra refinement benefit is not established; do not claim mixed artifacts are solved.\n\n'
    else:
        default += 'Paired ablations are missing; refinement benefit is not established.\n\n'
    default += f'19-plant run completion: {len(measurements)}/19. Full multi-plant numbers are in CROSS_PLANT_COMPARISON.csv. '
    default += 'No test plant was given different rules or parameters. Remaining no-gain/quality-blocked mixed artifacts motivate uniform training-time background T×alpha and confidence-weighted surface/contour constraints; this run did not validate those training changes.\n'
    (reports/'FINAL_METHOD_DECISION.md').write_text(default)
    report = reports/'FINAL_AUTOMATIC_GAUSSIAN_CLEANING_REPORT.md'
    with report.open('a') as f:
        f.write('\n## 批处理结束后的实测分析\n\n'+text+'\n\n'+default)
        f.write('\n### 每株混合 Gaussian 优化实际采用情况\n\n| Plant | Optimization attempts | Accepted checkpoints |\n|---|---:|---:|\n')
        for plant, counts in refinement.items():
            f.write(f'| {plant} | {counts["attempts"]} | {counts["accepted"]} |\n')
        f.write('\n完整绝对路径见 CROSS_PLANT_COMPARISON.csv 的 OriginalPLY/NativePLY/ViewerPLY。仅完成模型可填写收益；失败与无收益保留原样。骨架/叶缘/暗色前景和深度/法线指标均为实测代理，不认证缺少细分类真值的花盆/叶柄语义。\n')
    mechanically_complete = len(measurements) == 19 and len(paired) == 6
    (reports/'FINALIZATION_STATUS.json').write_text(json.dumps(dict(report_generated=True,
        models_and_paired_ablation_complete=mechanically_complete, measured_models=len(measurements), paired_ablation_plants=len(paired),
        universal_semantic_cleanliness_certified=False), indent=2))


if __name__ == '__main__':
    main()
