#!/usr/bin/env python3
"""Serializable, restartable training/cleaning/ablation driver; no input inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import argparse
import csv
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ForeSplat_AutoGaussianCleaner.cleaner.io import sha, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--prepare-missing-models', action='store_true')
    p.add_argument('--ablations', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--group', choices=['development', 'heldout_validation', 'all'], default='all')
    p.add_argument('--plant', action='append')
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/default.yaml')
    a = p.parse_args()
    a.output_root = a.output_root.resolve()
    a.output_root.mkdir(parents=True, exist_ok=True)
    lock = (a.output_root/'execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(a.manifest.read_text())
    scripts = Path(__file__).resolve().parent
    module, root = scripts.parent, scripts.parent.parent
    # Freeze source and default config BEFORE any heldout metric is generated.
    sources = sorted(list((module/'cleaner').glob('*.py'))+list((module/'evaluation').glob('*.py'))+list(scripts.glob('*.py')))
    fingerprint = dict(config_sha256=sha(a.config), sources={str(x.relative_to(module)): sha(x) for x in sources}, manifest_sha256=sha(a.manifest))
    frozen = a.output_root/'METHOD_FREEZE.json'
    if frozen.exists() and json.loads(frozen.read_text()) != fingerprint:
        raise ValueError('Method/config changed after batch freeze; use a new experiment/version and disclose test reuse')
    write_json(frozen, fingerprint)
    jobs = [x for x in manifest['plants'] if (a.group == 'all' or x['group'] == a.group) and (not a.plant or x['plant'] in a.plant)]
    jobs.sort(key=lambda x: (x['group'] != 'development', x['plant'] != 'KongQueZhuYu', x['plant']))
    records = {}
    status_file = a.output_root/'batch_status.json'
    if status_file.exists():
        records = json.loads(status_file.read_text()).get('plants', {})
    def status(plant, state, **kw):
        records[plant] = dict(status=state, updated=time.time(), **kw)
        write_json(status_file, dict(plants=records, updated=time.time(), physical_gpu=1))
        report_batch(manifest, a.output_root, records)
    def execute(plant, stage, command, cwd):
        current_fingerprint = dict(config_sha256=sha(a.config), sources={str(x.relative_to(module)): sha(x) for x in sources}, manifest_sha256=sha(a.manifest))
        if current_fingerprint != fingerprint:
            raise RuntimeError('Frozen method changed during batch; stopping dependent stage to protect heldout validity')
        log = a.output_root/'logs'/f'{plant}_{stage}.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('a') as f:
            f.write(json.dumps(dict(command=command, cwd=str(cwd), physical_gpu=1))+'\n'); f.flush()
            process = subprocess.Popen(command, cwd=cwd, env=os.environ.copy(), stdout=f, stderr=subprocess.STDOUT)
            status(plant, 'running', stage=stage, pid=process.pid, log=str(log))
            code = process.wait()
        if code:
            raise RuntimeError(f'{stage} exited {code}; see {log}')
    deferred = []
    for job in jobs:
        plant = job['plant']
        scene = json.loads(Path(job['scene']).read_text())
        try:
            if not Path(scene['model']).exists():
                if not a.prepare_missing_models:
                    status(plant, 'missing_gggs_checkpoint', expected_model=scene['model'])
                    continue
                training = Path(job['training_dir'])
                # Standard locked GGGS core and corrected camera interface; no cleaning training losses.
                events = training/'densification_events.jsonl'
                if events.exists() and not events.with_suffix('.complete.json').exists():
                    raise RuntimeError('Interrupted training requires checkpoint recovery; not silently restarting or using partial model')
                command = ['uv', 'run', '--no-project', '--python', str(root/'.venv/bin/python'), 'python',
                    str(root/'experiments/KongQueZhuYu/geometry_rgb_dual_frontend/scripts/train_observed.py'),
                    '--repo', scene['gggs_repo'], '--events', str(events), '--', '--source_path', scene['dataset'],
                    '--model_path', str(training/'checkpoints'), '--iterations', '30000', '--resolution', '2',
                    '--sh_degree', '3', '--sg_degree', '0', '--test_iterations', '10000', '20000', '30000',
                    '--save_iterations', '7000', '15000', '30000', '--checkpoint_iterations', '7000', '15000', '30000',
                    '--metrics_csv', str(training/'train_metrics.csv'), '--metrics_interval', '100']
                write_json(training/'TRAINING_PROTOCOL.json', dict(command=command, seed=0, iterations=30000,
                    input_bundle_reused=True, new_inference=False, frontend=scene['frontend'], source_hashes=fingerprint))
                execute(plant, 'train30K', command, Path(scene['gggs_repo']))
                if not Path(scene['model']).exists():
                    raise RuntimeError('Training completed without final PLY')
            ablation_plants = [x['plant'] for x in manifest['plants'] if x['group'] == 'development']
            # The heldout ablation sample is fixed from the manifest before results, not selected by performance.
            ablation_plants += sorted(x['plant'] for x in manifest['plants'] if x['group'] == 'heldout_validation')[:2]
            variants = ['C']
            for variant in variants:
                out = a.output_root/job['group']/plant/variant
                command = ['uv', 'run', '--no-project', '--python', str(root/'.venv/bin/python'), 'python',
                    str(scripts/'auto_clean_gaussians.py'), '--scene', job['scene'], '--output', str(out),
                    '--config', str(a.config), '--variant', variant]
                if a.resume:
                    command.append('--resume')
                execute(plant, variant, command, root)
            if (a.ablations and plant in ablation_plants) or scene.get('baseline_models'):
                deferred.append((job, scene, a.ablations and plant in ablation_plants))
            status(plant, 'cleaning_complete_ablations_pending' if (a.ablations and plant in ablation_plants) or scene.get('baseline_models') else 'complete',
                model=scene['model'], output=str(a.output_root/job['group']/plant))
        except Exception as error:
            status(plant, 'failed', error=str(error), expected_model=scene['model'])
            print(plant, 'FAILED', error, flush=True)
    # Prioritize cross-plant C measurements/model preparation before repeated ablation renders.
    for job, scene, do_ablations in deferred:
        plant = job['plant']
        try:
            if do_ablations:
                for variant in ['A_once', 'A', 'B_once', 'B']:
                    out = a.output_root/job['group']/plant/variant
                    command = ['uv', 'run', '--no-project', '--python', str(root/'.venv/bin/python'), 'python',
                        str(scripts/'auto_clean_gaussians.py'), '--scene', job['scene'], '--output', str(out),
                        '--config', str(a.config), '--variant', variant]
                    if a.resume:
                        command.append('--resume')
                    execute(plant, variant, command, root)
            if scene.get('baseline_models'):
                out = a.output_root/job['group']/plant/'historical_baselines'
                if not (out/'baseline_comparison.json').exists():
                    execute(plant, 'baselines', ['uv', 'run', '--no-project', '--python', str(root/'.venv/bin/python'),
                        'python', str(scripts/'evaluate_baselines.py'), '--scene', job['scene'], '--output', str(out),
                        '--config', str(a.config)], root)
                # R5 is a separate input baseline; do not confuse its prior manual edits with this method's gain.
                spec_r5 = {**scene, 'model': scene['baseline_models']['R5']}
                r5_scene = a.output_root/job['group']/plant/'R5_scene.json'
                write_json(r5_scene, spec_r5)
                execute(plant, 'R5_auto_C', ['uv', 'run', '--no-project', '--python', str(root/'.venv/bin/python'),
                    'python', str(scripts/'auto_clean_gaussians.py'), '--scene', str(r5_scene), '--output',
                    str(a.output_root/job['group']/plant/'R5_auto_C'), '--config', str(a.config), '--resume'], root)
            status(plant, 'complete', model=scene['model'], output=str(a.output_root/job['group']/plant))
        except Exception as error:
            status(plant, 'failed', error=str(error), expected_model=scene['model'])
            print(plant, 'FAILED', error, flush=True)
    report_batch(manifest, a.output_root, records)
    print('Batch statuses:', {k:v['status'] for k,v in records.items()}, flush=True)
    if any(r['status'] == 'failed' for r in records.values()):
        raise SystemExit(1)


def report_batch(manifest, output, states):
    rows, ablations = [], []
    for job in manifest['plants']:
        plant = job['plant']
        folder = output/job['group']/plant
        metrics_path = folder/'C/cleanup_metrics.json'
        measured = json.loads(metrics_path.read_text()) if (folder/'C/COMPLETE.json').exists() else None
        source = json.loads(Path(job['scene']).read_text())['model']
        row = dict(Plant=plant, Group=job['group'], OriginalPLY=source, ModelAvailable=Path(source).exists(),
            MainStageStatus=states.get(plant, {}).get('status', 'pending'), OriginalGaussians='', CleanedGaussians='', ArtifactReduction='',
            ForegroundPreservation='', FG_PSNR_Delta='', BoundaryF1_Delta='', ThinCoverage_Delta='',
            DepthP95='', NormalP95Deg='', RuntimeSeconds='', PeakAllocatedMiB='', NativePLY='', ViewerPLY='',
            Outcome=states.get(plant, {}).get('status', 'pending'), Failure=states.get(plant, {}).get('error', ''))
        if measured:
            initial, final = measured['initial']['mean'], measured['final']['mean']
            row.update(OriginalGaussians=measured['original_gaussians'], CleanedGaussians=measured['cleaned_gaussians'],
                ArtifactReduction=measured['artifact_reduction'], ForegroundPreservation=measured['foreground_preservation'],
                FG_PSNR_Delta=final['fg_psnr']-initial['fg_psnr'], BoundaryF1_Delta=final['boundary_f1']-initial['boundary_f1'],
                ThinCoverage_Delta=final['thin_coverage']-initial['thin_coverage'], DepthP95=final['depth_relative_p95'],
                NormalP95Deg=final['normal_angle_p95_deg'], RuntimeSeconds=measured['seconds'], PeakAllocatedMiB=measured['peak_allocated_mib'],
                NativePLY=measured['native_ply'], ViewerPLY=measured['supersplat_ply'], Outcome=measured['outcome'])
        rows.append(row)
        for variant in ['C', 'A_once', 'A', 'B_once', 'B']:
            f = folder/variant/'cleanup_metrics.json'
            if (folder/variant/'COMPLETE.json').exists():
                m = json.loads(f.read_text())
                ablations.append(dict(Plant=plant, Group=job['group'], Variant=variant,
                    ArtifactReduction=m['artifact_reduction'], ForegroundPreservation=m['foreground_preservation'],
                    Removed=m['removed_gaussians'], FG_PSNR_Delta=m['final']['mean']['fg_psnr']-m['initial']['mean']['fg_psnr'],
                    Rounds=m['rounds'], Outcome=m['outcome'], Model=m['native_ply']))
    reports = output/'reports'
    reports.mkdir(exist_ok=True)
    for name, data in [('CROSS_PLANT_COMPARISON.csv', rows), ('ABLATIONS.csv', ablations)]:
        if data:
            with (reports/name).open('w', newline='') as f:
                w=csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    done = sum(bool(r['NativePLY']) for r in rows)
    text = '# ForeSplat Automatic Gaussian Cleaning — 实测状态报告\n\n'
    text += f'完整清理输出 **{done}/19**；已完成方法变体 **{len(ablations)}** 份（含C）。这是运行状态报告；未完成的植物不构成泛化成功证据。\n\n'
    text += '| Plant | Group | Original | Cleaned | Artifact reduction | FG alpha ratio | Outcome |\n|---|---|---:|---:|---:|---:|---|\n'
    for r in rows:
        text += f'| {r["Plant"]} | {r["Group"]} | {r["OriginalGaussians"]} | {r["CleanedGaussians"]} | {r["ArtifactReduction"]} | {r["ForegroundPreservation"]} | {r["Outcome"]} |\n'
    text += '\n绝对模型路径及指标：CROSS_PLANT_COMPARISON.csv；完整候选和决策保存在各株 rounds/。\n\n'
    text += '1. P2/R5限制：中心/邻域保护未考虑完整footprint；删除前层后露出后层；混合点不能整点删除。\n\n'
    text += '2. 自动识别：原生T×alpha、前景/背景比例、有效视角、外部深度矩、协方差与空间距离；当前模型每轮重算。\n\n'
    text += '3. 暗色叶片：颜色不进入删除/修复分类；原RGB暗色前景单独检查Alpha和MSE回归。\n\n'
    text += '4. 混合Gaussian：只优化自动选出的各轴log-scale与旋转，固定有效opacity、XYZ和SH；全相机受约束选优/回滚。实际收益必须看C对B，不能只因已实现而认定成功。\n\n'
    text += '5. 薄结构：Mask叶缘、低厚度骨架贡献与逐相机质量闸门；只有二值Mask时不能独立认证所有叶柄/花盆语义。\n\n'
    text += '6. 停止：无可行收益、背景收益低于0.2%，或安全上限12轮。每轮与初始模型比较，防累计透明化。\n\n'
    text += '7. KongQueZhuYu：新版实测见表；历史510248/68362px、4011黑贡献点属旧固定相机协议，不能同新自由视角代理混比。\n\n'
    text += '8–9. 跨株收益和失败：仅表中实测完成项可评价；pending/running/failed不得填数值。失败日志在batch_status.json。\n\n'
    text += '10. 一键批量入口已实现，可补模型；全部19株完成之前不能宣称验收通过。\n\n'
    text += '11. 相比P2/R5已去除手工ROI/ID/世界距离依赖；泛化收益需独立多株及消融实测，尚未测完不作成功结论。\n\n'
    text += '12. 默认候选C，configs/default.yaml为全局配置，METHOD_FREEZE.json冻结源码/参数；独立测试触发升级则须新保留集，不得重新标签为独立。\n\n'
    text += '13. 若混合点在质量约束内无法改善，建议训练中增加颜色无关背景T×alpha、轮廓泄漏和置信度加权多视角表面损失。当前不修改训练损失，也不声称未来改进有效。\n\n'
    text += '自由视角由相机轨迹和外部深度点生成，外部点投影/包围盒为保守代理而非真实Mask。SSIM/LPIPS是前景代理。没有叶/盆/柄细分类GT，不得把代理指标写作语义错误全部排除。\n\n'
    text += '资料：本地原生GGGS与历史诊断；PyTorch2.10官方文档 https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/autograd.rst 。博查HTTP403额度不足，无可用搜索结果。\n'
    (reports/'FINAL_AUTOMATIC_GAUSSIAN_CLEANING_REPORT.md').write_text(text)
    (reports/'GENERALIZATION_ANALYSIS.md').write_text(f'# Generalization status\n\nCompleted {done}/19. Frozen dev=4; heldout=15. '
        'No plant-specific cleaning configuration. Only completed heldout measurements can support generalization. '
        'Historical Kong uses Standard VGGT; prepared plants reuse Omega bundles, disclosed frontend difference.\n')
    (reports/'FINAL_METHOD_DECISION.md').write_text('# Method decision\n\nCandidate default C is implemented. '
        'Acceptance as a generalizable default remains conditional on independent measured benefit and ablations. '
        f'Complete outputs: {done}/19. Unfinished/failed plants remain explicit in comparison CSV.\n')
    write_json(reports/'AUTOMATIC_FAILURE_PROXIES.json', {job['plant']: json.loads(
        (output/job['group']/job['plant']/'C/cleanup_metrics.json').read_text()).get('automatic_failure_proxies', {})
        for job in manifest['plants'] if (output/job['group']/job['plant']/'C/COMPLETE.json').exists()})


if __name__ == '__main__':
    main()
