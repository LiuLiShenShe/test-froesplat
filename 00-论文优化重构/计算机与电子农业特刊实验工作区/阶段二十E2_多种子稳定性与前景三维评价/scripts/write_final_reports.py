"""Write conclusions only after real matrix, evaluation and acceptance evidence exist."""
from __future__ import annotations
import json
import statistics
from common import *


def fmt(value):
    if value in ('',None):return 'undefined'
    if isinstance(value,str) and not value.replace('.','',1).replace('-','',1).replace('e','',1).replace('+','',1).isdigit():return value
    return f'{float(value):.6f}'


def table(rows,columns):
    return '\n'.join(['| '+' | '.join(label for _,label in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+
                     ['| '+' | '.join(fmt(r.get(key,'')) for key,_ in columns)+' |' for r in rows])


def main():
    acceptance=read_json(STAGE/'reports/FINAL_ACCEPTANCE.json')
    recovery=read_json(STAGE/'reports/FINALIZATION_RECOVERY.json')
    recovery_checks=[step for step in recovery['steps'] if step['script']!='write_final_reports.py']
    assert recovery['original_failure_preserved'] and recovery_checks
    assert all(step['status']=='PASS' for step in recovery_checks)
    decision=read_json(STAGE/'reports/ROBUSTNESS_DECISION.json')
    effects=read_csv(STAGE/'PAIRED_SEED_EFFECTS.csv')
    summary=read_csv(STAGE/'MULTISEED_SCENE_SUMMARY.csv')
    stats=read_csv(STAGE/'reports/PAIRED_MULTISEED_STATISTICS.csv')
    added_stats=read_csv(STAGE/'reports/NEW_SEEDS_ONLY_STATISTICS.csv')
    manifest=read_csv(STAGE/'MULTISEED_TRAINING_MANIFEST.csv')
    metric_omissions=read_csv(STAGE/'reports/METRIC_OMISSIONS.csv')
    paired_omissions=read_csv(STAGE/'reports/PAIRED_METRIC_OMISSIONS.csv')
    new=[r for r in manifest if int(r['seed'])>0]
    index={(r['sample'],r['metric']):r for r in stats}
    new_index={(r['sample'],r['metric']):r for r in added_stats}
    scene=decision['scene_decisions'];overall=decision['R1_3D_BENEFIT']
    full_roi=[]
    for sample in SCENES:
        for metric in ('PSNR','foreground_PSNR'):
            full_roi.append(index[sample,metric])
    conclusion=[]
    conclusion.append(f'# R1 三维重建证据判定\n\n实际新增完成 {len(new)}/8 组 30,000 迭代训练；Seed 0 原模型保持不变。总判定为 **{overall}**。WanNianQing1：**{scene["WanNianQing1"]["classification"]}**；CaoMei2：**{scene["CaoMei2"]["classification"]}**。这是重建模型外观及代理结构的有限证据，不是独立三维真值精度判定。\n')
    conclusion.append('冻结判据要求全图和固定 ROI PSNR 在三个 seed 中同向，且两项配对均值幅度均超过 B、C 各自跨 seed 的样本标准差，才给出 ROBUST_POSITIVE/NEGATIVE。该保守描述规则不是统计显著性检验。Seed 0 只有源码路径等价证据，没有历史现场 RNG hash；严格统一包装的三种子论断暂停，新增 Seed 1/2 单独统计同时保留。\n')
    conclusion.append(table(full_roi,[('sample','Scene'),('metric','Endpoint'),('seed0_delta','ΔS0'),('seed1_delta','ΔS1'),('seed2_delta','ΔS2'),('paired_mean','Paired mean'),('paired_sample_SD','Paired SD'),('B_sample_SD','B SD'),('C_sample_SD','C SD')]))
    conclusion.append('\n## 十二项问题的回答\n')
    for sample in SCENES:
        raw=index[sample,'PSNR'];roi=index[sample,'foreground_PSNR']
        conclusion.append(f'- {sample}：全图 PSNR 方向 `{raw["effect_signs"]}`，固定 ROI PSNR 方向 `{roi["effect_signs"]}`；全图均值效应/max(B SD,C SD)={fmt(raw["absolute_mean_effect_over_max_group_SD"])}，ROI 对应比值={fmt(roi["absolute_mean_effect_over_max_group_SD"])}。判定 `{scene[sample]["classification"]}`。')
    wn=scene['WanNianQing1']['classification'];cm=scene['CaoMei2']['classification']
    conclusion.extend([
        '\n1. **跨 seed 是否同向？** 以上逐项方向列是直接证据；只在三个符号一致且非零的单项上称同向，不把不同端点合并为一致结论。',
        f'2. **WanNianQing1 稳定改善？** {"在冻结的两个外观端点和保守波动阈值下观察到改善，但范围仅限此场景、三种子和非独立 ROI。" if wn=="ROBUST_POSITIVE" else "现有配对结果不支持将其写成稳定三维改善；最终分类为 "+wn+"。"}',
        f'3. **CaoMei2 稳定退化？** {"两个外观端点在本有限重复中同向退化并超过描述阈值；不能进一步推断真实几何退化。" if cm=="ROBUST_NEGATIVE" else "不能将旧 Seed 0 退化直接推广为稳定三维退化；最终分类为 "+cm+"。"}',
        '4. **是否超过训练波动？** 每项给出 B/C 跨种子 SD、配对 SD 和比值；超过描述阈值的单项也不等于统计显著性或几何精度改善。若方向反复则不判稳健。',
        '5. **前景与全图结论是否相同？** 逐 scene 的两行结果独立展示。Full raw RGB 包含背景目标差异，ROI 来源于 R0 并存在方法偏差；任何方向冲突保留为 MIXED，不将其平均掉。',
        '6. **背景抑制是否额外改善？** 下表提供 outside-ROI RGB energy、alpha 和缺失 alpha 的逐 seed 差值；负 energy/alpha 表示该代理降低。它与前景覆盖需并列解释，不能由黑背景模型全图分数代替。',
        '7. **是否支持更稳定 Gaussian 分布？** 数量、激活 scale/opacity、depth 和内部 normal 的 SD 均提供。没有独立结构 GT，分布收窄或数量变化不能验证更优/更稳定的真实植物几何；本阶段不作该已证明声明。',
        '8. **R1 是否应成为默认三维排序器？** 证据不足以基于额外三维收益将其设为 DEFAULT_3D；当前角色为 ROBUST_SELECTION_ONLY。默认选择模块的二维依据与三维收益主张必须分开。',
        '9. **是否只能定位为选择鲁棒性改进？** 本阶段三维证据不足以扩张其角色；保留此前独立二维验证支持的选择模块身份，未重新验证或夸大二维结论。',
        '10. **还需新场景和 seed？** 若要主张普遍三维增益，需要具有实际 R0/R1 输入差异的新场景与事先固定的重复预算；优先增加独立植物场景，再扩展种子。不按现有结果选择有利 seed。',
        '11. **是否进入 M2M3/mesh？** 完成本阶段验收后可另立 Phase20E3，以真实实现修正方法，并按事先冻结的 B_R0 Seed 0 参考做 mesh 恢复和描述性结构分析；不把这当作已证明的 R1 优势，也不自动执行。',
        '12. **Table 7 如何处理？** 旧四方法“20序列平均”、R1 稳定三维增益、未实现双向 M2M3、局部 Floor40 保证及统一条件加速声明需替换或降级。旧数字保持 PAPER_REPORTED_UNVERIFIED；本阶段只能提供两场景 B/C 三种子的范围明确消融，不能替换整个四方法 benchmark。',
    ])
    for label,metrics in [('背景与覆盖',('outside_ROI_RGB_energy','outside_ROI_alpha_mean','missing_alpha_fraction','foreground_coverage_proxy')),
                          ('描述性结构',('Gaussian_count','scale_p50','scale_p95','opacity_p50','depth_p50','normal_consistency_cosine_proxy'))]:
        selected=[index[s,m] for s in SCENES for m in metrics if (s,m) in index]
        conclusion.append('\n## '+label+'\n\n'+table(selected,[('sample','Scene'),('metric','Proxy'),('seed0_delta','ΔS0'),('seed1_delta','ΔS1'),('seed2_delta','ΔS2'),('paired_mean','Mean Δ'),('paired_sample_SD','SD Δ'),('B_sample_SD','B SD'),('C_sample_SD','C SD')]))
    conclusion.append('\n来源：完整逐视图 float 指标、PAIRED_SEED_EFFECTS.csv、reports/PAIRED_MULTISEED_STATISTICS.csv；所有 proxy 非独立 GT，全部负结果和三种子保留。')
    conclusion.append(f'可靠性缺项：{len(metric_omissions)} 个逐视角指标未定义，{len(paired_omissions)} 项 seed/metric 配对被省略。逐项原因与有效相机数见 reports/METRIC_OMISSIONS.csv、PAIRED_METRIC_OMISSIONS.csv 和 METRIC_VALID_VIEW_COUNTS.csv；没有补值或用不等相机子集作配对。')
    (STAGE/'R1_3D_ROBUSTNESS_DECISION.md').write_text('\n\n'.join(conclusion)+'\n',encoding='utf-8')
    block=f'''PHASE20E2 STATUS: {acceptance['status']}

ADDITIONAL_TRAININGS_COMPLETED: {len(new)}/8

SEEDS:
0, 1, 2

WAN_NIAN_QING_R1_3D:
{wn}

CAO_MEI_R1_3D:
{cm}

FOREGROUND_PROXY_EVALUATION:
COMPLETE

R1_3D_BENEFIT:
{overall}

R1_FINAL_ROLE:
ROBUST_SELECTION_ONLY

M2M3_NEXT_ACTION:
REWRITE_TO_IMPLEMENTATION

NEXT_PHASE:
{'PHASE20E3' if acceptance['status']=='PASS' else 'REMEDIATION_REQUIRED'}'''
    report=[f'# Phase20E-2 最终报告\n\n```text\n{block}\n```',
            f'本阶段真实完成 {len(new)} 组新增训练，每组达到 30,000 iterations，并通过 checkpoint/PLY 重载。继承 8 个历史实体模型，HongZhang B/C 共用其中一个；最终表含 17 个 group/seed 行、{acceptance["per_view_rows"]} 个逐视图行、{acceptance["unique_float_renders"]} 个独立模型-相机浮点渲染。两个变化场景各三种子，验证视角不是训练重复。',
            '## A. Full raw RGB appearance fidelity\n\n所有组使用同一 canonical undistorted raw RGB、冻结 heldout 相机、原色彩范围和浮点归一化、SSIM kernel 与 LPIPS 版本/权重、尺寸和黑背景。下表仅比较 B/C 的配对最终模型；A 组只有原 Seed 0，不对 A→B 的随机性作结论。',
            table(effects,[('sample','Scene'),('seed','Seed'),('B_PSNR','B PSNR'),('C_PSNR','C PSNR'),('delta_PSNR','ΔPSNR'),('delta_SSIM','ΔSSIM'),('delta_LPIPS','ΔLPIPS')]),
            '## B. Fixed proxy foreground appearance fidelity\n\n每视角统一使用原 R0_proxy mask；MSE 分母为 3×ROI 像素数。Composited SSIM/LPIPS 含共同黑背景及边界卷积影响，均为代理；R0 来源可能偏向 B。没有新增人工 GT。',
            table(effects,[('sample','Scene'),('seed','Seed'),('B_foreground_PSNR','B ROI PSNR'),('C_foreground_PSNR','C ROI PSNR'),('delta_foreground_PSNR','ΔROI PSNR'),('delta_fixed_mask_composited_SSIM','Δcomp SSIM'),('delta_fixed_mask_composited_LPIPS','Δcomp LPIPS'),('delta_foreground_absolute_color_error','Δabs color')]),
            '## C. Background suppression\n\nOutside-ROI RGB energy/alpha 与 missing-alpha 同时报告；外部 alpha 连通区域只是外观代理，不能全部叫 floating Gaussian。',
            table(effects,[('sample','Scene'),('seed','Seed'),('delta_outside_ROI_RGB_energy','Δenergy'),('delta_outside_ROI_alpha_mean','Δalpha'),('delta_missing_alpha_fraction','Δmissing alpha'),('delta_outside_alpha_components_ge16px','Δcomponents')]),
            '## D. Descriptive geometric behavior\n\nGaussian count、scale/opacity 分布、alpha 支持的 depth、渲染与深度法向内部一致性均为描述。深度单位为 inherited COLMAP 任意单位，无独立几何 GT，不计算真值 Chamfer/F-score。A/B/C/D 不合并成排名。',
            table(effects,[('sample','Scene'),('seed','Seed'),('B_Gaussian_count','B count'),('C_Gaussian_count','C count'),('delta_Gaussian_count','Δcount'),('delta_depth_p50','Δdepth median'),('delta_normal_consistency_cosine_proxy','Δnormal proxy')]),
            '## 多种子波动与证据范围\n\n配对单位是每个 scene 内的一次 seed B/C 训练对。样本 SD 使用 ddof=1；无 p 值，无将视角当独立实验的显著性声明。所有 seed、负值及描述性离群标记保留。',
            table(full_roi,[('sample','Scene'),('metric','Endpoint'),('paired_mean','Paired mean'),('paired_sample_SD','Paired SD'),('paired_min','Min'),('paired_max','Max'),('effect_signs','Directions'),('B_sample_SD','B SD'),('C_sample_SD','C SD')]),
            'Seed 0 的现场初始 RNG 未保存。源码、环境、原 safe_state 路径与当前重放支持路径等价，但不能伪造历史 hash；严格统一包装三种子声明暂停。新增 Seed 1/2 的真实初始化与 30,000 次相机选择逐对相同，四组配对均通过现场证据。两新增 seed 的独立描述如下，不将 n=2 夸大为显著性。',
            table([new_index[s,m] for s in SCENES for m in ('PSNR','foreground_PSNR')],[('sample','Scene'),('metric','Endpoint'),('seed1_delta','ΔS1'),('seed2_delta','ΔS2'),('paired_mean','Mean Δ'),('paired_sample_SD','SD Δ'),('B_sample_SD','B SD'),('C_sample_SD','C SD')]),
            f'## 运行与失败\n\n新增训练总 wall-clock（包含各运行场景加载与原生训练记录边界）为 {sum(float(r["training_wall_seconds"]) for r in new):.1f} 秒。物理 GPU UUID 均为 `{GPU_UUID}`，与历史一致。运行命令、环境、初始状态、checkpoint/PLY SHA256 和逐迭代曲线见 manifest 与 logs。原生 30,000 iterations 对应 29,999 次 optimizer.step，忠实继承原实现。',
            'WanNianQing1/B_R0/Seed 1 的前两次阶段记录启动在 iteration=0、optimizer_steps=0 时因 NumPy 标量 JSON 序列化错误失败。失败脚本、日志和目录保留；事先计划未覆盖此种记录错误，查看新增优化结果前已另存 PREOPTIMIZATION_REMEDIATION.json 公开补充。该偏离不能隐藏成原计划内成功。修复只转录元数据类型，不改训练数学、seed、输入或 GPU；没有质量驱动重跑。',
            '四个变化训练视图的原始候选与冻结 B/C RGBA 已只读核对：内部 RGB 与 canonical RGB 精确相同，掩膜外 RGB 精确为零，alpha 与冻结 Mask 一致。CaoMei2 raw0139 的注册图像实际为 crop_0124.png。输入可视化与直接查看记录见 CHANGED_TRAINING_INPUT_AUDIT.csv 和 INPUT_VISUAL_QA.md；不把训练视图选择差异当作 held-out 三维收益。',
            '## 可视化与真实重算\n\nfigures 包含事先按相机顺序冻结的 first/middle/last 视角，以及明确标为探索性的旧 Seed 0 ROI 最好/最差/近零案例。每个视角展示 canonical raw RGB、B/C 三种子、真实 RGB 差值、alpha、depth 和 native normal。差值与 depth 共用显示范围，显示 PNG 不用于正式评价；索引保存真实 float 来源和 SHA。最终图像目视 QA 与刷新命令记录见 reports/FIGURE_VISUAL_QA.md 和 reports/FIGURE_REFRESH_PROGRESS.json。',
            f'全部 {acceptance["unique_float_renders"]} 个实际浮点文件已由 recompute_metrics.py 重算，参考图与 ROI 原始/resize hash 逐视角一致。Seed 0 正式 RGB/alpha 直接继承原 float，depth/normal 为同冻结 PLY 新目录 replay；重放差异单列，未改旧指标。详见 reports/FLOAT_RENDER_METRIC_RECOMPUTATION.json。',
            '## 判定与下一阶段\n\n'+f'总体 `{overall}`；R1 当前定位 `ROBUST_SELECTION_ONLY`，不能写成额外三维增益已证明。逐项十二问、端点差异、波动比及结构代理解读见 R1_3D_ROBUSTNESS_DECISION.md。',
            'M2M3 下一动作 `REWRITE_TO_IMPLEMENTATION`：按真实加权剪枝/全局 floor 描述，未实现的双向匹配另立任务。Mesh 工程参考已预先固定旧 B_R0/Seed 0，物理尺度和表型资料仍缺。本阶段没有修改/执行 M2M3、mesh 或论文正文。详见 M2M3_METHOD_CORRECTION_HANDOFF.md 和 NEXT_PHASE_EXECUTION_PLAN.md。',
            '## 验收与恢复\n\n'+table(acceptance['gates'],[('gate','Gate'),('description','Requirement'),('status','Result')]),
            f'首次自动终结在 `{recovery["original_failed_script"]}` 因 HongZhang Seed 0 共享别名断言失败；原失败进度及日志保持原样，其 SHA256 为 `{recovery["original_failed_log_sha256"]}`。修复仅严格校验同一 checkpoint/render 的 B/C 别名，并对全部唯一浮点渲染重算；随后图表与 G1-G18 验收均通过。恢复命令、日志哈希和原始证据哈希见 `reports/FINALIZATION_RECOVERY.json`，未重新训练或渲染。',
            '完整恢复路径、命令与大文件索引见 RECOVERY_AND_REPRODUCTION.md、reports/LARGE_ARTIFACT_MANIFEST.csv。大模型保留独立管理目录，不上传 Git、不覆盖 E1。所有生产源和历史文件已复核 SHA256；本阶段之外 tracked diff 必须为空。',
            '来源：本仓库冻结输入、实际 GPU 训练/渲染、只读源码与 SHA256、float 重算报告。Context7 当前 PyTorch 文档已存档；博查检索 HTTP403 额度不足，外部搜索不可用，没有用训练数据填补实验结果。']
    (STAGE/'PHASE20E2_FINAL_REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    write_json(STAGE/'reports/FINAL_STATUS.json',{'timestamp':now(),'PHASE20E2_STATUS':acceptance['status'],'ADDITIONAL_TRAININGS_COMPLETED':f'{len(new)}/8',
                'SEEDS':[0,1,2],'WAN_NIAN_QING_R1_3D':wn,'CAO_MEI_R1_3D':cm,'FOREGROUND_PROXY_EVALUATION':'COMPLETE','R1_3D_BENEFIT':overall,
                'R1_FINAL_ROLE':'ROBUST_SELECTION_ONLY','M2M3_NEXT_ACTION':'REWRITE_TO_IMPLEMENTATION','NEXT_PHASE':'PHASE20E3' if acceptance['status']=='PASS' else 'REMEDIATION_REQUIRED'})
    print(block,flush=True)

if __name__=='__main__':main()
