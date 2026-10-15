"""Report observed intervention counts separately from scientific benefit."""
from common import *

def trajectory_comparison():
    records=[];summary=[]
    chosen=[1,100,500,1000,5000,10000,14999,17999]
    for sample in SCENES:
        group=reference(sample)['actual_group']
        paths={'M0':E1/f'metrics/{sample}_{group}_TRAINING_CURVE.csv',
               **{m:STAGE/f'logs/{sample}_{m}_CURVE.csv' for m in ['M1','M2']}}
        curves={m:{int(r['iteration']):r for r in rows(p)} for m,p in paths.items()}
        for left,right in [('M0','M1'),('M0','M2'),('M1','M2')]:
            counts=[];losses=[]
            for iteration in range(1,18000):
                a=curves[left][iteration];b=curves[right][iteration]
                na=int(a['gaussians'] if left=='M0' else a['Gaussian_count']);nb=int(b['Gaussian_count'])
                if na!=nb:counts.append(iteration)
                if float(a['rgb_loss'])!=float(b['rgb_loss']):losses.append(iteration)
                if iteration in chosen:
                    records.append({'sample':sample,'comparison':left+'_to_'+right,'iteration':iteration,
                                    'before_first_extra_pruning':True,'baseline_Gaussian_count':na,'comparison_Gaussian_count':nb,
                                    'Gaussian_count_difference':nb-na,'baseline_rgb_loss':a['rgb_loss'],'comparison_rgb_loss':b['rgb_loss'],
                                    'baseline_curve_sha256':sha(paths[left]) if iteration==1 else '',
                                    'comparison_curve_sha256':sha(paths[right]) if iteration==1 else '',
                                    'scope':'Descriptive pre-intervention trajectories; historical M0 camera draw sequence not archived'})
            summary.append({'sample':sample,'comparison':left+'_to_'+right,'first_count_difference_iteration':counts[0] if counts else None,
                            'first_rgb_loss_difference_iteration':losses[0] if losses else None,
                            'iterations_with_count_difference_before_18000':len(counts),
                            'cause_claim':'Observed before extra pruning; cannot attribute to M2M3 removal or Floor40'})
    write_csv(STAGE/'PRE_PRUNING_TRAJECTORY_COMPARISON.csv',records)
    write_csv(STAGE/'reports/PRE_PRUNING_TRAJECTORY_DIVERGENCE.csv',summary)
    return summary

def main():
    caps=rows(STAGE/'M2M3_NATIVE_CAPACITY_SUMMARIES.csv');events=rows(STAGE/'M2M3_PRUNING_EVENT_LOG.csv')
    assert len(caps)==6 and len(events)==30
    total=sum(int(r['floor_blocked_count']) for r in events if r['method']=='M2')
    activation='VERIFIED' if total else 'NOT_TRIGGERED'
    text=['# M2M3 真实剪枝与容量保护记录','',f'FLOOR40_ACTIVATION: **{activation}**。M2 全部 15 轮额外剪枝实际阻止删除合计 **{total:,}** 个。','',
          '下表从原生 `capacity_control/capacity_summary.json` 汇总，逐次参数/输入与来源 SHA256 见 `M2M3_NATIVE_CAPACITY_SUMMARIES.csv`；全部 30 次真实事件见 `M2M3_PRUNING_EVENT_LOG.csv`。原生标准 densify_and_prune 与额外 M2M3 路径分别记录，表内删除只计后者。','',
          '| 场景 | 方法 | 初始数 | 已观测峰值 | 最终数 | 额外轮数 | 请求总数 | 实际删除 | Floor 活动轮数 | Floor 阻止总数 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in caps:
        own=[e for e in events if e['sample']==r['sample'] and e['method']==r['method']]
        assert sum(int(e['removed_count']) for e in own)==int(r['total_removed'])
        assert sum(int(e['floor_blocked_count']) for e in own)==int(r['total_blocked_by_floor'])
        text.append('| '+ ' | '.join([r['sample'],r['method']]+[f"{int(r[k]):,}" for k in ['initial_count','max_seen_count','final_count','rounds','total_requested','total_removed','floor_active_rounds','total_blocked_by_floor']])+' |')
    text+=['','## 分清设置、计算、干预与收益','',
           '1. 参数开启：M2 capacity_control_mode=m2m3_floor，ratio=0.4，reference=peak。',
           '2. 逻辑计算：有候选时，在冻结活动窗内算全局 floor；候选为空会提前返回。',
           '3. 实际干预：只有 floor_blocked_count>0 才证明额外删除被拦截；进入活动窗不等于发生干预。',
           '4. 正向收益：即使有真实干预，也必须另由共同外观、前景覆盖与结构代理支持；计数下限不保证真实叶片完整。','',
           '每轮预算为当前容量的 3% 向下取整；floor = min(current, round(0.4*max_seen))。max_seen 是每次标准 densify/prune 完成后被记录的最大容量，而非算子内部任意瞬时峰值。标准增密在 15000 前结束，五次额外剪枝发生于 18000、21000、24000、27000、30000；本阶段 floor 参考在额外剪枝阶段不再增长。','',
           '标准 opacity/size 剪枝完全不受此 Floor40 保护。该模块没有局部容量、叶片分区、model-to-mask deficit、额外增密或双向目标。评分是 detach 后的全局最大值归一化与固定加权；只在 mask 支持不足的候选中排序。','']
    text+=['## 冻结预算与 Floor40 的可触发范围','',
           '额外剪枝阶段没有标准增密/剪枝。五轮每轮最多删除当前数量的 3%，因此即使每轮用满预算，末轮容量仍至少为 0.97^5 × 首轮剪枝前容量（整数取整只会增加这个下界）。把此下界与固定 peak 的 40% 比较，可以解释本协议是否有机会触及容量下限；这不是阈值调参或新增实验。','',
           '| 场景 | M2 首轮前容量 | peak 的40% | 五轮预算理论容量下界 | 预算本身保证不触及 floor |','|---|---:|---:|---:|---|']
    bounds=[]
    for sample in SCENES:
        own=sorted([e for e in events if e['sample']==sample and e['method']=='M2'],key=lambda e:int(e['iteration']))
        cap=next(r for r in caps if r['sample']==sample and r['method']=='M2')
        initial=int(own[0]['Gaussian_count_before']);floor=round(.4*int(cap['max_seen_count']))
        lower=initial*.97**len(own);guaranteed=lower>=floor
        bounds.append({'sample':sample,'M2_first_extra_round_count':initial,'floor_reference_count':int(cap['max_seen_count']),
                       'fixed_floor':floor,'five_round_budget_lower_bound':lower,'budget_guarantees_no_floor_intervention':guaranteed})
        text.append(f'| {sample} | {initial:,} | {floor:,} | {lower:,.2f} | {guaranteed} |')
    write_csv(STAGE/'M2M3_FLOOR_BUDGET_BOUNDS.csv',bounds)
    text.append('')
    if total==0:
        text+=['## 本次 Floor40 的可观测结论','',
               '**FLOOR40_BENEFIT = NO_EFFECT**。本次正式训练中 Floor40 没有改变任何额外删除决定，不能称为防止了实际过度剪枝，也不能宣称已验证结构保护收益。不得为了触发它而事后改变阈值、比例或种子。','',
               'M1/M2 即使有小的外观/容量/结构差异，也不能归因于一个从未改变删除决定的 floor。初始张量、RNG、相机序列一致仍不保证 CUDA 浮点原子操作的逐位确定性。本阶段没有通过选择有利运行消除这些变化。']
    else:
        text+=['## 本次 Floor40 的可观测结论','',
               'Floor40 在上述事件中实际改变了删除决定。是否减少了过度剪枝、改善覆盖或 Mesh 结构，需与 `M2M3_SCENE_SUMMARY.csv`、`M2M3_MESH_COMPARISON.csv` 和最终方法决策联合阅读；不能仅由阻止数量宣称正向收益。']
    trajectories=trajectory_comparison()
    text+=['','## 剪枝介入前的真实训练波动','',
           'PRE_PRUNING_TRAJECTORY_COMPARISON.csv 使用真实训练曲线，在固定的1、100、500、1000、5000、10000、14999、17999轮比较容量和 RGB loss；额外剪枝首次发生于18000轮。全部17999轮的首次差异另见 reports/PRE_PRUNING_TRAJECTORY_DIVERGENCE.csv。此为描述性诊断，不是新增显著性检验，不能用来重新选择运行。','',
           '| 场景 | 对照 | 首次容量差异轮次 | 首次 RGB loss 数值差异轮次 |','|---|---|---:|---:|']
    for r in trajectories:
        text.append(f"| {r['sample']} | {r['comparison']} | {r['first_count_difference_iteration']} | {r['first_rgb_loss_difference_iteration']} |")
    text+=['',
           '介入前已出现的差异不能归因于尚未执行的额外剪枝或 Floor40；同种子和相同源代码路径并不构成逐位确定性证明。历史 M0 未存逐轮相机序列，因此不把 loss 差异进一步拆解成具体随机源。M1/M2 的初始化与完整相机抽样序列有直接相等检验。','',
           '## 与 M0 的比较限制','',
           'M0 为冻结的历史 Seed0；禁用 M2M3 的分支与标准实现训练数学路径等价，且输入、相机、配置和 GPU 一致，但历史初始 RNG 哈希未保存。新方法都只运行一次。最终 Gaussian 数差不是单独的因果估计；真实额外删除数是在各自轨迹上可直接验证的机制效果。','',
           '训练时间包含新运行包装的逐步观察、输入校验和 I/O，历史 M0 观察开销不同；不将单次 wall-clock 差称为稳健加速。独立的共同渲染计时表提供本次推理效率 pilot，仍需保留共享主机负载与固定运行顺序限制。','',
           '启动诊断只使用丢弃的 20 iteration 模型，在虚拟 iteration=18000 调原生入口检查执行；其结果不是正式 30000 iteration 模型的证据。正式训练均从初始 sparse points 新建，未从诊断 checkpoint 恢复。']
    (STAGE/'M2M3_CAPACITY_EFFECT_REPORT.md').write_text('\n'.join(text)+'\n')
    write_json(STAGE/'reports/CAPACITY_ACTIVATION_DECISION.json',{'activation':activation,'floor_blocked_total':total,'benefit_if_no_intervention':'NO_EFFECT' if total==0 else 'REQUIRES_JOINT_EVALUATION','events':30,'M2_events':15})
if __name__=='__main__':main()
