"""Render actual measured tables plus an explicit, source-bound scientific decision."""
from common import *

def table(headers,values):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in values])

def main():
    decision=read_json(STAGE/'reports/SCIENTIFIC_DECISION.json')
    permitted={'POSITIVE','MIXED','NEGATIVE','NO_EFFECT','INCONCLUSIVE'}
    assert decision['weighted_pruning'] in permitted and decision['floor_benefit'] in permitted
    assert decision['mesh_effect'] in permitted-{'NO_EFFECT'}
    assert len(decision['answers'])==15
    for filename,digest in decision['measured_source_sha256'].items():assert sha(STAGE/filename)==digest,filename
    values=rows(STAGE/'M2M3_SCENE_SUMMARY.csv');assert len(values)==9
    meshes=rows(STAGE/'M2M3_MESH_COMPARISON.csv');assert len(meshes)==18
    trainings=rows(STAGE/'M2M3_TRAINING_RUNS.csv')
    completed=[r for r in trainings if r['method']!='M0' and r['status']=='PASS' and r['completed_iterations']=='30000']
    assert len(completed)==6
    cap=read_json(STAGE/'reports/CAPACITY_ACTIVATION_DECISION.json')
    if cap['activation']=='NOT_TRIGGERED':assert decision['floor_benefit']=='NO_EFFECT'
    accept=read_json(STAGE/'reports/FINAL_ACCEPTANCE.json') if (STAGE/'reports/FINAL_ACCEPTANCE.json').exists() else None
    status=accept['status'] if accept else 'PENDING_ACCEPTANCE'
    block=f"""PHASE20E3 STATUS: {status}
REFERENCE_MESH: PASS
M2M3_WEIGHTED_PRUNING: {decision['weighted_pruning']}
FLOOR40_ACTIVATION: {cap['activation']}
FLOOR40_BENEFIT: {decision['floor_benefit']}
M2M3_MESH_EFFECT: {decision['mesh_effect']}
M2M3_MANUSCRIPT: REWRITE_TO_IMPLEMENTATION
TABLE7_EVIDENCE: PARTIAL
PHASE20F_READY: CONDITIONAL"""
    method=['# M2M3 方法与贡献决策','',decision['lead'],'','```text',block,'```','',
            '## 加权剪枝','',decision['weighted_reason'],'','## 全局容量下限','',decision['floor_reason'],'',
            '## Mesh 对照','',decision['mesh_reason'],'','## 可用于论文的贡献强度','',decision['manuscript_reason'],'',
            '本次只有三场景各一个 Seed0；历史 M0 复用基于禁用分支的源码/配置等价及已归档初始参数证据，不是逐位相同训练轨迹。实际触发与正向收益分别判定。没有独立三维 GT，不宣称真实几何精度改善。','',
            'R1 不在本阶段调参或重训；前景参考保持冻结 R0。论文按 M2M3_MANUSCRIPT_CORRECTION_SPEC.md 改写到真实加权额外剪枝与可选全局 floor；本阶段没有修改论文正文。后续若要双向 deficit、边界效用或局部 Floor40，必须另立开发与实验，不当作既有贡献。','',
            '下一阶段先按 PHASE20F_HANDOFF.md 恢复物理尺度及同株同时间的独立表型记录；本阶段交付后停止，不自动进入 Phase20F。','',
            '## 实测来源绑定','']
    method += [f'- `{k}` SHA256 `{v}`' for k,v in decision['measured_source_sha256'].items()]
    (STAGE/'M2M3_METHOD_DECISION.md').write_text('\n'.join(method)+'\n')
    report=['# ForeSplat Phase20E-3：真实机制消融与植物 Mesh 验证','',decision['lead'],'','```text',block,'```','',
            '## 实际完成的工作','',
            '复用预先冻结的三个 M0（WanNianQing1/B_R0/Seed0、CaoMei2/B_R0/Seed0、HongZhang/B-C共享C_R1/Seed0）；六个新增 M1/M2 从相同 sparse points 重新训练到30000 iterations，原生29999次 optimizer steps。没有使用短测试 checkpoint、换种子或挑选有利 checkpoint。先完成三场景 M0 Mesh 和 WanNianQing1 启动诊断，再按固定顺序完成正式训练。','',
            '禁用剪枝/容量路径与标准 2DGS 等价，故预算为6次而非9次；源码证据见 reports/BASELINE_EQUIVALENCE.json。新增模型的初始 RNG、六类 Gaussian 张量、实际 alpha 与全部30000轮相机序列在 M1/M2 之间逐项一致；历史四类初始参数哈希另作独立验收，历史完整 RNG 状态仍缺失。','',
            '所有模型采用同一物理 GPU1。共享主机/磁盘负载、输入校验和逐步观察影响 wall-clock，不能把本次与历史训练时间差直接解释成算法加速；共同推理计时另见 M2M3_RENDER_TIMING_SUMMARY.csv。环境、扩展版本、GPU UUID 与哈希均保留。时间范围、PyTorch显存峰值与设备采样的区别详见 RESOURCE_METRIC_SPEC.md；这些峰值不是无观察开销的纯训练内存。','',
            '## 主 TSDF 协议','',
            '主提取在首次 GPU Mesh 工作和质量观察前冻结：bounded TSDF，depth_ratio=0，训练相机全集，冻结训练 alpha<0.5 的 depth=0；depth_trunc=2×相机半径，voxel=depth_trunc/1024，sdf_trunc=5×voxel，num_cluster=1。主参数不依模型质量修改。保留 raw 与 post；SH0 仅用于融合颜色，外观评价使用原生全SH渲染。','']
    protocol=[]
    for sample in SCENES:
        x=read_json(STAGE/f'configs/{sample}_MESH.json')
        protocol.append([sample,len(x['integration_cameras']),f"{x['camera_radius']:.8g}",f"{x['depth_trunc']:.8g}",f"{x['voxel_size']:.8g}",f"{x['sdf_trunc']:.8g}"])
    report += [table(['场景','融合相机数','r','depth_trunc','voxel_size','sdf_trunc'],protocol),'',
               '单位均为任意 COLMAP 单位；面积仅为其平方。没有物理尺度、独立3D GT、真值 Chamfer/F-score 或厘米级表型误差。','',
               '## 共同外观与固定前景代理','',
               '204个逐视角结果来自68个固定验证相机、9个模型。各场景表内是逐视角算术均值，验证视角不作为独立训练重复。全图采用共同 canonical undistorted raw RGB；前景采用完全相同的冻结 R0 ROI。ROI不是人工GT；composited SSIM/LPIPS受掩膜边界及卷积窗口影响。','']
    report += [table(['场景','方法','Gaussian','Full PSNR','SSIM','LPIPS','ROI PSNR','Comp. SSIM','Comp. LPIPS','ROI coverage','Outside RGB energy','Outside alpha'],
                     [[r['sample'],r['method'],f"{int(float(r['Gaussian_count'])):,}"]+[f"{float(r[k]):.6f}" for k in ['PSNR','SSIM','LPIPS','foreground_PSNR','fixed_mask_composited_SSIM','fixed_mask_composited_LPIPS','foreground_coverage_proxy','outside_ROI_RGB_energy','outside_ROI_alpha_mean']] for r in values])]
    report += ['', '## 主 Mesh 结构与一致性','',
               '下表包括全部原始/后处理 Mesh。post组件数由固定top-1规则直接影响，不能把组件减少解释成真实精度提高；小组件可能包含叶片。Mesh/Gaussian深度差只表示提取一致性。退化比例采用冻结的近零面积诊断阈值，与原生删除调用的判据不等同；post仍可有非零比例，没有据此额外修补。','',
               table(['场景','方法','阶段','顶点','面片','组件','最大组件面占比','开边','非流形边','退化面比例','ROI可见覆盖','深度MAE'],
                     [[r['sample'],r['method'],r['mesh_stage']]+[f"{int(r[k]):,}" for k in ['vertices','triangles','connected_components']]+[f"{float(r['largest_component_triangle_ratio']):.6f}"]+[f"{int(r[k]):,}" for k in ['boundary_edge_count','non_manifold_edge_count']]+[f"{float(r[k]):.6f}" for k in ['degenerate_face_ratio','mesh_ROI_coverage_proxy','mesh_Gaussian_depth_MAE']] for r in meshes]),'',
               '包围盒、面积、法向代理、有效坐标比例、重投影P95和完整逐视角结果保留在 MESH_STRUCTURE_METRICS.csv、M2M3_MESH_COMPARISON.csv、metrics/。全部主Mesh可从真实PLY重新计算，最终独立重载证据位于 reports/MESH_INDEPENDENT_RELOAD.csv。','',
               '## 补充 TSDF 参数敏感性','',
               '预声明的0.5×、2× voxel仅作用三个M0；保持depth_trunc不变、sdf=5voxel，不重训且不替换主结果。不同分辨率引起结构统计变化不能被解释成真实准确率排序。','']
    sensitivity=[]
    for sample in SCENES:
        runs=read_json(STAGE/f'reports/{sample}_M0_MESH_SENSITIVITY.json')
        structural=rows(STAGE/f'metrics/{sample}_M0_MESH_SENSITIVITY.csv')
        for r in runs:
            raw=next((x for x in structural if x['variant']==r['variant'] and x['mesh_stage']=='raw'),None)
            sensitivity.append([sample,r['variant'],r['status'],raw['vertices'] if raw else 'FAILED',raw['triangles'] if raw else 'FAILED',raw['connected_components'] if raw else 'FAILED',f"{r['wall_seconds']:.2f}"])
    report += [table(['场景','补充参数','状态','raw顶点','raw面片','raw组件','秒'],sensitivity),'',decision['sensitivity_reason'],'',
               '## 十五项问题的实际回答','']
    for i,a in enumerate(decision['answers'],1):report += [f'{i}. {a}','']
    report += ['## 验收、恢复与边界','',
               ('最终真实验收状态：'+accept['status']+'；'+str(sum(x=='PASS' for x in accept['gates'].values()))+'/'+str(len(accept['gates']))+'项通过。完整证据见 reports/FINAL_ACCEPTANCE.json。') if accept else '验收尚未完成，当前文档为待验收状态。',
               '', '全部204行外观/前景/背景代理由真实浮点重新计算，见 reports/FLOAT_METRIC_RECOMPUTATION.json。独立CPU检查9个checkpoint/PLY与18个主Mesh；5868条受保护文件和冻结配置逐个SHA256复查。测试日志、真实训练、剪枝和提取日志保留。准备错误、受限诊断及负结果见 EXECUTION_DEVIATIONS_AND_FAILURES.md。','',
               '恢复路径、命令、环境与逐文件SHA256见 RECOVERY_AND_REPRODUCTION.md、LARGE_ASSET_INDEX.csv 和 reports/RECOVERY_COMMANDS.json。大型文件位于独立外部目录，Git索引不能代替数据备份。只提交阶段脚本、配置、报告、小型图件与指标。','',
               '本阶段没有修改R1、SAM3、掩膜、GT、COLMAP、冻结划分、2DGS/M2M3生产数学逻辑或论文正文；没有将旧pilot当作本次结果。旧Table7的四方法20序列均值不被本次三场景pilot替换，详见 TABLE7_EVIDENCE_UPDATE.md。交付后立即停止，不开展Phase20F。','',
               '信息来源：本仓库冻结源码、真实运行日志、共同浮点渲染及真实Mesh；官方接口参考列于 MESH_EXTRACTION_PROTOCOL.md。科研判定记录的源文件SHA256见 reports/SCIENTIFIC_DECISION.json。']
    (STAGE/'PHASE20E3_FINAL_REPORT.md').write_text('\n'.join(report)+'\n')
    write_json(STAGE/'reports/FINAL_STATUS.json',{'status':status,'formatted_status':block,'scientific_decision_sha256':sha(STAGE/'reports/SCIENTIFIC_DECISION.json'),'time':now()})
    print(block,flush=True)
if __name__=='__main__':main()
