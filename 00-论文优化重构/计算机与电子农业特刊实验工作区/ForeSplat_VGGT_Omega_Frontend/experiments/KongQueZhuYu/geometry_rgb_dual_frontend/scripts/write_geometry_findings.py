"""Completed frontend/interface findings; reconstruction conclusions remain open."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'frontend_comparison.csv').open()));by={r['method']:r for r in rows}
depth=list(csv.DictReader((ROOT/'geometry_comparison/depth_summary.csv').open()))
for row in rows:
    for region in ['leaf_window','leaf_edge_window']:
        selected=next(v for v in depth if v['method']==row['method'] and v['source_offset']=='1' and v['region']==region)
        row[region+'_depth_support_ratio']=selected['support_5pct_fraction']
with (ROOT/'frontend_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def value(name,key):return float(by[name][key])
labels=[('E1 geometry','Omega_208_historical'),('E5 geometry','Standard_208_historical'),('E7/E8 geometry','VGGT_Omega_Original_RGB'),('E9/E10 geometry','Standard_VGGT_Original_RGB')]
report=['# 完成的 208 帧前端与像素接口验证','',
    '本页只报告已经完成的真实前端推理和投影检查。E7–E10完整训练、统一Checkpoint重渲染及最终路线决策，以后续FINAL_ANALYSIS.md和STATUS.json为准。','',
    '所有208帧顺序为0–209排除44/206；两前端均通过uv在物理GPU1推理。几何图像调用既有geometry_rgb_adapter恢复Mask外02-FFT背景，Mask内03-SAM前景逐像素不变。训练图像与Mask均保持历史原文件，不重新分割。','',
    '|前端条件|FG极线 px|BG极线 px|源Depth到目标FG RGB px|5% FG深度支持|5%共同边界深度支持|相对Depth误差|>90°相邻跳变|','|---|---:|---:|---:|---:|---:|---:|---:|']
for label,name in labels:
    v=by[name];report.append(f'|{label}|{float(v["foreground_epipolar_error_shared_px"]):.4f}|{float(v["background_epipolar_error_shared_px"]):.4f}|{float(v["foreground_rgb_depth_error_shared_px"]):.4f}|{float(v["foreground_depth_support_ratio"])*100:.2f}%|{float(v["foreground_edge_depth_support_ratio"])*100:.2f}%|{float(v["depth_consistency_relative_error"]):.6f}|{int(v["rotation_step_gt90_count"])}|')
report+=['','极线和独立RGB重投影统一到384×688共享坐标；不能直接称为1080×1920训练像素。支持率使用共同原像素查询和完整查询分母，5%相对深度阈值，source-ID间隔1双向相邻帧。该表沿用根因审计的mask_finite条件；它描述过滤前有效前景Depth的跨视角支持，并非p20点云每个点的支持率或公制真值精度。前景/背景RGB匹配在四条件中完全相同，图像RANSAC选择不使用模型Depth。','',
    '|条件|原有p20叶缘保留率，native grid|同一共享叶缘查询保留率|相邻旋转P90|','|---|---:|---:|---:|']
for label,name in labels:
    v=by[name];report.append(f'|{label}|{float(v["leaf_edge_point_coverage"])*100:.2f}%|{float(v["shared_grid_leaf_edge_point_coverage"])*100:.2f}%|{float(v["rotation_step_p90_deg"]):.4f}°|')
report+=['','这里的叶缘为审计固定原坐标窗口510≤x<720、630≤y<1110中的Mask边界，不是人工逐叶实例真值。恢复背景改善Camera/Depth的同时，原有p20筛选对该窗口保留率下降，标准模型更明显。整体前景native保留率约80%来自p20定义本身，不能作为恢复了80%真实叶片几何的证据。阈值、Mask和融合均没有为某模型另调。','',
    'Ω保留balanced@512实际384×688，标准保留原square1024→518及其原有黑Padding。标准有效图像内容宽约291像素，Ω为384像素；本轮没有混入balanced标准模型。两前端仅运行既有Camera/Depth API，World Point Map为原Depth反投影，没有新增learned point head。Ω沿用原融合，标准沿用原分层采样。','',
    '|实际GGGS投影链|旧接口均值 px|修正版均值 px|修正版最大 px|','|---|---:|---:|---:|']
for model in ['Omega','Standard']:
    values={}
    for interface in ['original','corrected']:
        v=[x for x in csv.DictReader((ROOT/'camera_interface'/f'projection_{interface}.csv').open()) if x['model']==model]
        values[interface]=(sum(float(x['projection_mean_px']) for x in v)/len(v),max(float(x['projection_max_px']) for x in v))
    report.append(f'|{model}，各208个Camera|{values["original"][0]:.6f}|{values["corrected"][0]:.6f}|{values["corrected"][1]:.6f}|')
report+=['','修复覆盖COLMAP K、训练PIL缩放的半像素项、实际Camera Fx/Fy/Cx/Cy、完整投影矩阵，以及CUDA渲染和sample_depth前向/反向射线。薄表面合成检查包含平移/旋转相机、Alpha重心、深度采样及有限非零反向梯度。原优化器、原损失、增密代码未改。','',
    '对照文件身份检查通过：E7/E8同一PLY，E9/E10同一PLY；RGB/Mask同原文件，焦距与尺寸不变。Ω原COLMAP内参已经有正确A，因此K文本变化为0，修复作用在GGGS真实Camera/投影链。标准的半像素A修复使COLMAP主点变化3.206564原图像素。Ω图像/观测文件保留旧字节，避免pycolmap对四元数额外归一化；姿态参数差0。标准姿态差≤1.2e−16。','',
    '|独立目标RGB真实重投影|旧接口 pair median px|修正版 pair median px|','|---|---:|---:|']
matches=list(csv.DictReader((ROOT/'camera_interface/matched_RGB_reprojection_summary.csv').open()))
for model in ['Omega','Standard']:
    for region in ['foreground','common_foreground_boundary']:
        a=next(v for v in matches if v['model']==model and v['region']==region and v['interface']=='original')
        b=next(v for v in matches if v['model']==model and v['region']==region and v['interface']=='corrected')
        report.append(f'|{model} {region}|{float(a["median_pair_reprojection_p50_training_px"]):.6f}|{float(b["median_pair_reprojection_p50_training_px"]):.6f}|')
report+=['','最后一表为真实1080×1920训练坐标，源Depth通过目标runtime full P投到独立RGB对应点；同时包含前端几何残差与接口位移。共同边界包括植物和花盆边界，不冒称叶片实例精度。它表明接口修复降低了真实投影误差，但不能替代30K重建收益评价。','',
    '轨迹闭合gap已经记录，但首尾照片不是同一已知真值位姿，不将其直接当作绝对闭环误差。完整多间隔双向Depth、RGB匹配、参数、相机轨迹NPZ、全量未过滤/前景PLY和逐帧Point Map均保留。','',
    '来源：[完整前端比较](../frontend_comparison.csv)、[几何协议](protocol.json)、[实际投影参数](../camera_interface/)、[对照身份验收](../configs/pair_invariants.json)、两前端真实日志与模型输出。']
(ROOT/'geometry_comparison/GEOMETRY_FINDINGS.md').write_text('\n'.join(report)+'\n')
print('Completed geometry/interface findings recorded')
