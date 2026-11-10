"""Publish measured, explicitly provisional evidence while formal runs continue."""
import csv
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]
OLD=OUT.parent/'geometry_mesh_final_selection'
DUAL=OUT.parent/'geometry_rgb_dual_frontend'

def read(path):
    return list(csv.DictReader(path.open())) if path.is_file() else []

def number(value, digits=4):
    try:return f'{float(value):.{digits}f}'
    except (TypeError,ValueError):return '未完成/无统一值'

def table(rows, fields):
    return '|'+ '|'.join(label for _,label in fields)+'|\n|'+ '|'.join('---' for _ in fields)+'|\n'+''.join('|'+ '|'.join(str(r.get(key,'')) for key,_ in fields)+'|\n' for r in rows)

def actual_link(label,relative):
    return f'[{label}]({relative})' if (OUT/relative).is_file() else f'{label}尚未生成（不计为已交付文件）'

def leaf_table(mesh_ids=None):
    rows=[r for r in read(OUT/'comparison/leaf_region_mesh_metrics.csv') if r['mesh_stage']=='post' and (mesh_ids is None or r['mesh_id'] in mesh_ids)]
    fields=[('mesh_id','路线'),('upper_silhouette_iou','上部叶区代理IoU'),('upper_foreground_coverage','覆盖率'),('upper_background_geometry_ratio','外部几何比例'),('upper_enclosed_missing_area_px','内部缺失像素均值'),('leaf_interior3px_coverage','叶区内部覆盖')]
    return table([{k:r[k] if k=='mesh_id' else number(r.get(k),6) for k,_ in fields} for r in rows],fields) if rows else '该区域统计仍在计算。\n'

def paired_view_table(family,baseline,candidates,metrics):
    rows=[r for r in read(OUT/'comparison/paired_view_changes/PAIRED_VIEW_CHANGE_SUMMARY.csv')
          if r['family']==family and r['baseline_method']==baseline and r['candidate_method'] in candidates and r['metric'] in metrics]
    fields=[('candidate_method','对照路线'),('candidate_stage','实际阶段'),('metric','指标'),('mean_change','均值变化'),('change_p10','差值P10'),('change_median','差值中位'),('change_p90','差值P90'),('improved_view_fraction','方向改善视角比例')]
    return '\n同一208源视角的描述性配对变化（对照减基线）：\n\n'+table([{k:r[k] if k in ['candidate_method','candidate_stage','metric'] else number(r[k],6) for k,_ in fields} for r in rows],fields)+'\n视角来自同株、同序列、Seed0；分位数和改善比例描述局部变化，不是独立植株/重复训练的显著性。孔洞减少可能伴随过填充，不能单独按“改善比例”选算法。\n\n' if rows else ''

def main():
    gaussian=read(OUT/'comparison/ALL_GAUSSIAN_EXPERIMENTS.csv')
    mesh=read(OUT/'comparison/ALL_MESH_EXPERIMENTS.csv')
    byid={r['experiment_id']:r for r in gaussian}
    paired=[]
    base={int(r['source_id']):r for r in read(DUAL/'render_comparison/E10/per_view_metrics.csv')}
    for eid,path in [('P2',OUT/'P2_GGGS_Cleaning/surface_inputs/per_view_metrics.csv'),('E12',OUT/'E12_GGGS_EO/surface_inputs/per_view_metrics.csv')]:
        for row in read(path):
            sid=int(row['source_id']);reference=base[sid]
            assert row['gt_sha256']==reference['gt_sha256'] and row['mask_sha256']==reference['mask_sha256']
            for key,higher in [('full_psnr',True),('fg_mask_psnr',True),('ssim',True),('fg_mask_ssim_map',True),('fg_gradient_retention',True),('lpips_alex',False),('background_alpha_mean',False)]:
                diff=float(row[key])-float(reference[key])
                paired.append(dict(experiment=eid,stage=row.get('stage','raw'),source_id=sid,metric=key,
                                   change=diff,improved=(abs(float(row[key])-1)<abs(float(reference[key])-1)) if key=='fg_gradient_retention' else diff>0 if higher else diff<0))
    with (OUT/'comparison/E10_P2_E12_paired_changes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    changes=[]
    for eid,stage,key in sorted({(r['experiment'],r['stage'],r['metric']) for r in paired}):
        rr=[r for r in paired if (r['experiment'],r['stage'],r['metric'])==(eid,stage,key)]
        assert len(rr)==208
        changes.append(dict(experiment=eid,stage=stage,metric=key,mean_change=float(np.mean([r['change'] for r in rr])),
                            median_change=float(np.median([r['change'] for r in rr])),improved_fraction=float(np.mean([r['improved'] for r in rr])),views=208))
    (OUT/'comparison/E10_P2_E12_paired_summary.json').write_text(json.dumps(dict(status='complete',rows=changes,
        interpretation='Paired same208 source images/trueRAP masks. Correlated training views, single seed, no independent repeated-trial significance.'),indent=2))
    fields=[('experiment_id','模型'),('iterations','迭代数'),('num_gaussians','基元数'),('psnr_raw','Raw PSNR'),('psnr_corrected','Corrected PSNR'),('fg_psnr_raw','Raw真FG PSNR'),('fg_psnr_corrected','Corrected真FG PSNR'),('ssim','Raw SSIM'),('ssim_corrected','Corrected SSIM'),('lpips','Raw LPIPS'),('lpips_corrected','Corrected LPIPS'),('background_alpha','背景Alpha均值')]
    rows=[{k:(number(r.get(k)) if k not in ['experiment_id','iterations','num_gaussians'] else r.get(k,'')) for k,_ in fields} for r in gaussian if r['experiment_id'] in ['E10','E11','E12','P2']]
    e12=byid['E12']
    text='# E12 GGGS + EO：已完成30K的实测分析\n\n本实验训练、208视角raw/corrected评价和TSDF/Tetrahedra提取均已完成。本报告记录E12独立证据；结合所有正式终态与直接视觉复核的综合选择见[最终算法选择报告](FINAL_RECONSTRUCTION_AND_MESH_SELECTION.md)。\n\n'+table(rows,fields)
    text+='\nEO后真前景PSNR提高，但SSIM和LPIPS并未共同优于E10；不能只按24.7312dB选照片分支。背景Alpha均值0.3780→0.9102，统一灰背景自由视角直接查看也确认大片暗云仍存在。EO颜色矩阵独立于PLY，不能用raw模型冒充corrected外观。\n\n'
    text+='这里的全图光度GT来自实际03-SAM训练RGB，其Mask外背景保存为黑色；几何前端的完整背景上下文来自02-FFT，二者不能混称。EO corrected前景SSIM映射均值0.652224→0.683271，前景梯度保留0.731265→0.744480，支持已观测叶片纹理拟合有收益。全图SSIM下降同时包含黑背景颜色偏差，不等同于前景SSIM下降；全图LPIPS不是单独前景LPIPS。FG SSIM是完整图像win7的SSIM map在真实RAP内取均值，仍受局部边界窗口影响。\n\n'
    eo_partition=OUT/'comparison/EO_region_decomposition/summary.csv'
    if eo_partition.is_file():
        text+='由相同SSIM map的实际像素数量推导，背景SSIM均值E10 raw为0.982919，E12 corrected为0.841317；背景PSNR为30.6832/30.6124dB。前景对全图SSIM的平均贡献变化为+0.010382，背景为−0.098125，两者合计−0.087743。这里明确区分区域均值与像素加权贡献，不将全图SSIM下降解释为前景纹理下降，也不将数学分解解释为曝光偏移分量的因果实验。[逐视角前景/背景分解](comparison/EO_region_decomposition/per_view.csv)和[真实Mask/原始评分来源](comparison/EO_region_decomposition/PROTOCOL.json)完整保留。\n\n'
    text+='光度目标实际为`0.8 L1(EO(render),GT) + 0.2 [1-SSIM(EO(render),GT)]`，并加上与E10完全相同的depth-normal、NCC和multi-view geometry项。旧EO实现仅L1使用校正RGB，SSIM仍使用raw；本次SSIM也改为校正RGB。`use_decoupled_appearance=1`是ExposureCompensator自身的3×4逐视角模式，未增加另一个外观网络；未启用前景RGB/Mask/背景Opacity Loss或训练Pruning。参数与核心源码审计见[E12 CONTROLLED_AUDIT](E12_GGGS_EO/CONTROLLED_AUDIT.json)，一行补丁见[EO_OBJECTIVE.patch](E12_GGGS_EO/EO_OBJECTIVE.patch)。\n\n'
    selected=[r for r in mesh if r['mesh_stage']=='post' and r['mesh_id'] in ['M4_E10_GGGS_TSDF','P1','E12_TSDF','E12_Tetrahedra']]
    mf=[('mesh_id','Mesh'),('silhouette_iou','IoU'),('boundary_chamfer_px','Chamfer px'),('background_geometry_ratio','外部几何比例'),('hole_area','封闭缺失像素均值'),('surface_roughness','相邻面1−cos')]
    text+=table([{k:r[k] if k=='mesh_id' else number(r.get(k)) for k,_ in mf} for r in selected],mf)
    text+='\nE12 TSDF的轮廓/边界代理改善来自实际Mesh重投影，不是从PSNR推断；其封闭投影缺失面积也较M4降低。Tetrahedra仍有较多外部几何和缺失。灰色Mesh直接查看显示两条TSDF路线仍存在细尺度起伏，不能称叶片变成真实平滑表面。相邻面指标受曲率、三角形尺寸和拓扑影响，应与统一空间尺度PCA和法线证据一起解释。\n\n'
    text+=leaf_table(['M4_E10_GGGS_TSDF','P3_shared','E12_TSDF','E12_Tetrahedra'])
    text+='\n叶区采用同一RAP包围框上60%高度的近似区域，没有新增逐叶分割。E12的叶区轮廓相对M4也改善；P3共享范围的IoU/外部几何更好，但覆盖略低。不能仅根据整株盆体或单一轮廓指标宣布最佳薄叶Mesh。\n\n'
    text+='旧训练RGB亮度口径有135个候选；旧法线采样在双线性插值后没有再次归一化，保留的角度中位5.1935°→4.2760°、P90 27.6621°→19.3252°属于该旧口径，统一法线指标使用下述v2。相对深度误差中位0.001674→0.001553，而P90 0.012095→0.013565未改善；邻近3D基元尺度比中位70.65→100.44也变大。对应真实三角面、像素和世界坐标见[反光Mesh对应CSV](comparison/reflection_mesh_correspondence/E12_TSDF_spatial_correspondence.csv)。亮度变化可能含阴影、曝光和匹配误差，并未确认为独立镜面反光。\n\n'
    normal_rows=[r for r in read(OUT/'comparison/reflection_normal_v2/ALL_NORMAL_METRICS.csv') if r['brightness_candidate'].lower()=='true']
    if normal_rows:
        text+='法线v2重新归一化采样方向，零/非有限/无有效插值支持的法线不填成90°，另列有效数量；原始CSV与深度/匹配点保持不变：\n\n'+table(normal_rows,[('method','方法'),('queries','旧固定候选数'),('valid_crossview_normal_queries','有效法线数'),('normal_crossview_abs_angle_deg_median','v2中位°'),('normal_crossview_abs_angle_deg_p90','v2 P90°')])+'\n'
    raw_audit=OUT/'comparison/reflection_analysis/ORIGINAL_RGB_AUDIT.json'
    if raw_audit.is_file():
        groups=json.loads(raw_audit.read_text())['candidate_group_counts']
        text+=f'实际02-FFT原图核验保留{groups["raw_context_candidates"]}个亮度候选，旧训练RGB候选中{groups["masked_only"]}个仅因Mask边界亮度变化超过阈值；旧集合没有删除。[原始/训练RGB与同像素审计](comparison/reflection_analysis/ORIGINAL_RGB_QUERY_VALIDATION.csv)和[原图候选几何指标](comparison/reflection_analysis/ORIGINAL_RGB_CANDIDATE_GEOMETRY.csv)分别保留。\n\n'
    text+=f'真实30K Gaussian PLY：`{e12["gaussian_ply_path"]}`。\n\n真实Checkpoint：`{e12["checkpoint_path"]}`。\n\n曝光参数：`{e12["exposure_parameters_path"]}`。\n\n'
    text+='[全部Mesh实际路径](artifact_index/MESH_FILE_INDEX.md) · [208逐视角配对](comparison/E10_P2_E12_paired_changes.csv) · [统一真实RGB及局部图](comparison/visualizations/gaussian/) · [有限直接视觉查看记录](comparison/visualizations/VISUAL_REVIEW.json)。\n\n来源：本地完整训练、实际模型、208浮点渲染、RAP Mask与Mesh raycast；[固定官方GGGS源码](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/tree/23b418c2f0f19b30b251acad6fe13445dae968c7)。博查搜索HTTP403，当前实验数据来自真实文件，不以训练数据填充。\n'
    text+=paired_view_table('Gaussian','E10',['E12'],['fg_mask_psnr','ssim','lpips_alex','background_alpha_mean'])
    text+=paired_view_table('Mesh','M4_E10_GGGS_TSDF',['E12_TSDF','E12_Tetra'],['silhouette_iou','hole_area_px'])
    (OUT/'E12_ANALYSIS.md').write_text(text)
    cleaning=json.loads((OUT/'P2_GGGS_Cleaning/cleaning_complete.json').read_text())
    text='# P1–P4：真实提取与同Checkpoint对照\n\n各主提取和统一评价均已完成；正式模型、统一空间尺度证据与直接视觉复核共同支持的综合选择见[最终算法选择报告](FINAL_RECONSTRUCTION_AND_MESH_SELECTION.md)。\n\n'
    text+='P1保留官方Marching Tetrahedra、10步Binary Search与原/post模型。Binary Search明显降低原始三角面的局部不稳定，但post IoU0.6284、外部几何0.2753弱于M4的0.8825/0.0602；来源0/53直接灰色查看仍有盆周大面积外部面。当前不支持P1优于TSDF，也不根据较少boundary edges认定薄叶无孔洞或无双层。\n\n'
    text+=f'P2从{cleaning["original_gaussians"]:,}个Gaussian删除{cleaning["removed_gaussians"]:,}个（{cleaning["removed_fraction"]:.2%}），保留{cleaning["cleaned_gaussians"]:,}个；依据208视角Mask支持、前/遮挡深度、局部邻域、尺度与拉长程度，保护叶缘，未按颜色/低Opacity单独删除。背景Alpha0.3780→0.3317，Raw真FG PSNR19.4605→19.4748。直接自由视角确认上方黑云明显减小，但标记牌射线和叶间黑块仍存在。\n\n'
    text+='必须区分两个答案：黑云明显减小但未全部消失；Mesh整体仅有微小变化。P2 TSDF IoU0.8829、Chamfer34.63px相对M4的0.8825/35.82px变化很小，投影内部缺失545.05→544.86px几乎不变；P2 Tetrahedra post IoU0.6266略低于P1。不能宣称清理同时解决Mesh孔洞。\n\n'
    text+='P3使用E10真实GGGS生成左右立体视图，官方DLNR权重/10次迭代/LR检查和TSDF保持；无重新3DGS训练。官方baseline4–20倍截断把深度上限设为0.755529，208输入支持只剩0.3912；补充共享上限1.569246保留0.9219支持。两条实际Mesh均保留，不能用共享范围结果替换原始失败倾向。额外DLNR先验的收益与GGGS表示收益分开。原生及补充详见[P3实际运行报告](P3_GS2Mesh/RUN_REPORT.md)。\n\n'
    mf=[('mesh_id','路线'),('silhouette_iou','IoU'),('boundary_chamfer_px','Chamfer px'),('foreground_coverage','覆盖率'),('background_geometry_ratio','外部几何比例'),('hole_area','封闭缺失像素均值')]
    selected=[r for r in mesh if r['mesh_stage']=='post' and (r['mesh_id'].startswith(('P1','P2','P3','P4')) or r['mesh_id'] in ['M4_E10_GGGS_TSDF','M5_E11_2DGS_TSDF'])]
    text+=table([{k:r[k] if k=='mesh_id' else number(r.get(k)) for k,_ in mf} for r in selected],mf)
    text+='\n同一208视角叶区补充：\n\n'+leaf_table()
    text+='\n叶区是RAP包围框上60%高度的代理，不是逐叶语义真值。内部缺失连通区域先在完整RAP上识别，再按区域统计，切分线不会生成额外“孔洞”。P3共享范围上部IoU0.9245、内部覆盖0.9930；E12 TSDF为0.9167/0.9963，说明P3轮廓更紧同时存在覆盖代价。\n\n'
    text+='[208可见叶区的统一空间尺度起伏](comparison/leaf_visible_surface/ALL_REGION_SURFACE_METRICS.csv)与[原图/训练RGB/法线采样口径审计](comparison/SURFACE_AND_REFLECTION_AUDIT.md)已补充。固定source131/157灰色窗口中P3的局部面较平顺，但不能代表全表面：至少2源ID组支持的D/64同向PCA起伏中位较低，P90仍高于M4。M4/P2在这些窗口很接近，E12仍有脊状起伏；这不认证其为真实叶脉或错误反光凹凸。\n\n'
    layers=[r for r in read(OUT/'comparison/same_ray_mesh_layers/ALL_MESH_LAYER_METRICS.csv') if r['mesh_stage']=='post' and r['group']=='all_fixed_queries' and r['mesh_id'] in ['M4_E10_GGGS_TSDF','P1','P2_TSDF','P2_Tetrahedra','P3_shared','P4_median_mask_alpha','E12_TSDF','E12_Tetrahedra']]
    if layers:
        text+='真实同射线薄表面补充：24,375条固定RGB匹配源射线来自69个源相机，源/目标并集195张照片；不是195个源视角，也不替代主208视角评价。\n\n'
        fields=[('mesh_id','路线'),('mesh_hit_fraction','查询命中率'),('near_front_4vox_layers_mean','前方4voxel层数均值'),('near_front_ge3_fraction_hit_queries','命中中近层≥3比例'),('close_same_facing_pair_fraction_hit_queries','近同向配对比例')]
        text+=table([{k:r[k] if k=='mesh_id' else number(r.get(k),6) for k,_ in fields} for r in layers],fields)
        joint=[r for r in read(OUT/'comparison/same_ray_mesh_layers/MATCHED_FIRST_PAIR_SPACING.csv') if r['reference_mesh_id']=='M4_E10_GGGS_TSDF' and r['group']=='all_fixed_queries']
        if joint:
            text+='\n同一查询同时具有相反平行首面配对时的法线投影间距：\n\n'
            fields=[('candidate_mesh_id','对照路线'),('joint_pair_queries','共同查询数'),('joint_pair_fraction_all_queries','共同有效比例'),('reference_joint_gap_voxels_median','M4间距voxel中位'),('candidate_joint_gap_voxels_median','对照间距voxel中位'),('paired_candidate_minus_reference_gap_voxels_median','逐射线差值中位')]
            text+=table([{k:r[k] if k in ['candidate_mesh_id','joint_pair_queries'] else number(r.get(k),6) for k,_ in fields} for r in joint],fields)
        text+='\nP1/P2 Tetrahedra局部近层较多，M4/P2 TSDF的相对前后间距也可能更大；不能把近层少单独等同于薄叶更准确。原图source0/79/157与实际有符号面交点保留为目的选取的局部示例，多层可来自卷曲和遮挡，不认证错误双层；首面间距也不是物理叶厚。[全部实际交点](comparison/same_ray_mesh_layers/PROTOCOL.json)、[共同查询配对CSV](comparison/same_ray_mesh_layers/MATCHED_FIRST_PAIR_SPACING.csv)和[直接实际射线图](comparison/visualizations/same_ray_layers/PROTOCOL.json)完整记录。\n\n'
    extra=OUT/'comparison/P3_leaf_tip_spatial/complete.json'
    if extra.is_file():
        fragment=json.loads(extra.read_text())
        text+=f'实际source157叶尖上方有{fragment["candidate_extra_pixels"]:,}个距RAP>3px的额外表面投影像素，对应'+', '.join(str(r['faces'])+'面' for r in fragment['component_records'])+'的两个原生组件，均与最大Mesh组件分离。原始02-FFT图保留其背景上下文；不是用置黑训练GT证明原图无几何。[世界坐标/实际面ID](comparison/P3_leaf_tip_spatial/source157_extra_surface_world_triangles.csv)及[原始背景窗口](comparison/P3_leaf_tip_spatial/source157_original_RGB_context.png)保留。这是观察后选定的局部诊断，不能当成全视角无偏发生率；原Mesh未修改。\n\n'
    text+='\nP4 Expected换Median确有收益（基础post IoU0.6183→0.6680），同一Median的Alpha>=0.5几乎无变化，符合E11背景Alpha接近1，Alpha不能充分筛背景的事实。0.75/1/1.25倍voxel/truncation中较粗1.25倍代理更好，但须检查叶尖细节，不能只按较低缺失面积选择过度平滑。\n\n'
    decomp=OUT/'comparison/P4_DECOMPOSITION.json'
    if decomp.is_file():
        info=json.loads(decomp.read_text());fields2=[('variant','方案'),('region','区域'),('source_depth_valid_fraction','源深度有效率'),('source_valid_but_raw_missing_fraction','有效深度但raw缺失'),('source_invalid_and_raw_missing_fraction','无效深度且raw缺失'),('post_removed_visible_fraction','仅post删除可见覆盖')]
        text+=table([{k:r[k] if k in ['variant','region'] else number(r[k],6) for k,_ in fields2} for r in info['summary']],fields2)
        text+='\n有效源深度仍可在融合/重投影后缺失：这里不能单独归咎于TSDF，因为有限正深度也可能几何错误。raw→post可见覆盖变化隔离组件删除造成的轮廓缺失，但仍可能删除前方表面而由更远几何保留轮廓；因此不能仅以覆盖几乎不变认定post几何影响很小。本次post不做平滑。上60%RAP内部是叶区近似，未声称为逐叶语义分割。\n\n'
    else:text+='[逐像素阶段分解](comparison/P4_DECOMPOSITION.json)仍在计算，尚不预填来自深度/融合/post的贡献。\n\n'
    local=OUT/'comparison/P4_local_failure/complete.json'
    if local.is_file():
        local_info=json.loads(local.read_text())
        text+=f'\nsource53的固定叶片纹理窗口中有{local_info["dark_foreground_pixels"]:,}个真实RAP前景像素满足GT亮度>0.15而E11亮度<0.01；其Alpha中位数{local_info["native_alpha_median"]:.6f}，Expected/Median源深度均为有限正值。四份Expected/Median raw/post对这些黑像素覆盖均约99.9703%，所以这里主要是错误暗色/遮挡表面，不能把黑色区域当成孔洞。该区域Median与E10表面深度的相对差异中位{local_info["E11_median_vs_E10_surface_relative_depth_median"]:.2%}，这是模型间不一致而非扫描误差。Expected/Median post分别有{local_info["expected_post_farther_gt1percent_common_hit_fraction"]:.2%}/{local_info["median_post_farther_gt1percent_common_hit_fraction"]:.2%}共同命中像素比raw更远1%以上，说明组件删除会改变遮挡深度，即使轮廓近乎不变。见[同像素世界坐标CSV](comparison/P4_local_failure/source53_same_pixel_spatial.csv)和[24张原生RGB与raw/post同窗口图](comparison/visualizations/P4_stage_visuals/PROTOCOL.json)。\n\n'
    post_depth=OUT/'comparison/P4_post_depth/complete.json'
    if post_depth.is_file():
        info=json.loads(post_depth.read_text());fields3=[('variant','方案'),('region','区域'),('removed_surface_without_losing_silhouette_fraction','轮廓仍在且post深度更远一voxel'),('post_depth_farther_gt1percent_region_fraction','post更远1%'),('raw_visible_post_missing_fraction','仅轮廓新增缺失')]
        text+='全部208视角的前方组件删除审计：\n\n'+table([{k:r[k] if k in ['variant','region'] else number(r[k],6) for k,_ in fields3} for r in info['summary']],fields3)
        text+='\n深度变远说明删除了原遮挡三角面，不说明删除正确；[原始逐视角证据](comparison/P4_post_depth/all208_per_view.csv)保留float32 raycast容差诊断。\n\n'
    text+='E11 source53局部原生RGB也有黑色楔形/条带，说明该视角的问题不能全部归因于Mesh后处理。必须同时看RGB、Alpha、Expected/Median Depth及raw/post，不能以某一整株纹理截图认证原版2DGS更适合薄叶。\n\n所有边界边数是拓扑边界；开放叶缘和RAP中的空白区域不计为本轮“封闭缺失投影”代理，空白区域尚未独立认证为真实物理孔洞。该代理是至少9像素且不接触RAP外缘3px带的缺失连通区域，单位为图像像素，绝不是物理3D孔洞面积。整株表外部几何字段沿用旧定义：Mask外被Mesh覆盖的像素/全部Mask外像素；叶区补充使用Mask外Mesh像素/该区域全部Mesh像素，两种分母明确区分。\n\n[全部实际Mesh](artifact_index/MESH_FILE_INDEX.md) · [原始/清理/删除Gaussian](artifact_index/GAUSSIAN_PLY_INDEX.md) · [同像素反光Mesh对应](comparison/reflection_mesh_correspondence/) · [直接视觉查看范围](comparison/visualizations/VISUAL_REVIEW.json)。\n\n来源：真实本地文件和统一208视角评价；[GGGS官方](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids)、[GS2Mesh固定源码](https://github.com/yanivw12/gs2mesh/tree/560f3e8a2349aaad24fe9444ef1ca2770aeec5ab)、[2DGS官方](https://github.com/hbb1/2d-gaussian-splatting)。博查HTTP403不可用。\n'
    text+=paired_view_table('Mesh','M4_E10_GGGS_TSDF',['P1','P2_TSDF','P2_Tetra','P3_native','P3_shared','P4_median_mask_alpha'],['silhouette_iou','hole_area_px'])
    (OUT/'P1_P4_ANALYSIS.md').write_text(text)
    text='# 全历史比较：证据台账\n\n历史E1–E11和M1–M5真实产物均保留；统一表以每行实际status、评价范围和文件为准。E1–E14及P2/E14各保存阶段均已使用实际208图像ID与相同GT/RAP文件哈希重算。E2/E3/E4/E6的旧数值/FG定义独立保留，历史原生相机与几何输入差异仍在表中记录，不能因评价帧统一而冒称严格受控训练比较。\n\nE6旧camera_runtime缺少Fx/Fy/Cx/Cy字段，已依据其原存FoV与投影矩阵恢复到新的评价元数据；208个原W2C/P/fullP/相机中心逐项未改。没有将E10修正接口施加给E6，也没有改写旧模型或旧元数据。见[相机字段恢复证明](comparison/history_uniform/E6/CAMERA_FIELD_RECOVERY_AUDIT.json)。\n\n'
    text+=table(gaussian,[('experiment_id','实验'),('frontend','前端'),('geometry_rgb','几何RGB'),('camera_interface','相机接口'),('iterations','步数'),('psnr_raw','Raw PSNR'),('psnr_corrected','Corrected PSNR'),('evaluation_scope','评价范围'),('status','实际状态')])
    text+='\nE1–E4包含早期Ω/EO/Foreground组合；E5/E6是旧前景几何输入，E7–E10恢复背景并修正接口。新增均固定标准VGGT和修正208相机，但QGS/2D-SuGaR保留原生损失/训练阶段及额外先验，不能假定全部损失权重相同。没有同株扫描，因此全部几何排序是多视角一致性/轮廓代理。\n\n'
    text+='统一光度评价使用03-SAM保存的训练RGB及真实RAP Mask；其背景大部分为黑色。前端几何RGB恢复02-FFT的背景上下文，新增反光诊断另用实际02-FFT原图核验候选。这三类RGB的来源与目的分开记录，不能把全图PSNR当成未分割场景的照片重建精度。[实际RGB源文件审计](comparison/reflection_analysis/ORIGINAL_RGB_SOURCE_FILES.csv)保留原图/训练图路径与测量差异。\n\n'
    text+='[统一Gaussian CSV](comparison/ALL_GAUSSIAN_EXPERIMENTS.csv) · [保留旧Gaussian指标](comparison/LEGACY_GAUSSIAN_METRICS.csv) · [统一Mesh CSV](comparison/ALL_MESH_EXPERIMENTS.csv) · [逐视角渲染](comparison/rendering_metrics.csv) · [Mesh逐视角](comparison/mesh_metrics.csv) · [全部实际模型](artifact_index/ALL_ARTIFACTS.csv)。\n\n'
    text+='Fast3R/AMB3R历史前端比较已在独立[完整历史前端表](comparison/FRONTEND_HISTORY.csv)与[共同帧表](comparison/FRONTEND_HISTORY_MATCHED.csv)保留，不重新推理。下面为旧条件的共同208帧诊断；Confidence/q85未跨模型校准，自重投影/邻帧一致性也不是扫描精度，不与恢复背景后的VGGT数值冒称同条件。\n\n'
    frontend=read(OUT/'comparison/FRONTEND_HISTORY_MATCHED.csv')
    text+=table(frontend,[('method','历史前端'),('common_frame_count','共同帧'),('comparison_status','状态'),('foreground_coverage','FG覆盖'),('edge_coverage','叶缘覆盖'),('adjacent_depth_agreement_fraction','相邻深度支持'),('projection_overlap_fraction','相邻投影重叠')])
    text+='\n恢复背景后的标准/Ω独立表仍见[双前端真实208结果](../geometry_rgb_dual_frontend/frontend_comparison.csv)，不能将旧Fast3R/AMB3R几何RGB条件更名为新增实验条件。\n\n来源：真实本地模型/CSV/训练日志；博查HTTP403，未取得联网搜索结果；官方固定源码直接读取成功。\n'
    text+='\n[叶区与下部区域208分层统计](comparison/leaf_region_mesh_metrics.csv)及[逐视角证据](comparison/leaf_region_mesh_per_view.csv)避免只由盆体主导整株比较。原始208视角同时是本轮训练输入，没有独立留出照片；这些PSNR/SSIM/LPIPS衡量已观测视角拟合，自由视角仅有真实渲染外观和几何一致性证据。原始RAP中的空白区域排除在缺失前景代理之外，未逐一人工认证其是否为真实叶片孔洞。\n'
    text+='\n[实际全视角配对变化汇总](comparison/paired_view_changes/PAIRED_VIEW_CHANGE_SUMMARY.csv)、[逐视角原值及差值](comparison/paired_view_changes/PAIRED_VIEW_METRIC_CHANGES.csv)和[来源/适用范围](comparison/paired_view_changes/PROTOCOL.json)补充均值、中位、P10/P90及方向改善比例。它们保留全部208视角，不把相关视角当成独立重复试验；Gaussian到各自Mesh的深度差不具有共同扫描真值，因此未按该差值跨模型排名。实际ECDF已从保存的终态模型结果重算，提供PNG和PDF。最终综合选择及适用范围见[算法选择报告](FINAL_RECONSTRUCTION_AND_MESH_SELECTION.md)。\n'
    (OUT/'FULL_HISTORY_COMPARISON.md').write_text(text)
    for eid,directory in [('E13','E13_QGS'),('E14','E14_2D_SuGaR')]:
        r=byid[eid]
        root=OUT/directory
        state=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').is_file() else {'status':'waiting_for_preceding_native_stage'}
        content=f'# {eid}：原生训练/模型的实际证据\n\n'
        if r.get('status')=='model_and208_metrics_verified':
            content+=table([r],[('experiment_id','实验'),('iterations','原生迭代数'),('num_gaussians','原生基元数'),('psnr_raw','Raw PSNR'),('fg_psnr_raw','真FG PSNR'),('ssim','SSIM'),('lpips','LPIPS'),('training_minutes','原生训练min'),('peak_vram_gb','峰值GiB')])
            content+=f'\n真实原生模型：`{r["gaussian_ply_path"]}`。\n\n'
        else:
            content+='正式训练或统一终态评价尚未完成。此处不预填30K模型路径或性能，不使用其他实验冒充。\n\n```json\n'+json.dumps(state,ensure_ascii=False,indent=2)+'\n```\n\n'
        if eid=='E13':
            comparative=[row for row in gaussian if row['experiment_id'] in ['E10','E11','E13']]
            content+='相同修正208视角的终态外观比较：\n\n'+table(comparative,[('experiment_id','模型'),('psnr_raw','Raw PSNR'),('fg_psnr_raw','真前景 PSNR'),('ssim','SSIM'),('lpips','LPIPS'),('background_alpha','背景Alpha')])+'\n'
            qgs_mesh=[row for row in mesh if row['mesh_id']=='E13_QGS']
            content+='QGS原生TSDF的raw、共同组件后处理和官方top10后处理分别保留，避免把后处理差异当成表示收益：\n\n'+table(qgs_mesh,[('mesh_stage','阶段'),('vertices','顶点'),('faces','面'),('silhouette_iou','IoU'),('boundary_chamfer_px','Chamfer px'),('hole_area','内部缺失像素'),('surface_roughness','相邻面起伏代理')])+'\n'
            content+='原生Quadratic Primitive保留六个有符号尺度/曲率参数，普通XYZRGB中心点云不能代替。当前主运行采用官方Mip预设：depth_ratio=1、lambda_dist=10、lambda_normal=.015、增密阈值1，距离/法线起始分别3000/7000，增密至15K。多视角几何/NCC起始和终止均30K，官方严格分支在本次30K内没有贡献；无额外深度法线先验。\n\n'+actual_link('原生运行记录','E13_QGS/E13_RUN_SUMMARY.md')+'；[原生Viewer指南](E13_QGS/NATIVE_VIEWER_GUIDE.md)、[参数](E13_QGS/actual_config.yaml)、[相机和源码适配](E13_QGS/PREPARATION_REPORT.md)保留。终态几何排序使用统一Mesh和实际视角。\n\n'
            content+='官方法线项逐通道计算sum(1−normal*depth_normal)，等价于带权3−dot；当前wo_image_weight=True使用detach后的对数曲率权重，curvature_clamp_threshold=-1约束该权重而非基元几何曲率。额外的2λmean(weight)不贡献同一步参数梯度，但会改变日志标量；日志normal_loss不能直接解释成法线角误差。[实际公式与源码SHA](E13_QGS/NATIVE_NORMAL_OBJECTIVE.md)及'+actual_link('真实终态208图上的标量分解','E13_QGS/native_normal_objective_decomposition/complete.json')+'分别记录。原生核心未替换。\n\n'
            content+=actual_link('实际Checkpoint数值状态与训练稳定性图','E13_QGS/native_stability/NUMERICAL_STABILITY.md')+'；'+actual_link('首轮OOM与原生恢复记录','E13_QGS/RECOVERY_REPORT.md')+'。原生参数中的NaN原样保留，逐位来源核验通过不等于参数全部有限，也不等于几何质量正常。普通XYZRGB导出若排除无效中心，须另列原生索引映射，不用它代替六尺度QGS原文件。\n\n'
            content+='base预设两次真实OOM发生在11700和13301步，失败Checkpoint/PLY和实际成本单独保留。主Mip运行从相同VGGT初始化与208相机/RGB重新开始Seed0的30K，没有恢复base参数。配置变化及限制见[官方预设](E13_QGS/OFFICIAL_NATIVE_PRESETS.csv)、[适用范围](E13_QGS/NATIVE_PRESET_SCOPE.md)和[归档路径](E13_QGS/runs/base_expandable_oom_13301/RETAINED_FILE_PATHS.json)。不推广为所有QGS配置结论。\n\n'
            content+=actual_link('失败导出实际读取核验','E13_QGS/failure_exports/20261109T084120Z_1535250/VALIDATED_DELIVERY_STATUS.json')+'。该失败PTH的ZIP64偏移超过真实文件长度，Python和原生PyTorch均无法读取，不是可恢复Checkpoint；独立原生PLY的精确字节长度正确，作为失败状态模型保留。捕获阶段标志不替代文件读取核验。\n\n'
            retained=root/'retained_native_states/iteration_1000/NATIVE_EXPORT_PROVENANCE.json'
            if retained.is_file():
                capture=json.loads(retained.read_text())
                content+=f'主Mip实际1K被动捕获另保留原生PTH与六尺度PLY：{capture["primitives"]:,}个基元，{capture["nonfinite_native_primitives"]}个原生无效基元原样保留，所有PLY字段逐位等于实际PTH；SH活跃阶数为{capture["active_sh_degree"]}。这是中间模型，不能冒充正式30K；实际绝对路径见[1K来源核验](E13_QGS/retained_native_states/iteration_1000/NATIVE_EXPORT_PROVENANCE.json)与[全部原生文件索引](artifact_index/GAUSSIAN_PLY_INDEX.md)。\n\n'
        else:
            stage_rows=[row for row in gaussian if row['experiment_id'] in ['E11','E14_initial','E14_cluster_before','E14_cluster_after','E14','E14_refined']]
            content+='阶段分解采用实际保存模型和相同208相机/RGB/Mask：\n\n'+table(stage_rows,[('experiment_id','模型阶段'),('iterations','该阶段迭代'),('num_gaussians','基元数'),('psnr_raw','Raw PSNR'),('fg_psnr_raw','真前景 PSNR'),('ssim','SSIM'),('lpips','LPIPS'),('background_alpha','背景Alpha')])+'\n'
            native_mesh_rows=[row for row in mesh if row['mesh_id'] in ['E14_cluster_before','E14_cluster_after','E14_initial','E14_refined']]
            content+='各阶段真实Mesh及共同评价：\n\n'+table(native_mesh_rows,[('mesh_id','阶段'),('mesh_stage','Raw/Post'),('vertices','顶点'),('faces','面'),('components','组件'),('boundary_edges','拓扑边界边'),('silhouette_iou','IoU'),('foreground_coverage','覆盖'),('hole_area','内部缺失像素')])+'\n'
            content+='7K同迭代清理是实际DBSCAN的局部对照；30K原生训练和附加15K精修是两个独立保存的终态。精修后前景拟合有改善，但原生30K及精修的实际视角53仍存在明显结构错误，不能因局部纹理恢复宣称已恢复薄叶片几何。最终是否改善Mesh以同视角覆盖、内部缺失和灰色实际Mesh为准。\n\n'
            if (root/'refined_mesh_exact_weld_validation.json').is_file():
                content+='SuGaR原生OBJ保留836,610个顶点、1,588,063个面。Open3D的UV导入产生逐纹理缝重复顶点，第一次共同组件后处理没有得到有效post；已保留该尝试。仅对完全相同XYZ位置进行精确合并，几何PLY的所有1,588,062个三角面坐标逐位不变，新的共同post保留同样的面数。原始OBJ/MTL/纹理未改动。进一步实际核验定位到原始OBJ没有末尾换行，Open3D跳过最后一面；临时副本仅补换行即导入全部1,588,063面。该末面非退化，面积约1.4284226e-6未标定VGGT gauge²。原始OBJ仍保留全部面，统一评价的导入PLY计数为1,588,062；不把该导入细节当成算法孔洞收益。[拓扑恢复与计数](E14_2D_SuGaR/MESH_TOPOLOGY_RECOVERY.md)和[逐位三角坐标核验](E14_2D_SuGaR/refined_mesh_exact_weld_validation.json)可复查。\n\n'
                content+='实际15K原生Checkpoint已完成后，纹理导出因PyTorch可信本地Checkpoint加载默认值失败；从该真实15K恢复导出，没有重训。Refinement总耗时列包含原训练尝试、恢复导出和精确拓扑适配；15K训练峰值没有独立留存，主峰值字段留空，4.5541GiB仅是恢复导出的Torch allocated峰值，不用于声称训练成本低。[可信原生Checkpoint恢复](E14_2D_SuGaR/recovery_checkpoint_loader/RECOVERY.md)记录具体一行补丁和来源。\n\n'
            content+='依次保留Metric3Dv2 ViT-giant2深度/法线、官方逐图点轨迹中位尺度对齐到VGGT gauge、RAP范围深度引导初始化、原生2DGS30K、7K的DBSCAN清理、初始TSDF、SuGaR15K Mesh绑定Refinement。额外先验不是新的真实扫描，不把最终收益都归因于2DGS表示。原生法线先验损失为完整世界法线图上的均值，没有RAP/Alpha乘项；输入先验RGB保持E11已置黑训练图，RAP另限制初始化采样和Mesh融合。没有把它误称为前景专用法线损失，也未增加独立前景光度/Mask Loss。\n\n清理前/后7K模型及各自Mesh隔离退化基元清理的影响；终态30K→Refined比较附加15K优化与Mesh绑定阶段，仍同时改变表示/优化。与E11的整体比较包括先验、初始化、原生损失和Pruning的共同变化，没有单独禁用prior的新增正式消融，不能分配先验与损失的唯一因果百分比。\n\n[原生阶段保存指南](E14_2D_SuGaR/PREPARATION_REPORT.md)、'+actual_link('真实阶段结果','E14_2D_SuGaR/STATUS.json')+'及[真实文件索引](artifact_index/ALL_ARTIFACTS.csv)为准。带纹理OBJ必须保留其真实MTL/纹理依赖。\n\n'
        content+='[统一Gaussian指标](comparison/ALL_GAUSSIAN_EXPERIMENTS.csv) · [统一Mesh指标](comparison/ALL_MESH_EXPERIMENTS.csv) · [同像素空间对应](comparison/reflection_mesh_correspondence/) · [原生全尺寸RGB/Alpha/Depth/Normal与局部图](comparison/visualizations/gaussian/)。\n\n没有同株扫描；几何结论限于轮廓与跨视角代理。正式终态、全部阶段Mesh和文件索引已交付；综合结论见[最终算法选择报告](FINAL_RECONSTRUCTION_AND_MESH_SELECTION.md)，本报告保留该方法各阶段的测量与限制。\n\n'
        url='https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55' if eid=='E13' else 'https://github.com/prajwalcr/2d-sugar/tree/e62770ec6ab50076ebb6c0e3b4ac39f777f14d27'
        content+=f'来源：[固定官方源码]({url})及真实本地训练/模型/评价记录。博查HTTP403不可用；Context7解析没有返回该具体方法的匹配官方项目，官方源码直接核对成功。\n'
        content+=paired_view_table('Gaussian','E10',[eid,'E14_refined'] if eid=='E14' else [eid],['fg_mask_psnr','ssim','lpips_alex','background_alpha_mean'])
        (OUT/(eid+'_ANALYSIS.md')).write_text(content)
    print('Published current measured E12/P1-P4/history reports; no global-final claim',flush=True)

if __name__=='__main__':main()
