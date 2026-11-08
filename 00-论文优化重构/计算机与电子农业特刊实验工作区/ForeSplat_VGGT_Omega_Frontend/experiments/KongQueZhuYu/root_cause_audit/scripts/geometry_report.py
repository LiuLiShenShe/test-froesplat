"""Generate audit findings from actual saved measurements; historical runs stay read-only."""
from geometry_common import *

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])

def number(x,digits=2):return 'unavailable' if x is None else f'{x:.{digits}f}'

def main():
    out=AUDIT/'geometry_comparison';sources=camera_sources()
    summary=json.loads((out/'multiview_summary.json').read_text());trajectory=json.loads((out/'camera_trajectory_independent.json').read_text());sim=json.loads((out/'camera_trajectory_Sim3.json').read_text())
    with (out/'primary_depth_confidence_per_frame.csv').open() as f:depth_rows=list(csv.DictReader(f))
    with (out/'input_source_rgb_fingerprints.csv').open() as f:rgb_rows=list(csv.DictReader(f))
    with (out/'manual_GT_mask_boundary.csv').open() as f:gt_rows=list(csv.DictReader(f))
    with (out/'input_blur_detail_fixed_rois.csv').open() as f:detail_rows=list(csv.DictReader(f))
    with (out/'confidence_region_summary.csv').open() as f:conf_rows=list(csv.DictReader(f))
    protocol=json.loads((out/'input_audit_protocol.json').read_text())
    point_check=json.loads((out/'standard_saved_point_export_check.json').read_text())
    counts={}
    for method in ['Omega_depth','Standard_depth']:
        rows=[r for r in depth_rows if r['method']==method]
        counts[method]=dict(frames=len(rows),foreground_pixels=sum(int(r['foreground_pixels']) for r in rows),finite_positive_depth_pixels=sum(int(r['foreground_finite_positive_depth']) for r in rows))
    inventory=dict(scope='Saved complete208 source views only, no rerun/inference/BA, no GPU, no scan/GT scale registration',source_order=[source_id(n) for n in sources['Omega_depth']['names']],
        excluded_unreadable_RAP_source_ids=[44,206],original_rgb_payload_audit_ids=protocol['actual_audited_ids'],
        Omega_depth=dict(path=str(OMEGA_DATA),full_depth_frames=208,full_confidence_frames=208,saved_model_RGB_frames=208,saved_model_mask_frames=208,
            camera_storage=str(OMEGA_DATA/'per_frame/cameras.json'),pixel_transform_storage=str(OMEGA_DATA/'metadata/preprocess_transforms.json'),
            actual_input_wh=[384,688],original_wh=[2160,3840],raw_learned_point_head_maps_available=False,
            point_head_status='Unresolved: no saved original learned point-head maps available in audited artifacts; depth-derived local/world maps are algebraic unprojection'),
        Standard_depth=dict(path=str(STANDARD/'Standard_VGGT_Frontend'),full_depth_frames=208,full_confidence_frames=208,
            saved_model_RGB_frames=0,saved_model_mask_frames=0,replayed_model_mask_frames=208,replayed_model_RGB_frames=208,
            replay_provenance='Exact saved original encoded RGB/mask SHA256 checked against manifest for every208 frame; replay source pad/resize code on CPU. RGB visualization quantized to uint8; no original saved input tensor exists.',
            camera_storage=str(STANDARD/'Standard_VGGT_Dataset/sparse/0'),camera_storage_notes=sources['Standard_depth']['camera_storage'],
            actual_input_wh=[518,518],continuous_content_wh=[291.375,518],padding_original_ltrb=[840,0,840,0],original_wh=[2160,3840],
            pixel_transform_storage=str(STANDARD/'Standard_VGGT_Frontend/metadata/input_manifest.json'),
            raw_learned_point_head_maps_available=False,point_head_run=False,point_head_status='Producer inference.py explicitly uses aggregator/camera_head/depth_head only; saved points are depth unprojection',
            sampled_saved_point_check=point_check),primary_foreground_geometry_counts=counts,
        old_bundle_read_representation='Complete global_raw.PLY row-major XYZRGB restores final exported pointmaps losslessly when vertex count equals every unpadded input pixel. Each method first-frame three rawNPZ XYZ probes checked byte-equivalent; complete confidence maps and masks read. Native/hybrid final pointmaps already contain official SfMemory fusion; no extra fusion in this audit.')
    files=[OMEGA_DATA/'per_frame/cameras.json',OMEGA_DATA/'metadata/preprocess_transforms.json',STANDARD/'Standard_VGGT_Dataset/sparse/0/cameras.txt',STANDARD/'Standard_VGGT_Dataset/sparse/0/images.txt',STANDARD/'Standard_VGGT_Frontend/metadata/input_manifest.json',
        STANDARD.parents[2]/'vggt_foreground/inference.py',STANDARD.parents[2]/'vggt_foreground/preprocess.py',STANDARD.parents[2]/'run_standard_vggt_frontend.py']
    inventory['small_source_artifact_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()}
    write_json(out/'geometry_sources_inventory.json',inventory)
    def stat(method,region,offset,control='mask_finite'):
        return next(r for r in summary if r['method']==method and r['region']==region and r['source_offset']==offset and r['control']==control)
    lines=['# 保存前端几何的独立根因审计：KongQueZhuYu',
        '',
        '结论：当前实际标准 VGGT 前端的相机与跨视角几何明显弱于既有 Ω，足以形成训练前的上游风险。完整源帧/相机编号错配、重复相机和 sampled XYZ 导出数值失真未被发现；标准非中心像素变换遗漏确实存在，但不能单独解释当前大幅失配。Ω 也存在少数强烈相机跳转及叶缘不确定性，不能据此宣称它已经具有真值级精度。',
        '',
        '本轮所有数值来自原有保存预测、原有 RGB/Mask 与人工二维标注。仅使用 uv 管理的既有 `.venv-fast3r` CPU 环境，脚本设置 `CUDA_VISIBLE_DEVICES=""`。没有重新推理、训练、分割、曝光处理、平均点云融合或尺度配准，没有改变历史模型/图像/相机/预测。此前 Bocha 检索返回403；本报告对本地实验的判断依赖文件实测，官方资料及真实像素变换控制见 [02相机审计](02_camera_geometry_audit.md)。',
        '',
        '## 1. 先确定实际保存了什么',
        '',
        table(['输入/输出','Ω','标准 VGGT'],[
            ['完整源序列','208：0–209，44/206原RAP不可读','相同208原文件名及顺序'],
            ['原图','2160×3840','相同原RGB/Mask（训练输入全208 hardlink由provenance核验）'],
            ['模型尺寸','384×688，无padding/crop','518×518，原图左右各840px padding；有效portrait约291.375×518'],
            ['RGB预处理','归档保存了真实model_rgb','square→1024 bicubic→518 bilinear，原模型输入tensor未保存，本轮依据sourceSHA和代码重放'],
            ['Mask','208保存的真实model_mask','208 sourceSHA核验后重放single nearest→518，不新增Mask'],
            ['depth/confidence','208全分辨率原始数组','208全分辨率原始数组'],
            ['相机/K','per_frame/cameras.json中raw model W2C/K','raw数组未单独保存；完整COLMAP text恢复W2C和Kmodel，无BA'],
            ['learned point_head原始图','原归档中不可用，未把depth派生图当point_head','producer未调用point_head；保存的是depth反投影采样点'],
        ]),
        '',
        '`Standard_VGGT_Dataset/sparse/0/{cameras.txt,images.txt}` 是标准全208相机的具体路径。恢复使用 `K_model=A_recorded K_original`，按文件名恢复 COLMAP W2C；测量 RGB/Mask 像素位置使用真实 half-pixel A。全部相机 C2W/W2C 逆误差≤1.2e−7，编号均与输入顺序一致，所有方法没有重复源ID或连续完全相同相机。详见 [文件清单与小元数据SHA](geometry_comparison/geometry_sources_inventory.json)、[相机编号检查](geometry_comparison/camera_source_checks.json)。',
        '',
        f'标准保存的全部 {point_check["saved_points"]:,} 个 sampled XYZ 与对应原depth逐点复核：depth值不同数为0，重建XYZ误差p50={point_check["reconstructed_xyz_error_p50"]:.3g}、p90={point_check["reconstructed_xyz_error_p90"]:.3g}、最大={point_check["reconstructed_xyz_error_max"]:.3g}（模型任意尺度）。这验证了已保存点的反投影/相机导出代数链；不证明跨视角真实几何正确，更不验证未运行的point_head。[原始逐点检查](geometry_comparison/standard_saved_point_export_check.json)。',
        '',
        '两种原生depth在全部前景模型像素均有限且为正：Ω17,051,207/17,051,207；标准9,743,236/9,743,236。单视角覆盖100%不能替代跨视角一致性。总点数差异首先受有效portrait分辨率影响，不直接代表某一版表面更致密或更完整。[逐帧depth/confidence统计](geometry_comparison/primary_depth_confidence_per_frame.csv)。',
        '',
        '## 2. 完整208视图的双向、多offset几何验证',
        '',
        '所有方法先使用相同的 Ω 保存Mask像素，经 A⁻¹映到原图坐标作为查询，再各自测试模型坐标/Mask/有限性；每ROI每source最多1500点，按sourceID固定随机种子，全部208保存预测参与。source offset直接按原ID计算，不把缺44/206后的数组邻居假称gap1。两方向均测量。',
        '',
        '主支持率分母是全部共同前景查询，包括投影到画外、目标前景不重合和预测遮挡的不支持查询。source projected z-buffer采用1.005阈值；target projected source depth大于1.05×target depth单独记为“预测遮挡”。遮挡判定也可能来自错误几何，所以表中的全查询支持与raw-overlap误差保留失败证据，另存visible条件统计。误差除以target预测depth，不进行真实尺度修正。近邻低误差、远距高误差或重复/塌缩表面不能独立证明绝对几何准确。',
        '',
        table(['原source offset','双向pair数','Ω5%全查询支持','标准5%全查询支持','Ω raw重叠相对depth p50','标准 raw重叠相对depth p50'],[
            [off,stat('Omega_depth','foreground',off)['bidirectional_pairs'],number(100*stat('Omega_depth','foreground',off)['support_5pct_fraction'])+'%',number(100*stat('Standard_depth','foreground',off)['support_5pct_fraction'])+'%',number(100*stat('Omega_depth','foreground',off)['raw_overlap_relative_depth_error_p50_across_pair_median'])+'%',number(100*stat('Standard_depth','foreground',off)['raw_overlap_relative_depth_error_p50_across_pair_median'])+'%'] for off in [1,5,15,30,60]
        ]),
        '',
        'depth p50一栏是每pair像素误差中位数再对pair取中位数，不是所有点合并的总体误差，也不是相机重投影真值误差。下面保留分布尾部，而不是只报平均值：',
        '',
        table(['offset','Ωpair支持p10/p50','标准pair支持p10/p50','Ωraw depth像素p90的pair中位数','标准对应p90'],[
            [off,number(100*stat('Omega_depth','foreground',off)['per_pair_support_5pct_p10'])+'/'+number(100*stat('Omega_depth','foreground',off)['per_pair_support_5pct_p50'])+'%',number(100*stat('Standard_depth','foreground',off)['per_pair_support_5pct_p10'])+'/'+number(100*stat('Standard_depth','foreground',off)['per_pair_support_5pct_p50'])+'%',number(100*stat('Omega_depth','foreground',off)['raw_overlap_relative_depth_error_p90_across_pair_median'])+'%',number(100*stat('Standard_depth','foreground',off)['raw_overlap_relative_depth_error_p90_across_pair_median'])+'%'] for off in [1,5,15,30,60]
        ]),
        '',
        '相邻pair投影目标前景重叠Ω90.54%、标准51.10%；两者相邻source有效查询接近100%。标准差距并非简单来自缺少单帧depth，而包含相机/深度共同不能解释目标视图的情况。预测遮挡及visible分母、2/5/10%支持、原图cycle像素误差、全340组摘要与125,664行pair/ROI/control记录详见 [完整摘要](geometry_comparison/multiview_summary.csv)、[协议](geometry_comparison/multiview_protocol.json)。',
        '',
        '![多offset深度支持与误差](geometry_comparison/multiview_support_by_offset.png)',
        '',
        '![共同源像素投影残差](geometry_comparison/cross_view_depth_projection_maps.png)',
        '',
        '残差图采用全部共同源前景像素，颜色固定±25%相对depth；灰色表示没有目标正depth前景重叠。它不应用主表的随机query采样/z-buffer资格，因此图内dense支持率是独立展示控制，不能和主表1500query统计混为同一个数字。',
        '',
        '### 固定叶片/叶缘窗口与盆边',
        '',
        '原坐标窗口：leaf=[540,570,1080,1230]；leaf_edge=[510,630,720,1110]；pot_rim=[800,1680,1700,2000]。窗口由source0可见结构确定，固定在所有视图相同图像位置；不是跨视图跟踪同一物理叶片，也不是叶片实例GT。前景边界是共同384×688Mask的一像素内边缘，包含盆/蓝色参考物，不能称整株指标为“纯叶缘”。空窗口pair显式记录并排除pair分位数；全部共有前景支持分母不隐藏失败。',
        '',
        table(['相邻区域','Ω5%支持','标准5%支持','无共同参考像素pair/全部pair'],[
            [reg,number(100*stat('Omega_depth',reg,1)['support_5pct_fraction'])+'%',number(100*stat('Standard_depth',reg,1)['support_5pct_fraction'])+'%',str(stat('Omega_depth',reg,1)['no_reference_query_pairs'])+'/'+str(stat('Omega_depth',reg,1)['bidirectional_pairs'])] for reg in ['foreground_boundary_common_grid','leaf_window','leaf_edge_window','pot_rim_window']
        ]),
        '',
        '### 与原有Fast3R/AMB3R结果使用同一查询的参照',
        '',
        table(['保存结果','相邻5%全查询支持','pair支持p10/p50','raw相对depth p50'],[
            [method,number(100*stat(method,'foreground',1)['support_5pct_fraction'])+'%',number(100*stat(method,'foreground',1)['per_pair_support_5pct_p10'])+'/'+number(100*stat(method,'foreground',1)['per_pair_support_5pct_p50'])+'%',number(100*stat(method,'foreground',1)['raw_overlap_relative_depth_error_p50_across_pair_median'])+'%'] for method in sources
        ]),
        '',
        'Ω/标准使用原生depth；其它方法此主表使用最终world-point head在保存相机下的camera-z与目标worldmap。Base强head/camera gauge不一致仍是限制，低支持不能直接宣布其独立depth head失效；前阶段已实际保存的depth分支对照位于旧工程 `amb3r_depth_channel_comparison.csv`，本轮未重新模型推理。Native/Hybrid final worldmaps已经经过官方SfMemory融合，这里没有另行平均。208/208注册仅表示pipeline映射状态，不能推出像素准确度。',
        '',
        '## 3. 独立轨迹，再做有约束的source-ID Sim3诊断',
        '',
        table(['方法','相邻旋转p50/p90 (deg)','>90deg个数','平移step p90/自身camera范围','原fx p50 (px)'],[
            [r['method'],number(r['rotation_step_p50_deg'])+'/'+number(r['rotation_step_p90_deg']),r['rotation_step_gt90_count'],number(r['translation_step_p90_normalized'],3),number(r['fx_original_p50'],1)] for r in trajectory
        ]),
        '',
        '全部方法无文件名错位或重复相机。标准86→87/87→88分别179.37°/176.53°；Ω162→163/164→165分别164.81°/152.66°。原图照片显示相邻主体/背景布局连续，这支持将这些帧列为预测相机风险，但照片视觉相似本身不是相机GT；独立RGB对应验证提供另外一条证据。',
        '',
        '![独立保存轨迹](geometry_comparison/camera_paths_independent.png)',
        '',
        '每个独立轨迹图使用自身第一相机坐标与自身2.5–97.5%稳健范围，仅用于展示，未修改输出。旋转/焦距逐帧见 [逐源ID曲线](geometry_comparison/camera_rotation_and_focal.png)。',
        '',
        '所有208source名称严格配对，156帧用于稳健正尺度/正旋转Sim3拟合，52帧留作诊断。可靠性预先在代码显式约定：中心误差p50<0.10且p90<0.25×Ω范围、方向误差p50<15°且p90<30°、camera路径第二奇异值/第一>0.05、四份拟合尺度spread<0.20。该门槛是本轮相对一致性诊断，不是GT标准，也不能使Ω成为真值；留出帧存在时间相关性。',
        '',
        table(['方法对Ω','留出中心误差p50/p90 /Ω范围','留出方向误差p50/p90 (deg)','可靠全局相对Sim3'],[
            [r['method'],number(r['held_out_center_error_p50_normalized'],3)+'/'+number(r['held_out_center_error_p90_normalized'],3),number(r['held_out_orientation_error_p50_deg'])+'/'+number(r['held_out_orientation_error_p90_deg']),'supported' if r['reliable_similarity_comparison'] else 'Unresolved'] for r in sim
        ]),
        '',
        'Native AMB3R-SfM 是唯一通过上述全局相对门槛的其它保存轨迹（中心p50约5.60%范围，方向p50约7.89°）；标准、Fast3R及Hybrid不具备可靠的单个全局相似变换。没有为了对齐而逐段拟合再拼接，也没有把未通过拟合的点云叠加图当作公平真实尺度几何比较。[拟合参数、尺度稳定性、留出逐帧残差](geometry_comparison/camera_trajectory_Sim3.json)。',
        '',
        '![相对Sim3诊断，标准未通过/Native通过](geometry_comparison/camera_paths_Sim3_diagnostic.png)',
        '',
        '![原图大跳转附近检查](geometry_comparison/camera_failure_source_photos.png)',
        '',
        '## 4. 真实输入、人工Mask和低置信度区域',
        '',
        f'original JPEG源为 `/data/fj/F2DMAS/02-FFT/KongQueZhuYu/NNNN.jpg`，RAP为 `/03-SAM/KongQueZhuYu/crop_NNNN.png` 与 `mask_NNNN.png`。固定18 sourceIDs={protocol["actual_audited_ids"]}；包含5GT、12均匀代表帧及68/73诊断帧。Mask内RGBMAE最大{max(float(r["full_mask_rgb_mae"]) for r in rgb_rows):.5f}/255，前景内部完全相同RGB像素比例最低{100*min(float(r["interior_exact_rgb_pixel_fraction"]) for r in rgb_rows):.6f}%，median100%；边界MAE最大{max(float(r["edge_rgb_mae"]) for r in rgb_rows):.3f}/255。大体保留原色，但不写全部208original JPEG或所有边缘像素完全等同。本轮没有曝光改变。[18帧内容SHA/像素差](geometry_comparison/input_source_rgb_fingerprints.csv)。',
        '',
        '训练数据的全208RGB hardlink同源、实际GGGS mask使用情况由独立 [provenance文件](gaussian_analysis/experiment_provenance.json)记录；本报告不把模型前景Mask都存在误解为每个历史训练都实际加载了Mask。',
        '',
        table(['GT sourceID','Mask IoU','precision/recall','RAP→人工边界p50/p90原px','人工→RAP边界p50/p90原px'],[
            [r['source_id'],number(100*float(r['iou']))+'%',number(100*float(r['precision']))+'/'+number(100*float(r['recall']))+'%',number(float(r['rap_boundary_to_manual_p50_original_px']))+'/'+number(float(r['rap_boundary_to_manual_p90_original_px'])),number(float(r['manual_boundary_to_rap_p50_original_px']))+'/'+number(float(r['manual_boundary_to_rap_p90_original_px']))] for r in gt_rows
        ]),
        '',
        '5帧GT IoU中位96.47%，recall97.15%，precision99.65%；边界p50约4–5.7原px，source75人工→RAP边界p90约35.1px。因此已有前景分割大体可用，但细叶缘/漏片不能认为无误。GT同时包括盆和蓝块，不是叶片实例，也没有3D相机/表面真值。Mask边界误差不能解释标准和Ω共享输入却差距巨大；它仍会同时限制两条链的薄结构初始化。',
        '',
        '![实际原图与RAP、人工GT边界五视图](geometry_comparison/original_RAP_manual_GT_five_views.png)',
        '',
        '标准square padding保存完整视野，但portrait实际占用像素更少；RGB两次resize和Mask一次nearest具有相同连续像素仿射，离散插值对边缘的影响不同。真实 half-pixel变换遗漏与GGGS对主点处理已由02报告的CPU float-ramp控制独立验证（约3.20656原px）；本轮不修改历史K或声称这个小修复解决大失配。',
        '',
        '统一288×512原FoV、同前景内部测量Laplacianvariance，standard/Ω逐帧比值中位：整株0.742（18视图）、固定leaf窗口0.711（16非空视图）、leaf_edge窗口0.645（12）、盆边0.860（17）。窗口需>100有效像素；这说明标准输入重放的高频细节较少，但Laplacian也受纹理/遮挡/视距/照明影响，不能独立证实原视频运动模糊或把训练差异量化为模糊因果。[实际分辨率/细节统计](geometry_comparison/input_blur_detail_fixed_rois.csv)。',
        '',
        '低/高置信度控制使用每模型每source原FG的q25/q75，保留并记录ties，在同6000共同原坐标查询上选尾部，target只检查Mask和depth有限正数，不按target置信度另筛；固定18源视图及真实target±1/5/15，不冒充全208confidence-tail评价。跨模型tail选择不同，只比较各自低/高尾部趋势，不能认为两个conf值是共同校准概率。',
        '',
        table(['offset','Ω低q25→高q75的5%支持','标准低q25→高q75的5%支持'],[
            [off,number(100*float(next(r for r in conf_rows if r['method']=='Omega_depth' and r['region']=='source_confidence_low_q25' and int(r['source_offset'])==off)['support_5pct_fraction']))+'% → '+number(100*float(next(r for r in conf_rows if r['method']=='Omega_depth' and r['region']=='source_confidence_high_q75' and int(r['source_offset'])==off)['support_5pct_fraction']))+'%',number(100*float(next(r for r in conf_rows if r['method']=='Standard_depth' and r['region']=='source_confidence_low_q25' and int(r['source_offset'])==off)['support_5pct_fraction']))+'% → '+number(100*float(next(r for r in conf_rows if r['method']=='Standard_depth' and r['region']=='source_confidence_high_q75' and int(r['source_offset'])==off)['support_5pct_fraction']))+'%'] for off in [1,5,15]
        ]),
        '',
        '高置信度有实际的相对质量区分作用，但标准高置信度子集仍支持较弱，过滤不能替代几何注册。tail阈值、占比与ties见 [confidence_region_thresholds.csv](geometry_comparison/confidence_region_thresholds.csv)，完整pair/摘要与固定18scope见 [confidence_region_protocol.json](geometry_comparison/confidence_region_protocol.json)。',
        '',
        '## 5. 与独立RGB几何证据交叉核对',
        '',
        '本报告的depth consistency仍涉及模型输出之间的内部证据；不是GT。02报告另外使用同一RGB独立SIFT/RANSAC对应，未用预测depth来选择匹配：69组被接受的相邻pair预测F的SampsonΩ3.001px、标准23.531px（共同384×688），image-RANSAC自身约0.211px；offset5/15标准也更差。相同14near-pair子集的真实匹配目标RGB depth投影误差Ω8.013px、标准111.264px，p20 confidence control后Ω8.151、标准104.209px，证明不能只凭algebraic self-reprojection≈0把标准结果宣称几何正确。远距匹配数量少、重复纹理和最低12inlier可能产生假对应，主要依据近邻；69pair、14pair、410双向depth pair是不同统计样本，绝不混读中位数。',
        '',
        '同14相邻RGB pair参照：ΩSampson4.332、Fast3R16.193、Native7.117、Hybrid16.616px。Native相对camera轨迹通过Sim3、208/208注册均未自动转化为比Ω更高的像素精度。详见 [RGB-only全pair摘要](camera_projection_comparison/epipolar_summary.json)、[RGB/depth真实对应控制](camera_projection_comparison/independent_match_depth_projection_summary.json)、[其它保存前端同匹配参照](camera_projection_comparison/reference_epipolar_summary.json)。',
        '',
        '## 6. 可确认、排除和仍未解决的原因',
        '',
        table(['假设','状态','证据/边界'],[
            ['当前标准已保存相机/深度在此植物全序列上较弱','Confirmed','相同完整sourceID双向depth支持/误差、独立RGB极线/目标匹配投影、轨迹跳转共同支持'],
            ['源帧与相机文件名错配/连续重复相机是主因','Rejected','全部8套208相机 exactsource order，ID正确，duplicates均0'],
            ['标准采样XYZ反投影或COLMAP输出数值失真是主因','Rejected','全部1,252,383 saved点与depth/恢复camera逐点近FP32误差范围一致；不排除模型预测本身'],
            ['标准half-pixel A缺失/GGGS丢主点','Confirmed，但非充分解释','02 CPU真实变换与相机链验证；量级约几原px，不能解释大量23–111 sharedpx失配'],
            ['前景RGB被曝光改变/普遍被换成另帧','本次抽检未支持','18original02FFT/RAP内部近乎逐像素一致，训练全208同源hardlink；全208originalJPEG内容未逐像素复查'],
            ['Mask误差/薄叶缘缺失','Confirmed，非单独差距解释','5人工2D参考有边界与漏边误差，两模型共享RAP输入'],
            ['较小有效portrait和两次resampling降低细节','Confirmed为输入差异，几何因果Unresolved','真实尺寸/代码和统一FoV纹理统计；未重新进行同分辨率模型控制'],
            ['黑背景/208全局上下文或模型权重能力分别贡献多少','Unresolved','多个因素共变，本轮无新的模型推理因果控制'],
            ['point_head learned图已经被独立验证','Unresolved/标准未运行','当前Ω/标准保存的是depth链，不能伪造缺失头输出'],
            ['视线法线不是surface normals导致当前GGGS初始化失败','Rejected','根代理实际GGGS create_from_pcd只用points/colors，不读normals；详见总体训练/数据链报告'],
            ['Native或Hybrid全注册足以替代精确相机','Rejected为充分条件','新depth/相同RGB匹配仍有失配；Native只在全局相对轨迹较接近'],
            ['任何模型具有已验证公制尺度/绝对叶缘准确性','Unresolved','没有本轮相机/表面3DGT，无扫描配准；2DMask与内部几何指标不能证明测量精度'],
        ]),
        '',
        '本轮支持先定位并控制相机/几何条件，再解释GGGS训练指标；不能把重新换模型名称、更多初始点或self-reprojection≈0当作问题已经解决。参数、初始化/损失/渲染链的因果贡献由总体 ROOT_CAUSE_ANALYSIS 汇总，本报告不宣称已经将所有训练差距归因于单个网络模块。',
        '',
        '## 7. 复现与交付',
        '',
        '使用既有 uv 环境执行 `scripts/geometry_trajectory_audit.py`、`geometry_multiview_audit.py`、`geometry_input_audit.py`、`geometry_confidence_regions.py`、`geometry_camera_figures.py`、`geometry_projection_figures.py`、`geometry_report.py`；全部CPU，源码在本审计独立 scripts/目录。`geometry_refresh_summary.py`只从已经保存的pair CSV重建摘要、处理空ROI分位数，不读全量原始数组。大数组只按模型顺序缓存一次；标准重放RGB/Mask辅助cache位于/tmp，最终表/图/协议均存本持久化审计目录，不依赖/tmp作为最终结果。',
        '',
        '核心接口文件：[geometry_sources_inventory.json](geometry_comparison/geometry_sources_inventory.json)、[multiview_summary.csv](geometry_comparison/multiview_summary.csv)、[camera_trajectory_independent.csv](geometry_comparison/camera_trajectory_independent.csv)、[camera_trajectory_Sim3.csv](geometry_comparison/camera_trajectory_Sim3.csv)、[manual_GT_mask_boundary.csv](geometry_comparison/manual_GT_mask_boundary.csv)、[input_source_rgb_fingerprints.csv](geometry_comparison/input_source_rgb_fingerprints.csv)。所有主pair逐帧数据和图在 geometry_comparison/，历史前端Bundle与训练Dataset保持只读。',
        '']
    path=AUDIT/'03_frontend_geometry_audit.md';content='\n'.join(lines)
    # Historical report regeneration retains explicitly authorized later-control
    # interpretation updates, while rebuilding the old208 numerical tables.
    marker='\n## 后续定向控制更新：'
    if path.is_file() and marker in path.read_text():content+=marker+path.read_text().split(marker,1)[1]
    path.write_text(content)
    print('Saved-source inventory and 03_frontend_geometry_audit.md generated from actual measurements')

if __name__=='__main__':main()
