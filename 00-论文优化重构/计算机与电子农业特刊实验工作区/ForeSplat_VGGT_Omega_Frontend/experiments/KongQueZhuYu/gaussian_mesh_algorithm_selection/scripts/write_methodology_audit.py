"""Summarize actual foreground, sampled surface and original-RGB audits."""
import csv,json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
def read(p):return list(csv.DictReader(p.open())) if p.is_file() else []
def table(rows,fields):
 return '|'+ '|'.join(label for key,label in fields)+'|\n|'+ '|'.join('---' for _ in fields)+'|\n'+''.join('|'+ '|'.join(str(r.get(k,'')) for k,label in fields)+'|\n' for r in rows)

def main():
 text='# 表面起伏与反光对应：实际评价口径审计\n\n这份报告补充已完成模型的证据。E13/E14终态尚未完成时不填性能，也不宣布全方法最终排名；后续实际模型加入后按真实CSV更新。\n\n'
 text+='统一光度GT是实际03-SAM训练RGB，Mask外保存黑色；前端几何RGB另恢复02-FFT背景上下文。对照图必须分别标出训练RGB和原始背景RGB。全图PSNR/SSIM/LPIPS与真前景PSNR/SSIM不是同一评估区域，不能把全图SSIM下降等同于叶片纹理下降。\n\n'
 gaussian=[r for r in read(OUT/'comparison/ALL_GAUSSIAN_EXPERIMENTS.csv') if r['experiment_id'] in ['E10','P2','E11','E12','E13','E14','E14_refined']]
 text+=table(gaussian,[('experiment_id','模型'),('iterations','实际步数'),('fg_psnr_raw','Raw真FG PSNR'),('fg_psnr_corrected','EO真FG PSNR'),('fg_ssim_map_raw','Raw FG SSIM map'),('fg_ssim_map_corrected','EO FG SSIM map'),('gradient_preservation','Raw FG梯度比'),('gradient_preservation_corrected','EO FG梯度比'),('status','状态')])
 text+='\nFG SSIM是统一win7完整SSIM map在真实RAP区域的均值，不是独立前景LPIPS。梯度比目标为1，过大也可能来自毛刺/噪声；不能无限追求更高。全部208输入同时参与训练，因此这些是已观测视角拟合指标。\n\n'
 eo_regions=read(OUT/'comparison/EO_region_decomposition/summary.csv')
 if eo_regions:
  text+='实际Mask内/外的光度分解：背景SSIM与背景PSNR根据已有全图/前景评分及真实像素数量推导，未重渲染或修改RGB。以下仍是208逐视角均值。\n\n'
  text+=table(eo_regions,[('method','模型'),('stage','颜色阶段'),('foreground_ssim','前景SSIM'),('derived_background_ssim','背景SSIM'),('derived_background_psnr','背景PSNR'),('foreground_ssim_full_image_contribution','前景对全图SSIM贡献'),('background_ssim_full_image_contribution','背景对全图SSIM贡献')])
  before=next(r for r in eo_regions if r['method']=='E10' and r['stage']=='raw')
  after=next(r for r in eo_regions if r['method']=='E12' and r['stage']=='corrected')
  fg_delta=float(after['foreground_ssim_full_image_contribution'])-float(before['foreground_ssim_full_image_contribution'])
  bg_delta=float(after['background_ssim_full_image_contribution'])-float(before['background_ssim_full_image_contribution'])
  text+=f'\nE10→E12 corrected的全图SSIM变化由前景贡献{fg_delta:+.6f}和背景贡献{bg_delta:+.6f}相加构成。背景SSIM下降占主导，前景结构拟合小幅改善；这不是曝光矩阵某个分量的因果隔离，也不是几何准确性的证据。EO矩阵对“精确黑输入”的偏移单独记录，不将其误标为实际背景RGB。[真实像素数与逐视角分解](EO_region_decomposition/per_view.csv) · [来源与推导规则](EO_region_decomposition/PROTOCOL.json)。\n\n'
 roughness=[r for r in read(OUT/'comparison/leaf_visible_surface/ALL_REGION_SURFACE_METRICS.csv') if r['mesh_stage']=='post' and r['radius_divisor']=='64' and r['neighborhood']=='same_oriented_normal' and r['region'] in ['upper_leaf_interior_supported','upper_leaf_supported_across2_id_bins']]
 text+='现有独立面积均匀表面采样点以原生208视角raycast深度核验可见性；在RAP包围框上60%且内缩3px的区域，至少2个视角支持才归入叶区代理。另列至少2个源ID分组支持；8个连续源ID组不是独立试验或已校准基线。局部PCA邻域半径为统一标准点云D/64，含真实曲率及邻近薄片，不能当成扫描距离误差。\n\n'
 text+=table(roughness,[('mesh_id','Mesh'),('region','可见区域规则'),('selected_queries','区域样本数'),('valid_PCA_selected_fraction','PCA有效比例'),('local_plane_rms_normalized_median','RMS/D中位'),('local_plane_rms_normalized_p90','RMS/D P90'),('effective_support_spacing','实际支撑间距')])
 text+='\n各Mesh在自己的面积上采样，不是相同解剖位置；采样数量上限与法线同向过滤都可能影响结果，应同时保留全邻域/同向邻域、D/32/D/64及有效数量。[全部区域记录](leaf_visible_surface/ALL_REGION_SURFACE_METRICS.csv) · [每个实际采样点的世界位置、面ID与208可见支持](leaf_visible_surface/PROTOCOL.json)。\n\n'
 layer_rows=[r for r in read(OUT/'comparison/same_ray_mesh_layers/ALL_MESH_LAYER_METRICS.csv') if r['mesh_stage']=='post' and r['group']=='all_fixed_queries']
 if layer_rows:
  text+='固定图像匹配生成24,375条源相机射线，实际源相机69个，目标相机150个，源/目标RGB并集195张；主轮廓/孔洞评价仍为208视角。射线从同一修正K/W2C生成，直接读取真实Mesh所有三角面交点，按世界距离合并重复面边交点。前方4个统一voxel内的交点数与2个voxel内同向面配对分别记录。\n\n'
  text+=table(layer_rows,[('mesh_id','Mesh'),('mesh_hit_fraction','查询命中率'),('positive_distinct_layers_mean','全正深度层数均值'),('near_front_4vox_layers_mean','前方4voxel层数均值'),('near_front_ge3_fraction_hit_queries','命中中近层≥3比例'),('close_same_facing_pair_fraction_hit_queries','命中中近同向配对比例')])
  text+='\n多交点可以是真实叶片前后两面、卷曲或遮挡叶片，不认证为错误双层。TSDF前后表面较远时，第二面可能在4voxel窗口之外，近层数较少不能单独证明更薄、更准确。下面列首两个相反且近似平行面在首面法线方向的投影间距，法线点积≤−0.9、射线入射余弦绝对值≥0.1；每条路线的有效查询比例随几何改变。\n\n'
  pairs=[r for r in read(OUT/'comparison/same_ray_mesh_layers/FIRST_PAIR_SPACING_METRICS.csv') if r['mesh_stage']=='post' and r['group']=='all_fixed_queries']
  text+=table(pairs,[('mesh_id','Mesh'),('opposite_parallel_pair_fraction_all_queries','符合首面配对比例'),('parallel_plane_gap_voxels_median','法线间距voxel中位'),('parallel_plane_gap_voxels_p10','P10'),('parallel_plane_gap_voxels_p90','P90')])
  joint=[r for r in read(OUT/'comparison/same_ray_mesh_layers/MATCHED_FIRST_PAIR_SPACING.csv') if r['reference_mesh_id']=='M4_E10_GGGS_TSDF' and r['group']=='all_fixed_queries']
  if joint:
   text+='\n为避免不同有效样本集的条件中位数冒充相同位置比较，再使用两种方法均符合条件的同一查询ID进行配对：\n\n'
   text+=table(joint,[('candidate_mesh_id','对照Mesh'),('joint_pair_queries','共同有效查询'),('joint_pair_fraction_all_queries','共同有效比例'),('reference_joint_gap_voxels_median','M4共同集间距中位'),('candidate_joint_gap_voxels_median','对照共同集间距中位'),('paired_candidate_minus_reference_gap_voxels_median','逐条差值中位'),('paired_candidate_gap_greater_fraction','对照间距更大比例')])
  text+='\n首面配对仍可能对应不同解剖叶片；正常投影间距不是物理叶厚，也不是同株扫描误差。SIFT匹配偏向纹理丰富位置，未保证所有点为同一叶片或面积无偏采样。[实际全部交点/原生面法线](same_ray_mesh_layers/PROTOCOL.json) · [首面与共同查询规则](same_ray_mesh_layers/FIRST_PAIR_PROTOCOL.json) · [同射线实际示例](visualizations/same_ray_layers/PROTOCOL.json)。示例按P1/M4近层差异选取，不作为无偏发生率；M1的Ω gauge未注册到标准相机，未放入此标准同射线比较。\n\n'
 audit=OUT/'comparison/reflection_analysis/ORIGINAL_RGB_AUDIT.json'
 if audit.is_file():
  data=json.loads(audit.read_text())
  text+=f'固定24,375个图像匹配点的原图审计实际读取{data["actual_original_files"]}张参与匹配的02-FFT文件；旧135个训练RGB亮度候选中，原图阈值保留{data["candidate_group_counts"]["raw_context_candidates"]}个。剩余1个在source30/31距离RAP边界约2.28/2.69px处，原图亮度变化0.10975，训练Mask背景参与5px窗口后为0.12301，跨过0.12阈值。旧候选与匹配档案原样保留。\n\n'
  rows=[r for r in read(OUT/'comparison/reflection_analysis/ORIGINAL_RGB_CANDIDATE_GEOMETRY.csv') if r['RGB_candidate_group']=='raw_context_candidates']
  text+=table(rows,[('method','方法'),('queries','原图候选'),('valid_native_geometry_queries','有效深度对应'),('matched_relative_depth_error_median','相对深度中位'),('matched_relative_depth_error_p90','相对深度P90'),('normal_crossview_abs_angle_deg_median','法线中位°'),('normal_crossview_abs_angle_deg_p90','法线P90°'),('normal_metric_version','法线口径')])
  text+='\nE10→E12原图候选的深度中位有小幅收益，但P90变差；统一法线角中位和P90改善。亮度仍可能含曝光、阴影、匹配误差及叶片运动，不将其认证为镜面反光；没有同株扫描，不能用这些相对一致性数值宣布真实表面准确。\n\n'
 text+='法线v2在像素归一化后双线性采样，再归一化采样方向；零/非有限或无有效插值支持的法线不填成90°。每个几何有效数量与法线有效数量分开；原始CSV及旧未重新归一化的角度完整保留，新CSV还包含旧角度列。[v2协议及旧源SHA](reflection_normal_v2/PROTOCOL.json) · [原始RGB与训练RGB逐点核验](reflection_analysis/ORIGINAL_RGB_QUERY_VALIDATION.csv)。\n\n'
 shapes=[r for r in read(OUT/'comparison/native_primitive_shapes/ALL_NATIVE_SHAPE_SUMMARY.csv') if r['group']=='original_RGB_brightness']
 if shapes:
  text+='原生基元形状另行分解，避免把扁薄叶片当成针状尖刺。3D GGGS的最大/次大sigma作为针状轴比，次大/最小sigma单列扁平度；2DGS仅有两个切向sigma，QGS使用两个有符号切向跨度的绝对比，其第三有符号尺度是曲率而非厚度。每种方法使用自身全部有效原生基元的P99，仅为描述性分位，不是学到的伪影分类器。\n\n'
  text+=table(shapes,[('method','原生模型'),('queries','原图亮度候选'),('associated_finite_shape_queries','有效近中心对应'),('unique_associated_native_primitives','独立原生ID数'),('tangent_or_needle_aspect_median','切向/针状比中位'),('tangent_or_needle_aspect_p90','P90'),('threeD_flattening_second_over_smallest_median','3D扁平度中位'),('legacy_max_over_min_aspect_median','旧最大/最小比'),('aspect_above_own_P99_fraction_associated','超过自身轴比P99比例')])
  text+='\nE10/E12在现有原图候选的有效最近中心中没有超过自身针状比P99的基元，旧最大/最小比却很高，不能据此宣称高亮造成尖刺。P2在这些位置的形状与E10相同。最近中心距离限制为5个统一voxel，可能是邻近叶片，不能代替光栅器的实际贡献归因；相同中心在多个匹配点被重复采样，已列独立ID数。各方法的轴比定义不同，不作跨表示绝对名次。[原生参数/查询来源SHA与定义](native_primitive_shapes/PROTOCOL.json)、[全部形状及无效对应](native_primitive_shapes/ALL_NATIVE_SHAPE_SUMMARY.csv)和[同查询实际Mesh面对应](native_primitive_shapes/ALL_NATIVE_SHAPE_MESH_JOIN.csv)保留。模型及原始指标未修改。\n\n'
 mesh_highlights=[r for r in read(OUT/'comparison/reflection_original_RGB_mesh_metrics.csv') if r['RGB_candidate_group']=='raw_context_candidates']
 if mesh_highlights:
  text+='同一原图亮度候选直接对应真实Mesh三角面，另列以下指标。法线突变来自查询射线命中的原生三角面与其3px邻域，不使用继承的Gaussian旧插值角度。\n\n'
  text+=table(mesh_highlights,[('mesh_method','实际Mesh'),('queries','原图候选'),('mesh_hit_queries','真实面命中'),('mesh_missing_queries','查询未命中'),('mesh_vs_native_relative_depth_median','Mesh/原生深度差中位'),('mesh_local_normal_jump3px_deg_median','局部面法线角中位°'),('mesh_local_normal_jump3px_deg_p90','局部面法线角P90°')])
  text+='\nM4→E12 TSDF的局部面法线角P90降低，但中位略升；不能从Gaussian法线改善推出每项Mesh法线也改善。P3 shared的局部法线中位较低，但与E10原生深度的差较大。Mesh/原生深度差衡量方法分歧，没有哪一方自动是真值；查询未命中也不等于全部208视角孔洞面积。[原图分组/原生三角面汇总及源SHA](reflection_mesh_correspondence/ORIGINAL_RGB_SUBSET_AUDIT.json)。原有135训练RGB候选与完整世界坐标CSV原样保留。\n\n'
 ghost=OUT/'comparison/P3_leaf_tip_spatial/complete.json'
 if ghost.is_file():
  data=json.loads(ghost.read_text())
  text+=f'固定source157的192px叶缘窗口中，P3 shared有{data["candidate_extra_pixels"]:,}个RAP外额外表面像素，距Mask>3px，属于两个原生连接组件：'+', '.join(str(r['faces'])+'面' for r in data['component_records'])+'；均与最大组件分离。对应实际原图是背景上下文，不把置黑训练GT当成原始背景证据。所有原图像素、camera-Z、世界坐标、面ID及连接组件保留；重新raycast与原缓存深度最大差为'+str(data['exact_camera_cached_depth_max_difference'])+'。\n\n'
  text+='[实际三角面世界坐标](P3_leaf_tip_spatial/source157_extra_surface_world_triangles.csv) · [真实原图窗口](P3_leaf_tip_spatial/source157_original_RGB_context.png) · [同窗口几何对照](visualizations/leaf_surface_continuation/PROTOCOL.json)。这是直接观察后的局部诊断，不能推广成全部视角的无偏发生率；Mask也不是扫描真值。Mesh原文件未改。\n\n'
 text+='来源：实际本地训练RGB、原始02-FFT、保留RAP、native浮点Depth/Normal、真实Mesh三角面和统一采样文件；[固定QGS](https://github.com/will-zzy/QGS/tree/74d05c945e99fcaef7afe5a8831903be71ad9b55)、[固定GS2Mesh](https://github.com/yanivw12/gs2mesh/tree/560f3e8a2349aaad24fe9444ef1ca2770aeec5ab)。博查HTTP403不可用；不以训练数据填充实测结果。\n'
 path=OUT/'comparison/SURFACE_AND_REFLECTION_AUDIT.md';path.write_text(text)
 print('Actual foreground/surface/originalRGB audit written',path,flush=True)

if __name__=='__main__':main()
