# ForeSplat 最终流程决策：KongQueZhuYu

**默认完整流程与当前最佳整体几何流程：标准 VGGT → Point Map 后 RAP 过滤 → 修正相机 → GGGS → 原生 Median Depth TSDF，同时交付 raw/post。照片重建使用同一标准 VGGT+GGGS。标准 VGGT直接Point Map→NKSR保留为边界距离更优的辅助几何分支。**

本次实际完成E11官方2DGS完整30K，复用E8/E10已有30K，运行M1–M5主Mesh和Ω同源GGGS→NKSR补充；全部成功Mesh均评价208视角raw/post。E11→NKSR是额外可选项，两次同参数OOM，没有成功Mesh且未计入几何排名。所有结论限于单株、Seed0、相关训练视角；没有已确认同株扫描，不声称真实厘米级测量精度。

## 三项独立选择

| 决策 | 选择 | 关键证据 |
|---|---|---|
| 前端 | 标准VGGT | 直接NKSR post IoU0.8750 vs Ω0.8263；Boundary Chamfer19.92 vs27.86px；照片E10 PSNR23.9005 vsE8 23.2159 |
| Gaussian后端 | GGGS | E10/E11 PSNR23.9005/22.2615，FG-PSNR19.4605/17.5925，FG梯度0.7313/0.6729；E11原生Mesh结构较差 |
| Gaussian后的Mesh | 原生TSDF | 同一E10输入：TSDF/NKSR post IoU0.8825/0.6551，背景外部几何0.0602/0.2640；真实渲染显示NKSR外扩 |
| 无Gaussian几何分支 | 标准VGGT直接NKSR | post Chamfer19.92px低于TSDF35.82px；IoU0.8750、覆盖0.9864较TSDF0.8825/0.9919略低；不能以一个指标认证全面最优 |

## 对最终十项问题的直接回答

1. **哪个前端更适合植物Mesh？** 当前选标准VGGT。直接NKSR的轮廓、精确率、边界距离均优于Ω，五个人工历史polygon亦支持标准。Ω直接Mesh前景覆盖0.9971高于标准0.9864，但伴随更多外部几何，不能据此称保留了更多真实叶片。原有Ω叶缘查询点优势本次没有转化成整体Mesh优势，1364万点不是精度证明。

2. **E11 2DGS还是E10 GGGS？** 选GGGS。E11同相机/RGB/初始化点云、完整30K，但PSNR低1.6390dB，FG-PSNR低1.8679dB、LPIPS更差；背景Alpha均值0.9931 vs0.3780。原生TSDF post IoU0.6183 vs0.8825，视角53上部叶片条带/缺失明显。2DGS更少Surfel(248629 vs576396)并未带来更好表面或更快训练(48.52 vs40.21min)。显存缓存方式不同，不把Torch峰值差异当固有算法优势。

3. **NKSR是否比TSDF适合薄结构？** 本轮不支持全面优于TSDF。GGGS表面→NKSR明显外扩，原生TSDF更稳定。直接标准NKSR边界距离较好，但破孔、非流形和颗粒结构仍多；没有独立薄叶厚度/双层伪面GT，不能宣布任何算法已解决薄结构。

4. **Gaussian是否显著改善原始几何？** 照片质量与默认TSDF整体结构有实际收益，但不能将其解释为普遍真实三维精度改善。固定标准前端，直接NKSR/M3 GGGS→NKSR post IoU0.8750/0.6551，Chamfer19.92/64.36px，Gaussian后的NKSR更差。深度一致性参照随输入不同，不跨参照冒充独立扫描精度。Ω同源补充的完整数字见下方补充。

5. **直接NKSR是否优于Gaussian→NKSR？** 是，标准前端同源对照明确存在：直接M2的轮廓IoU、外部几何及边界距离都优于M3。需要Gaussian进行照片优化，并不意味着每条几何输出都必须经过它。

6. **哪种保留最多真实叶片/叶缘？** 不能只按覆盖率认定真实。M4目前整体结构/轮廓的综合证据最强，M2边界Chamfer更低；M1/M3/M5高覆盖伴随外部表面。固定八视角真实RGB/Depth/Normal/轮廓、统一外缘窗口均保留，各方法仍有缺失、外扩和过平滑。没有可靠独立证据确认细锯齿、叶柄连接和无双层伪面已恢复。

7. **最适合高度、冠幅、叶长叶宽测量？** M4 raw+post是当前整体结构测量候选，M2为叶缘距离辅助候选；尚未认证为准确测量流程。预测坐标没有厘米标定，bbox轴不是已确认植物高度轴，当前RAP还包含盆/参考标记。需要同株同状态扫描或人工测量、确定尺度与植物语义范围后验证。叶长/叶宽薄表面误差目前不足以作测量精度承诺。

8. **最终完整链路？** 几何输入保持原流程：Mask内旧RAP前景RGB不变，Mask外恢复原背景→标准VGGT已完成208预测→Point Map后Mask/Confidence过滤→修正Camera和COLMAP→GGGS30K→照片输出与native median-depth TSDF raw/post。共享前端直接NKSR分支保留，适用于边界研究和后续独立几何验证；GGGS→NKSR不作为默认。

9. **论文哪些独立贡献值得保留？** 原背景几何上下文与后置前景过滤的受控分离、完整像素中心/非中心主点/Camera ray/渲染-Mesh接口验证、照片与几何分支的证据化选择、完整raw/post及删除组件审计可保留为ForeSplat方法设计/工程验证贡献。没有开发新前馈网络、Gaussian损失或NKSR数学；官方VGGT/GGGS/2DGS/NKSR本身不能冒充原创算法。GGGS→NKSR收益不存在的负结果应保留，避免只报告漂亮Mesh。

10. **是否值得进入19株？** 照片分支值得后续扩展验证；几何测量分支暂不宜直接全面推广。先用有确定同株参照的少量样本验证薄结构/尺度/测量误差，再决定19株全量。当前单株Seed0、相关训练视角和未确认扫描不足以支撑跨植株精准测量结论。本轮未运行19株。

## 产物与复现边界

正式E11模型/训练日志/208浮点RGB、Depth、Normal、Alpha在`E11_2DGS_Corrected/`与`surface_inputs/e11_2dgs_surface/`索引；E8/E10只复用Checkpoint重新渲染表面，没有重训或前馈重推。`meshes/`保存成功主实验及Ω同源补充的raw/post、删除位置和全部208重投影。大文件真实路径见`comparisons/LARGE_ASSET_INDEX.csv`，完整参数/命令/源码和执行偏差见`configs/`，接口Diff见`camera_interface/`。

细表：[照片指标](comparisons/rendering_comparison.csv)、[结构指标](comparisons/mesh_structure_comparison.csv)、[Mesh重投影](comparisons/mesh_reprojection_comparison.csv)、[扫描参照状态](comparisons/mesh_scan_comparison.csv)、[后端分析](E11_BACKEND_ANALYSIS.md)、[Mesh分析](MESH_SELECTION_ANALYSIS.md)、[验收](configs/DELIVERY_VERIFICATION.json)。扫描数值字段为空且明确原因，结构面积/范围为各预测坐标单位，非真实cm。208视角并非208次独立实验。

来源：本地真实完整训练/提取/统一评价；[官方VGGT](https://github.com/facebookresearch/vggt)、[官方GGGS](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids)、[官方2DGS](https://github.com/hbb1/2d-gaussian-splatting)、[官方NKSR](https://github.com/nv-tlabs/nksr)。博查本轮额度不足，官方源码固定版本直接核对；Open3D/PyTorch/cuBLAS文档经Context7检索，不以训练数据替代实验结果。

## Ω 同源补充：真实208视角结果

E8 GGGS surface→NKSR raw/post IoU为0.654731/0.654773，post覆盖0.995787、精确率0.656870、外部几何比例0.277511、Boundary F1@2px 0.088214、Chamfer 71.4294px。对应Ω直接M1 post IoU0.826343、外部几何0.093035、Chamfer27.8635px。Ω与标准前端都出现Gaussian→NKSR较直接NKSR结构退化，不只针对一个前端。E8原输入6,721,643点、同一相对voxel/法线/ks权重/PCG/提面协议，raw4,666,500顶点/9,209,914面，post4,661,772顶点/9,204,826面，耗时84.68s。数据与所有成功Mesh一样按自身相机评价，未做扫描尺度配准。
