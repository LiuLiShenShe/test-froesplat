# E11 与最终 Mesh 选择实验协议

本目录独立于 E1–E10。复用两份已经完成的208帧几何预测及E8/E10的30000步Checkpoint，不重新推理VGGT、不重新分割、不重新训练GGGS。所有GPU进程仅可见物理GPU1，使用uv；官方TSDF的CPU融合和CPU几何统计保留官方实现。

E11复用标准VGGT的COLMAP_Corrected、原208帧RGB、原初始化PLY与Seed0。真实运行时1080×1920。完整208个K/W2C/P和浮点GT与E10逐项核对，CUDA真实surfel投影、反向梯度、深度反投影和Open3D相机通过验证。非中心主点与像素中心仅修复相机接口，官方surfel CUDA核心完整投影矩阵和数学保持原样。独立源码及E6→E11补丁位于camera_interface/，正式30K使用官方train.py，lambda_normal=.05、lambda_dist=.2、depth_ratio=0、SH3，原生优化/增密/剪枝，无新前景损失或Alpha剪枝。

直接NKSR输入读取所有原始逐帧前景点，而不是融合COLMAP PLY或Gaussian中心。原有p20有效掩膜保留，各模型置信度不跨模型数值比较。法线由同帧深度反投影的世界点计算，中央或单边邻域有限差分，深度邻域跳变阈值为5%，由来源相机定向。法线无效的点仍保存在完整输入档案。Gaussian表面来自真实208帧渲染，NKSR有效性采用原RAP Mask、Alpha≥.5、有限正深度和相同法线协议。

每点保留XYZ、RGB、法线、来源相机位置、source ID、像素、深度及Confidence或Alpha。每空间体素和法线半球保留原始最早观测，避免相反薄表面法线平均抵消。完整原点云留存。参数先于Mesh质量观察确定；同一前端的直接点云、GGGS及2DGS使用相同绝对尺度参数。空间参考来自完整前景点的确定性逐帧抽样分位范围，只定义尺度，不裁剪原始几何，不进行扫描尺度配准。Omega只应用已有导出刚性变换以对应自己的COLMAP Camera。

主NKSR为官方ks预训练权重，显式voxel=前景参考对角线/512、采样voxel=对角线/1024，MISE1和官方trim。具体绝对参数、点数、法线策略及权重SHA见configs/和各complete.json。官方TSDF保留原GGGS VoxelBlockGrid及2DGS ScalableTSDFVolume，使用同一前端voxel、sdf=5voxel、同源原深度最大值×1.25的depth_trunc。主TSDF使用RAP Mask和原生深度；额外相同Alpha筛选的TSDF对照用于隔离NKSR输入筛选与提取算法，不能仅由主Mesh差异认定算法独立收益。

所有raw Mesh完整保留。post仅删除不足50面的组件，不采用仅保留最大组件；删除位置、索引、面积及与208帧RAP Mask的投影重叠单独记录。轮廓和深度重投影使用原生1080×1920整数像素射线和各自坐标系相机。结构量、深度自一致性及RAP预测掩膜属于几何代理，不替代真实三维扫描精度。五个人工标注帧另报告历史范围拆分的来源。没有确定同株同状态扫描对应时，扫描指标为空并说明原因，不编造Chamfer或厘米精度。

统一渲染指标使用E10同一环境、同一审计函数和同一208帧浮点GT/Mask SHA，LPIPS Alex不缩放，SSIM win7/sample covariance，前景和5px内侧边缘及Rec709 Sobel指标保持一致。208帧是同一植株Seed0的相关训练视角，不宣称独立多种子显著性。最终报告依据真实30K结果、全部主Mesh及少量尺寸受控的直接视觉核验作出三个独立决策，不以点数、组件数或Watertight单独排名。

来源：本地E6、E8、E10源码/参数/Checkpoint；[2DGS官方](https://github.com/hbb1/2d-gaussian-splatting)、[NKSR官方及使用定义](https://github.com/nv-tlabs/nksr/blob/master/NKSR-USAGE.md)、[GGGS官方固定源码](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/tree/23b418c2f0f19b30b251acad6fe13445dae968c7)、[Open3D官方](https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.VoxelBlockGrid.html)。博查搜索403配额不足；直接阅读官方源码，Open3D通过Context7获取文档。真实指标仅来自本地执行文件。
