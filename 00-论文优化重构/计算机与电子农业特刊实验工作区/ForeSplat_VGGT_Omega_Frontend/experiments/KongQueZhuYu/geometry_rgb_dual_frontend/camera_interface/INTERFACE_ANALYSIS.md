# 独立像素接口修复及数值证据

这是 E8/E10 相对于 E7/E9 的独立因素。两份前端预测只运行一次；每个前端的原/修正版共享同一个初始化 PLY、相同训练 RGB 与 Mask。没有改变点云融合、权重、损失、增密、曝光或扫描尺度。

## 像素约定与完整投影链

采用整数坐标对应像素中心的约定。若原图像素中心坐标为 `u`，实际 resize 的映射为 `u' = s*(u+0.5)-0.5`。Crop/Padding 的平移必须先进入原图到模型图的 A，模型预测内参转换为原图内参使用 `K_original = inverse(A) * K_model`。

- Ω 保留实际 balanced@512 的 384×688 resize；既有 A 已包含半像素项。原/修正版 K、姿态与初始化点云保持一致。修复作用于 GGGS 实际相机和投影链。
- 标准 VGGT 保留 square padding→1024→518。原接口保留 E5 的 A；修正版加入 `(s-1)/2`。这不会重做推理或改变 Point Map。原图 COLMAP 的 cx/cy 分别改变 3.2065637 px；fx/fy、尺寸、相机姿态和点云不变。
- GGGS 的 source→training resize 同样采用 `(c+0.5)*s-0.5`，训练分辨率仍为 1080×1920。
- `CameraInfo` 保留 source cx/cy，`loadCam` 保留 resize 后的 fx/fy/cx/cy；`Camera` 实际使用这些内参。
- CUDA 使用 `pixel = ((ndc+1)*size-1)/2`。因此相机投影矩阵的主点项为 `(2*cx+1)/W-1` 与 `(2*cy+1)/H-1`，尺度项为 `2*fx/W` 与 `2*fy/H`。现有转置/存储约定、W2C/C2W 与深度轴方向保留。
- Render、sample-depth 的前向和反向 CUDA 都读取同一 view/full projection 参数恢复非中心主点，射线为 `(u-cx)/fx, (v-cy)/fy, 1`。只修改 cameras.txt 无法实现这一步。公开 Python/C++ ABI 保持原样。
- 既有 NCC/几何 warp 已读取 Camera 的 Cx/Cy，使用整数像素中心，因此沿用原损失实现。

旧仓库保持不变；修正版与私有编译扩展位于本目录。实际变更、源码 SHA 和扩展 SHA 见 `final_source_changes.json`、`repair_provenance.json`、`diffs/` 与编译日志。

## 已实际执行的验证

合成测试包含平移及 0°/19°/−35° 旋转相机、非中心主点和薄 Gaussian。检查实际投影、alpha 质心、深度采样射线，以及 render/sample 反向传播。修正版投影误差约 1e−5 px，alpha 质心误差约 0.011 px，采样射线重投影约 4e−6 px，反向梯度有限且非零；详见 `synthetic_corrected.json`。

全部两前端×208视角×原/修正版的 832 个实际 GGGS Camera 参数与预测几何进行数值比较：

|前端|原接口平均投影偏差|修正版平均投影偏差|修正版最大投影偏差|
|---|---:|---:|---:|
|Ω|1.981045 px|0.0000861 px|0.000758 px|
|标准 VGGT|2.620938 px|0.0000727 px|0.000420 px|

上述是与同一预测相机的接口偏差，不能当作相机真值精度。另用固定 RGB 前景匹配，独立投影到目标真实 GGGS Camera，并与 RGB 目标匹配位置比较：

|前端|前景匹配中位重投影，原→修正|前景边界匹配中位重投影，原→修正|
|---|---:|---:|
|Ω|4.576869→4.083108 px|4.962450→4.009617 px|
|标准 VGGT|3.726077→3.088905 px|4.058914→3.464692 px|

这些值位于相同 1080×1920 训练坐标；同一前端的原/修正版使用相同源深度、p20 有效点及 RGB 匹配。边界包含植物和盆的 RAP 边界，未冒充单叶实例真值。全部原始逐对结果与参数见 `matched_RGB_reprojection.csv`、`matched_RGB_reprojection_summary.csv`、`matched_RGB_reprojection_protocol.json`、`projection_original.csv`、`projection_corrected.csv` 和四份 `*_runtime_parameters.json`。

精确控制检查见 `../configs/pair_invariants.json`。Ω 的姿态与观测二进制/文本记录逐字节保留，避免重序列化四元数引入额外变量；标准姿态浮点差仅 1.11e−16。

重建收益必须使用完整30K的 E8−E7 和 E10−E9 得出，不能由这里的数值投影改进推定。

来源：本目录已实际执行的源码修复、合成 rasterizer 检查、208视角 Camera 与 RGB 匹配结果；[GGGS 固定官方源码](https://github.com/HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids/tree/23b418c2f0f19b30b251acad6fe13445dae968c7)。
