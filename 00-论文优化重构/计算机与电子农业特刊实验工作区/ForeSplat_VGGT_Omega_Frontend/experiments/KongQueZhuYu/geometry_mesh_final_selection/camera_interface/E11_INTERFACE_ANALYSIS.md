# E11：与 E10 等价的相机投影接口

全部 208 个运行时 Camera 的有效尺寸均为 1080×1920。Fx/Fy/Cx/Cy 与 E10 相差小于 1e-8，World-to-Camera、Projection 和完整投影矩阵相差小于 1e-7；208 个浮点训练 RGB 和 RAP Mask 的哈希也逐帧一致。验证结果见 `E11_validation.json`、各表面输入目录的 `render_complete.json`。

相机从 E10 的 Standard_VGGT_Original_RGB/COLMAP_Corrected 读取。训练图像缩放采用像素中心约定：`c'=(c+0.5)*scale-0.5`，主点不强制居中。以列向量表示的透视矩阵采用 `P[0,2]=(2*cx+1)/width-1`、`P[1,2]=(2*cy+1)/height-1`。源码储存的是其转置。深度反投影采用整数像素中心 `((u-cx)/fx*z, (v-cy)/fy*z, z)`，再由 C2W 转换到世界坐标。

E6 已部署的官方 surfel CUDA forward/backward 使用完整投影矩阵构造 splat-to-pixel 单应性。实测可正确传递非中心主点，因此没有修改 CUDA 核心数学。Python 补丁修改 CameraInfo、Camera、缩放内参、深度反投影和 Open3D Mesh 相机接口，并仅增加原生 median/expected depth 输出字段。训练循环使用官方 train.py；原有 E6 GaussianModel、增密、剪枝和优化器保留，附加前景、Alpha、Mask 损失均关闭。

数值验证：随机三维点的完整投影最大偏差 4.60e-5 px；非中心合成相机的真实 CUDA Alpha 质心偏差 0.000134 px，反向梯度有限且非零。另取实际 0、53、105 视角的旋转姿态构造平面，CUDA 质心偏差分别为 0.000960、0.001853、0.002421 px，深度光线误差均小于 1.8e-7。真实 CUDA expected depth 最大偏差 1.1e-6，median depth 偏差为零。参见 `rotated_surfel_validation.json`。

原始补丁与源码哈希保存在 `diffs/` 和 `2dgs_patch_complete.json`。训练后发现原先宽泛的字符串替换也触及了未使用的 MiniCam，已修复预览接口；实际训练使用的 Camera 类完全相同。训练时 Camera 文件的原样副本存于 `training_source_snapshot/scene/cameras.py`，其 SHA256 与 `actual_training_parameters.json` 一致；不能将修复后的整个文件哈希冒充训练时哈希。该情况完整记于 `configs/execution_deviations.json`。

Mesh 重投影使用每种前端各自的 Camera 和坐标系。Omega 仅应用已有 COLMAP 导出的刚性变换，不做扫描尺度拟合或跨前端 Sim(3) 配准。原生 TSDF 使用修正的真实 K、W2C；Mesh 光线不归一化，使命中参数等于相机轴深度。原始前馈输入与运行时 Camera 的数值检查见 `direct_surface_camera_validation.json`。

来源：[官方 2DGS](https://github.com/hbb1/2d-gaussian-splatting)、[Open3D Raycasting](https://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html)；本地实际源码、相机、训练和验证记录。博查检索本轮因额度不足不可用。
