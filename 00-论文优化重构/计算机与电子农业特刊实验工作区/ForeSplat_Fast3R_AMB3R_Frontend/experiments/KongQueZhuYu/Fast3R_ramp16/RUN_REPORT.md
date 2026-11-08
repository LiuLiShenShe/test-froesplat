# Fast3R_ramp16 实际运行记录

输入 16 个连续可用视角，完整序列：False。这是冒烟/视角扩展检查，不能代替全部208帧实验。

一次保留全部输入视角交互的官方 Fast3R 前向。物理 GPU1，uv 独立环境；网络和权重未修改。

- 输入尺寸 W×H：[288, 512]；原始尺寸：[2160, 3840]；无裁剪、无曝光修改、无 Mask 腐蚀。
- 网络推理：0.7286s；PnP：0.3799s；模型+PnP：1.1085s。
- 峰值显存 allocated：4.5910GiB。
- 官方 PnP 实际成功：16/16；niter=100；first_view_from_global_head。
- 前景自投影误差各帧中位数的中位数：1.4561px。此项检查点图与相机约定，不是三维GT精度。
- 第一帧C2W到单位矩阵距离：0.0270696，没有强行固定为单位矩阵。
- Git commit：33104d4b5b8df43795ecded236194958bbdac572。
- 权重SHA256：1357d42e74539ba03510ea1986c7464a7facf17157779e2b7ebd6284e79e0427。

Global Point Map 使用第一输入相机参考系，Local Point Map 使用各视角独立相机系；两者分别保存，未做新融合或尺度校正。

原始预测、置信度、相机、焦距、像素变换和完整前景点云均保存到本 Bundle。网络/PnP时间不含输入准备、权重加载、诊断和输出序列化。

来源：本目录实际 bundle.json / geometry_validation.json / pose_status.json；[Fast3R官方源码](https://github.com/facebookresearch/fast3r)。