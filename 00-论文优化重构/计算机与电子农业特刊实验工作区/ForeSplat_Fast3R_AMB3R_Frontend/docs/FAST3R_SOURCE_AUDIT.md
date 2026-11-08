# Fast3R 官方源码与论文核验

论文已直接下载至 `sources/Fast3R.pdf`（https://arxiv.org/pdf/2501.13928），文本 `sources/Fast3R.txt`。源码 commit `33104d4b5b8df43795ecded236194958bbdac572`，未修改源码。

- 论文 §3.1：Global pointmap 在第一输入相机坐标系；Local pointmap 在各观看相机坐标系。§3.2 独立归一化点图回归，不保证真实公制尺度。
- §3.3：共享 CroCo ViT-L 编码器独立编码图像；24 层融合 Transformer 对所有视角的拼接 token 做 all-to-all 注意力。图像索引嵌入识别第一视角。两个 DPT Head 分别输出 Global/Local XYZ 和对应 Confidence。
- §3.2 置信度激活为 `1 + exp(logit)`。实际权重 config `conf_mode=[exp,1,Infinity]` 一致。`conf>1` 是官方有效性基线；并不等价于概率校准后的正确几何。
- `models/fast3r.py:_encode_images()` 原有默认 encoder 分块400；解码前仍拼接全部输入 token，不是几组独立重建。`set_max_parallel_views_for_head()` 是官方暴露的显存设置，只分块 Head，保留全局融合。
- `dust3r/inference_multiview.py:inference()` 在 profiling=True 返回 `(dict,profile)`，False 返回 dict；README 的通用示例并不能覆盖所有返回模式。
- 本commit的 `loss_of_one_batch()` 只有字符串 `precision == "32"` 分支关闭外层autocast；`torch.float32` 会落入默认autocast参数。真正FP32对照通过原有字符串协议 `dtype="32"` 调用，保留原始Attention内部BF16计算及其FP32回转，不修改官方函数。新Runner `--precision float32` 明确记录这一协议，以完整208帧对照检查数值精度对薄结构的影响。
- `models/multiview_dust3r_module.py:estimate_camera_poses()` 按 `first_view_from_global_head` 从第一图 Global 输出估计焦距，然后对所有帧运行 PnP。
- `estimate_cam_pose_one_sample()` 使用 `conf>1.0`，失败返回单位矩阵。原返回值没有失败标签；独立 observational wrapper 记录原函数的实际成功与原 OpenCV RANSAC inlier 数，不改变调用参数或输出。
- `dust3r/cloud_opt/init_im_poses.py:fast_pnp()` 调用 SQPNP RANSAC，重投影阈值5像素，原始结果是 W2C，返回前求逆为 C2W。验证通过 C2W 求逆后重投影实际前景点，并记录旋转正交性、正深度比例与第一帧接近单位矩阵程度。
- 本地模型源为官方 `jedyang97/Fast3R_ViT_Large_512` 预先下载目录；保留原网络及权重。没有安装 cuROPE；官方 README 明确其会影响预测。

输入保持2160:3840纵横比，512长边得到288×512，无裁剪、无需额外 padding。保存含 half-pixel 偏移的像素坐标变换，可计算 `K_original=A^-1 K_model`。

引用：[官方网络](https://github.com/facebookresearch/fast3r/blob/33104d4b5b8df43795ecded236194958bbdac572/fast3r/models/fast3r.py)、[官方相机估计](https://github.com/facebookresearch/fast3r/blob/33104d4b5b8df43795ecded236194958bbdac572/fast3r/models/multiview_dust3r_module.py)、[论文](https://arxiv.org/abs/2501.13928)。
