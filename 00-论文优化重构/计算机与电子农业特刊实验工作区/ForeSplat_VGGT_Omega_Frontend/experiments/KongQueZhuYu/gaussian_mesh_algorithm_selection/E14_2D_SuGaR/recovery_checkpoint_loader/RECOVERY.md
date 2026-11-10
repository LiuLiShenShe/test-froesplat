# 完成15K后的纹理提取恢复

原生15K训练成功并保存15000.pt与三尺度Gaussian PLY；随后官方extractor的torch.load在PyTorch2.10默认weights_only=True下拒绝该本次自训练模型的NumPy损失标量。按Context7当前PyTorch2.10序列化文档，仅对本次已核验的原生本地模型显式weights_only=False。官方表面绑定、训练、纹理优化及提取逻辑保持不变。

恢复时读取实际15000.pt并核对iteration=15000、非空绑定面、原始完整配置和实际训练日志；直接续接纹理/网格导出，不再优化或重复训练。原始失败日志与终态PT保持原样。耗时合计保留失败提取成本；恢复峰值显存单独标为texture/Mesh export，不能冒称原训练峰值。

来源：[PyTorch2.10官方序列化说明](https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/serialization.rst)、本次Context7查询、原生PT及完整训练/提取日志。
