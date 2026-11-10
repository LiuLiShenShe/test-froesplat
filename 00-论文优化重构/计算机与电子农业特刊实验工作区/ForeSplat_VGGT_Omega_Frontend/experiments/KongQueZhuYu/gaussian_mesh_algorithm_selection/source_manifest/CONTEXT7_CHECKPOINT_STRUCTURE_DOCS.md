# Checkpoint ZIP64：当前官方文档与实际文件核验

2026-11-09 通过 ctx7 library PyTorch 解析官方项目，再通过 docs /websites/pytorch 查询保存布局与完整性。返回官方 https://docs.pytorch.org/docs/2.14/notes/serialization.html 和 https://docs.pytorch.org/docs/2.14/generated/torch.save.html 。当前文档版本2.14，实际E13运行PyTorch2.5.1+cu121；没有采用新版本独有的metadata属性或据其推测本地加载结果。

官方文档说明 torch.save 默认使用 ZIP64，元数据与张量存储分开保存。实际文件另由本地Python ZIP读取和已安装PyTorch原生PyTorchFileReader读取：失败导出PTH两者均失败，ZIP64目录记录的偏移大于真实文件长度。原始文件不改写、不通过目录存在认定可恢复。该结构问题的原因未确定。

只读取原生archive目录和data.pkl，不把该检查标为所有张量CRC、数值有限性或参数来源逐位核验。固定正式Checkpoint的真实tensor/PLY对应由另一个CPU核验脚本执行。PLY结构检查按实际标量字段大小和顶点数计算精确文件长度，实际失败时独立PLY长度正确；不把它冒充有效优化器Checkpoint。

实际证据：artifact_index/NATIVE_ARTIFACT_STRUCTURE_AUDIT.json；E13_QGS/failure_exports/20261109T084120Z_1535250/NATIVE_CHECKPOINT_READ_VALIDATION.json及VALIDATED_DELIVERY_STATUS.json。
