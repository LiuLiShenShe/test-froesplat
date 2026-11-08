只允许使用GPU1
使用uv管理依赖，绝对不允许使用conda
/data/fj/F2DMAS/03-SAM 目录下包含植物的前景图像和mask图像
/data/fj/F2DMAS/02-FFT 是原图

1. 第一阶段：VGGT-Ω COLMAP 适配
   - 使用 tomhol 社区版本。
   - 增加 RAP-FSAM3 掩膜过滤。
   - 修正相机内参和图像坐标映射。
   - 输出 GGGS 标准数据结构。
   第一步
2. 第二阶段：GGGS 原始版本运行
   - 使用 VGGT-Ω 初始化。
   - 先跑通官方 GGGS。
   - 检查初始点云和 Gaussian 重建效果。
   - 导出原生 Mesh。
3. 第三阶段：ForeSplat 改造 GGGS
   - 独立曝光补偿模块。
   - 前景 RGB 和几何约束。
   - 前景 Gaussian 初始化和剪枝。
   - 完成各项消融。
4. 第四阶段：NKSR
   - 导出最终定向点云。
   - 接入 NKSR。
   - 比较 GGGS 原生 Mesh 和 NKSR Mesh。
   - 验证植物表型测量精度。



