# Table 7 证据审计

论文 Table 7 声称 20 序列平均，但本地证据只显示：历史 2DGS `cfg_args` 模型 11 个，Mesh 日志 12 个，M2M3 full summary 2 个文件且每个是 3 场景 pilot，不能拼成20序列同协议平均。
逐行实际产物在 `tables/TABLE7_RECONSTRUCTION_SOURCE.csv`；可作为 legacy/pilot 的行在 `TABLE7_REUSABLE_RESULTS.csv`。
论文四行的 PSNR/SSIM/LPIPS/训练时间/网格时间被解析并保存在 source CSV；没有任何一行获得20序列可复核证据。
M2M3 pilot 有真实训练、render、metrics elapsed 日志，故时间可作为 pilot 结果复用；不能替换论文四方法 Table7 时间。
COLMAP、3DGS+VFM、Standard 2DGS、ForeSplat 的 common view split、评价区域、失败样本处理和 wall-clock 证据未同时闭合。
