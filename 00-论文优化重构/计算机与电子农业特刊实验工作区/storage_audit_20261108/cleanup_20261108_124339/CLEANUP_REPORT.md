# 已授权存储清理执行报告

仅执行用户选定的三类清理；没有使用GPU、conda或重跑模型。

| 类别 | 删除文件 | 释放分配块 |
|---|---:|---:|
| old_colmap_intermediate | 11,096 | 184.23 GiB |
| early_smoke_predictions | 1,012 | 4.03 GiB |
| temporary_sfm_duplicate | 5,145 | 48.72 GiB |

数据盘可用空间变化：188.26 GiB；系统盘可用空间变化：48.72 GiB。其他任务可能同时影响df，逐项分配块见清单。

保留验证通过：20份最终融合PLY、7份正式208 Bundle、4套早期运行的报告/参数/RGB/Mask/相机/可视化。59份大型预测目录中的文本记录另存备份。

所有模型主权重、环境、缓存、E1–E6 Checkpoint、原始图像/Mask/GT、根因审计结果均不在删除清单中。

历史存储审计是清理前快照，未改写。旧COLMAP直接重融合需重新生成深度/法向；早期Smoke Bundle的数组已清理，目录新增CLEANUP_NOTICE记录。

[删除清单](deletion_manifest.json)、[逐项执行日志](deletion_log.jsonl)、[最终核验](CLEANUP_RESULT.json)、[删除前保留证据](protected_artifacts_before.json)、[临时重复副本复核](temporary_duplicate_preflight.json)。

来源：本机文件系统、既有持久化核验和用户明确授权。博查检索返回403额度不足，无搜索结果可引用。
