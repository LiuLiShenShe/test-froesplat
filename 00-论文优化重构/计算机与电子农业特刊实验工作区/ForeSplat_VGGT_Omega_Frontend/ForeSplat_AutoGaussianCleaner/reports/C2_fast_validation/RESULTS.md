# C2_raw 三株实测（2026-11-14）

孔雀竹芋新入口保留 428,967 / 576,396 个 Gaussian，逐 ID 与历史 C2_raw 完全一致。六项回归检查通过；未改动历史 rules() 或清理阈值。

| 植物 | 原始点数 → 保留点数 | 删除比例 | 观测背景黑像素减少 | 前景 PSNR（dB） | 实际图像检查 | CPU时间 | 总时间 | 应用层峰值 I/O |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| 长寿花1 | 441,929 → 299,528 | 32.22% | 99.49% | 20.56 → 17.12（-3.44） | 明显过删：暗叶与花盆碎裂 | 8.98s | 14.63s | 19.60 MiB/s |
| 豆瓣绿1 | 748,702 → 638,938 | 14.66% | 99.59% | 19.42 → 18.72（-0.70） | 主体保留，暗部局部损伤 | 10.60s | 21.87s | 19.58 MiB/s |
| 网纹草1 | 9,172,385 → 8,550,282 | 6.78% | 99.74% | 10.20 → 12.54（+2.34） | 两个暗视角植株近乎消失 | 48.55s | 260.15s | 19.62 MiB/s |

## 覆盖和资源

| 植物 | 边缘覆盖：原始 → C2 | 薄结构覆盖：原始 → C2 | 新增前景透明孔洞 | 峰值 RSS | 逻辑读取 | 逻辑写入 | 完整读/写次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 长寿花1 | 90.20% → 57.32% | 96.78% → 87.21% | 10.41% | 0.240 GiB | 106.21 MiB | 71.99 MiB | 1 / 1 |
| 豆瓣绿1 | 100.00% → 66.29% | 100.00% → 96.78% | 3.51% | 0.339 GiB | 179.93 MiB | 153.55 MiB | 1 / 1 |
| 网纹草1 | 95.12% → 52.77% | 95.16% → 55.66% | 20.66% | 3.235 GiB | 2204.36 MiB | 2054.86 MiB | 1 / 1 |

三株平均清理墙钟时间 98.88 秒，平均 CPU 时间 22.71 秒。GPU 渲染评价单独执行，其时间与 I/O 不计入清理性能。

每株采用相同的16个既有观测相机（长边960），复用图像与mask。背景黑像素为原评价器的 dark_bg_high_alpha_pixels，汇总16视角；不是孔雀竹芋固定8自由视角协议，百分比不与其直接对标。边缘、薄结构和骨架指标依赖既有前景mask，是结构代理而非逐叶语义真值。

20 MiB/s 为应用层 PLY 读写合计限速。表中峰值为实测尾随一秒窗口逻辑吞吐；未测硬盘设备物理峰值，内核预读/回写及其他进程不受此限速保证。每株保留全部原生字段和保留记录字节，无中间模型、removed/viewer PLY、逐点NPZ/CSV或完整PLY哈希扫描。

## 直接图像检查

### 长寿花1

直接查看全部16视角拼图及全尺寸view 00、07：大面积黑云被清除，但多片下部/内侧暗叶出现透空和碎裂，叶片连接区域显著变稀，花盆表面出现大片透明孔洞。上部亮叶与花序仍可见，不能据此认定完整植株受到保护。代表图为crop_0055，新增前景alpha孔洞16.76%；16视角平均10.41%。未对每条叶柄做语义标注，不能保证其完整性。

![同相机原始与C2](ChangShouHua1/paired_view_07.png)

[全部16视角并排图](ChangShouHua1/all_16_pairs.png) · [原始渲染](ChangShouHua1/original_first_observed_full.png) · [清理后渲染](ChangShouHua1/C2_raw_first_observed_full.png) · [逐视角指标](ChangShouHua1/paired_quality.json)

[清理后原生PLY](../../experiments/C2_fast_validation/ChangShouHua1/cleaned_gaussians.ply) · [清理资源报告](../../experiments/C2_fast_validation/ChangShouHua1/cleanup_report.json)

### 豆瓣绿1

直接查看全部16视角拼图及全尺寸view 00、08：主体叶片、叶尖和可见主茎总体保留，未见主茎整段消失；下层阴影叶片、盆口内部与花盆暗侧局部变稀/侵蚀。view 14、15等近景仍见黑色叶缘/残余黑边，因此不能描述成全部黑点和黑条均已清零。最差view 08（crop_0092）新增前景alpha孔洞8.87%；平均3.51%。边缘覆盖下降也受原始黑云对mask的过覆盖影响，不能将其全部解释成叶片丢失。

![同相机原始与C2](DouBanLv1/paired_view_00.png)

[全部16视角并排图](DouBanLv1/all_16_pairs.png) · [原始渲染](DouBanLv1/original_first_observed_full.png) · [清理后渲染](DouBanLv1/C2_raw_first_observed_full.png) · [逐视角指标](DouBanLv1/paired_quality.json)

[清理后原生PLY](../../experiments/C2_fast_validation/DouBanLv1/cleaned_gaussians.ply) · [清理资源报告](../../experiments/C2_fast_validation/DouBanLv1/cleanup_report.json)

### 网纹草1

直接查看全部16视角拼图、全尺寸view 00/02与crop_0013目标图：多数视角的大面积黑云显著消失，但仍有黑条、亮色漂浮物和模糊/发亮的叶片。view 01（crop_0012）与view 02（crop_0013）中原本可见的暗色植株几乎消失；后者对应目标图确有可见绿色植株，不能将该区域的删除解释成单纯去除黑云。两视角新增前景alpha孔洞分别87.09%和99.39%；16视角平均20.66%。其他视角原模型本就模糊且被黑云遮挡，不能保证每片叶和叶柄完整。尽管平均PSNR由10.20升至12.54 dB，这两个视角的明显丢失已不满足结构保留要求。

![同相机原始与C2](WangWenCao1/paired_view_02.png)

[全部16视角并排图](WangWenCao1/all_16_pairs.png) · [原始渲染](WangWenCao1/original_first_observed_full.png) · [清理后渲染](WangWenCao1/C2_raw_first_observed_full.png) · [逐视角指标](WangWenCao1/paired_quality.json)

[清理后原生PLY](../../experiments/C2_fast_validation/WangWenCao1/cleaned_gaussians.ply) · [清理资源报告](../../experiments/C2_fast_validation/WangWenCao1/cleanup_report.json)

[crop_0013：目标RGB、原始GGGS、C2三联图](WangWenCao1/target_original_c2_view02.png)。

## 结论

C2_raw在三株的观测背景黑像素上均清除超过99%，可固定为强去云基线和低I/O入口；当前不建议直接统一应用于19株。长寿花1有明显暗叶/花盆过删，网纹草1在crop_0012/crop_0013暗视角中植株几乎消失；豆瓣绿1主体较完整但暗部有局部侵蚀、残余黑边。三株平均清理98.88秒，917万点网纹草260.15秒，应用层峰值均低于20 MiB/s。本轮未调整阈值或关闭KNN；后续是否调整现有参数，应基于这些损伤再决定，不能仅依据平均PSNR或背景黑像素下降而验收。

清理入口：[fast_clean_c2.py](../../fast_clean_c2.py)；复现说明：[FAST_C2_README.md](../../FAST_C2_README.md)。原始/最终点数、时间、RSS和I/O来自每株 cleanup_report.json；渲染质量来自每株 paired_quality.json。实际输入和源码哈希记录在 [protocol.json](protocol.json)。

网纹草首次GPU评价收到SIGTERM（退出143），中断日志和原始视角结果保留在WangWenCao1_interrupted。随后单独重跑完整16视角原始/C2配对评价，已完成；未重新清理或覆盖模型。中断和重跑均不计入清理资源统计。

联网检索已尝试博查，返回 HTTP 403；实验结论以本地源码、实测 JSON 和直接查看的渲染图为来源。SciPy 文档通过 Context7 拉取，来源：https://github.com/scipy/scipy/blob/v1.18.1/doc/source/tutorial/spatial.rst 。
