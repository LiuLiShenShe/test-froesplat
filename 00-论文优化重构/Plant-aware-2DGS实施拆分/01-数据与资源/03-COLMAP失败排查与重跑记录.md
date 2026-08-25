# COLMAP 失败排查与重跑记录

更新日期：2026-05-17  
工作区：`/data/fj/F2DMAS`

## 1. 本次排查目的

用户提醒“有一些 COLMAP 是失败的”，因此本次检查不只看文件夹是否存在，而是检查：

- `sparse/0/cameras.bin`
- `sparse/0/images.bin`
- `sparse/0/points3D.bin`
- 输入图像数量
- 注册图像数量
- 注册率
- 3D 点数量
- `images/` 去畸变图像数量

判断阈值：

- `OK`：注册率 >= 70%，且 3D 点数大于 0。
- `WARN`：注册率 30%-70%，且 3D 点数大于 0。
- `FAIL`：注册率 < 30%，缺模型，或点数不可用。

## 2. 当前 `04-COLMAP` 实际状态

初查结果：

| sample | status | input | registered | rate | points3D | 备注 |
|---|---:|---:|---:|---:|---:|---|
| BaiZhang | FAIL | 214 | 2 | 0.9% | 116 | `sparse/0` 失败，但同目录还有 `sparse/2` 更大模型 |
| CaoMei1 | OK | 210 | 210 | 100.0% | 24226 | 可用 |
| CaoMei2 | WARN | 210 | 106 | 50.5% | 26199 | 建议重跑以提高注册率 |
| ChangShouHua1 | FAIL | 213 | 2 | 0.9% | 16 | 失败 |
| ChangShouHua2 | OK | 212 | 212 | 100.0% | 20796 | 可用 |
| ChangShouHua3 | MISSING | 210 | 0 | 0.0% | 0 | 缺少 sparse 模型 |
| DouBanLv1 | OK | 215 | 215 | 100.0% | 10764 | 可用 |
| DouBanLv2 | OK | 210 | 210 | 100.0% | 21653 | 可用 |
| DouBanLv3 | OK | 210 | 200 | 95.2% | 47352 | 可用，但相机模型为 OPENCV |
| HongZhang | OK | 213 | 213 | 100.0% | 24912 | 可用 |
| KongQueZhuYu | FAIL | 210 | 3 | 1.4% | 162 | 失败 |
| WanNianQing1 | OK | 213 | 213 | 100.0% | 23259 | 可用 |
| WanNianQing2 | OK | 215 | 215 | 100.0% | 17715 | 可用 |
| WangWenCao1 | OK | 211 | 149 | 70.6% | 35160 | 勉强可用 |
| WangWenCao2 | OK | 210 | 210 | 100.0% | 33317 | 可用 |
| XianKeLai1 | OK | 208 | 208 | 100.0% | 47148 | 可用 |
| XianKeLai2 | OK | 208 | 208 | 100.0% | 26405 | 可用 |
| XianKeLai3 | OK | 206 | 206 | 100.0% | 40140 | 可用 |
| XiangPiShu1 | OK | 208 | 203 | 97.6% | 36875 | 可用 |
| XiangPiShu2 | FAIL | 211 | 2 | 0.9% | 224 | 失败 |

需要处理的样本：

```text
BaiZhang
CaoMei2
ChangShouHua1
ChangShouHua3
KongQueZhuYu
XiangPiShu2
```

## 3. 为什么这些样本需要重跑

### BaiZhang

- `sparse/0` 只有 2/214 张注册，注册率 0.9%。
- 进一步检查发现同一目录下还有多个模型：
  - `sparse/0`：2 images，116 points
  - `sparse/1`：63 images，13494 points
  - `sparse/2`：131 images，34222 points
- 说明 mapper 生成了多个模型，`sparse/0` 不是最佳模型。
- 需要重跑或至少选择最大模型重新 undistort。

### CaoMei2

- 当前 `sparse/0` 注册 106/210，注册率 50.5%。
- 属于 WARN，可用于部分测试，但不适合主实验或消融基准。
- 建议重跑以争取更高注册率。

### ChangShouHua1

- 当前只注册 2/213，注册率 0.9%。
- 旧 `rerun_log.json` 也显示原图重跑后仍只有 2 张注册。
- 需要尝试更换 matcher、降低帧密度、或使用 feature mask/不同输入策略。

### ChangShouHua3

- 当前缺少 `sparse` 输出。
- 旧 `rerun_log.json` 显示曾经只注册 2/210，且 mask 过滤后点数为 0。
- 需要重跑。

### KongQueZhuYu

- 当前只注册 3/210，注册率 1.4%。
- 另有 `KongQueZhuYu_undist_pinhole` 目录，但检查后仍只有 3 images、162 points。
- 需要重跑。

### XiangPiShu2

- 当前只注册 2/211，注册率 0.9%。
- 旧 `rerun_log.json` 也显示原图重跑后仍只有 2 张注册。
- 需要重跑。

## 4. COLMAP 输入策略判断：原图、mask，还是 masked 图？

当前建议：

> 用原图或 FFT 保留后的原图进行 COLMAP 位姿估计；mask 不直接替换输入图像，而用于 feature mask、稀疏点过滤、后续 2DGS mask loss 和 meshing 边界约束。

原因：

1. 这批项目历史脚本 `rerun_colmap_with_originals.py` 已经记录过：`crop images lost background texture, causing COLMAP to fail on some folders`。
2. 当前失败样本的 `input/` 多为 `crop_*.png` 或 masked/crop 图，多个样本注册率只有 0.9%-1.4%。
3. 植物叶片弱纹理、重复纹理、细结构多，直接 masked/crop 图会减少可匹配背景纹理，导致 SfM 位姿估计更不稳定。
4. 原图中背景虽然不应进入最终 Gaussian 表达，但对 COLMAP 求相机位姿可能是有帮助的。
5. mask 更适合用在：
   - COLMAP feature mask：只屏蔽明显干扰区域，同时保留足够几何上下文。
   - sparse point filtering：位姿用原图求，点云用 mask 去背景。
   - 2DGS training：作为 M3 的 `L_mask` 和 `L_bg-opacity`。
   - M5 meshing：作为边界距离和 boundary confidence。

推荐后续比较三种策略：

| 策略 | 用途 | 风险 |
|---|---|---|
| 原图/FFT 原图进 COLMAP | 当前优先策略，保留足够特征 | 背景点多，需要后处理过滤 |
| 原图 + COLMAP feature mask | 推荐后续主策略候选 | mask 过严会损失可注册特征 |
| masked/crop 图直接进 COLMAP | 可作为对照 | 当前已有明显失败证据 |

## 5. 本次实际重跑动作

新增脚本：

```text
04-COLMAP/rerun_failed_original_linux.py
```

输出目录：

```text
04-COLMAP-rerun-original/
```

设计原则：

- 非破坏重跑，不覆盖现有 `04-COLMAP`。
- 输入使用 `02-FFT/<sample>` 中 FFT 保留后的原始 RGB 帧。
- 优先使用 `sequential_matcher`。
- 如果注册率低于 70%，自动 fallback 到 `exhaustive_matcher`。
- mapper 生成多个模型时，自动选择注册图像数最多、点数最多的模型。
- 输出 `rerun_report.json` 记录每个样本的状态。

截至 2026-05-17，6 个问题样本均已完成非破坏重跑，并已同步到新版数据管理索引。

| sample | 当前动作 | 输出目录 | 当前状态 |
|---|---|---|---|
| BaiZhang | 已完成非破坏重跑 | `04-COLMAP-rerun-original/BaiZhang/sequential/` | success，209/214 注册，97.7%，72760 points |
| CaoMei2 | 已完成非破坏重跑 | `04-COLMAP-rerun-original/CaoMei2/sequential/` | success，210/210 注册，100.0%，110756 points |
| ChangShouHua1 | 已完成非破坏重跑 | `04-COLMAP-rerun-original/ChangShouHua1/exhaustive/` | sequential 仅 119/213，fallback exhaustive 后 212/213 注册，99.5%，83883 points |
| ChangShouHua3 | 已完成非破坏重跑 | `04-COLMAP-rerun-original/ChangShouHua3/exhaustive/` | sequential feature extraction 失败，fallback exhaustive 后 205/210 注册，97.6%，69008 points |
| KongQueZhuYu | 已完成非破坏重跑 | `04-COLMAP-rerun-original/KongQueZhuYu/sequential/` | success，210/210 注册，100.0%，177918 points |
| XiangPiShu2 | 已完成低内存非破坏重跑 | `04-COLMAP-rerun-original/XiangPiShu2/sequential_lowmem/` | 常规 sequential/exhaustive 特征提取被系统杀掉，低内存 sequential 后 211/211 注册，100.0%，120720 points |

说明：

- 第一次脚本运行因为相对路径和 `cwd` 组合错误，`feature_extractor` 立即失败，没有产生有效重建结果。
- 已修复脚本为绝对路径。
- 修复后先仅对 `BaiZhang` 做单样本验证，已验证成功。
- `XiangPiShu2` 常规 sequential 和 exhaustive 均在 `feature_extractor` 阶段以 exit code `-9` 结束，判断为内存压力导致，因此单独增加低内存配置：
  - `SiftExtraction.num_threads=4`
  - `SiftExtraction.max_image_size=2000`
  - `SiftExtraction.max_num_features=6000`
  - 关闭 affine shape 和 domain size pooling
  - `SiftMatching.num_threads=4`
  - sequential overlap 为 20，关闭 loop detection
- `XiangPiShu2` 低内存脚本和超参已存放到：
  - `数据管理/07-运行脚本与超参/S3-COLMAP位姿重跑与锁定/scripts/rerun_xiangpishu2_lowmem_colmap.py`
  - `数据管理/07-运行脚本与超参/S3-COLMAP位姿重跑与锁定/configs/xiangpishu2_lowmem_colmap.json`
  - `数据管理/07-运行脚本与超参/S3-COLMAP位姿重跑与锁定/docs/xiangpishu2_lowmem_colmap说明.md`

## 6. 当前有效重跑结果

| sample | matcher | input | registered | rate | points3D | output |
|---|---|---:|---:|---:|---:|---|
| BaiZhang | sequential | 214 | 209 | 97.7% | 72760 | `04-COLMAP-rerun-original/BaiZhang/sequential/` |
| CaoMei2 | sequential | 210 | 210 | 100.0% | 110756 | `04-COLMAP-rerun-original/CaoMei2/sequential/` |
| ChangShouHua1 | exhaustive | 213 | 212 | 99.5% | 83883 | `04-COLMAP-rerun-original/ChangShouHua1/exhaustive/` |
| ChangShouHua3 | exhaustive | 210 | 205 | 97.6% | 69008 | `04-COLMAP-rerun-original/ChangShouHua3/exhaustive/` |
| KongQueZhuYu | sequential | 210 | 210 | 100.0% | 177918 | `04-COLMAP-rerun-original/KongQueZhuYu/sequential/` |
| XiangPiShu2 | sequential_lowmem | 211 | 211 | 100.0% | 120720 | `04-COLMAP-rerun-original/XiangPiShu2/sequential_lowmem/` |

结论：

- 旧 `04-COLMAP` 中失败或低注册率的 6 个样本，均已获得可用于后续 baseline 的重跑候选。
- 新版索引已刷新：
  - `数据管理/00-规范与索引/dataset_index.csv`
  - `数据管理/00-规范与索引/dataset_index.json`
- 重跑候选已同步到：
  - `数据管理/02-位姿COLMAP/02-rerun_original_candidates/<sample>`
  - `数据管理/02-位姿COLMAP/03-final_locked/<sample>`
- 该结果进一步支持“COLMAP 位姿阶段使用 FFT 保留后的原图，而不是直接使用 masked/crop 图”的策略。

## 7. 后续建议

1. 不覆盖旧 `04-COLMAP/<sample>`，后续训练优先使用 `数据管理/02-位姿COLMAP/03-final_locked/<sample>`。
2. 进入 S4 前，先抽查 `final_locked` 中每个样本的 `images/`、`sparse/0/`、`input/` 是否完整。
3. baseline 阶段仍然不使用 mask、FSAM3、M1-H-VQG、M3、M4、M5，只跑纯 2DGS。
4. M1 的 Gate 3 可以复用本次 COLMAP 注册数、重投影误差和稀疏模型信息，作为后续 geometry reliability 的统计来源。
