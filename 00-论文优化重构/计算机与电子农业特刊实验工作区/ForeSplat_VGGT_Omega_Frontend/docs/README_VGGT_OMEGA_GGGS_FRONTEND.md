# ForeSplat × VGGT-Ω 前端（社区仓库之上的适配层）

在社区开源仓库 **tomhol/vggt-omega** 之上开发的 ForeSplat 专用前端：
`RAP-FSAM3 前景图像 + 二值掩膜 → VGGT-Ω 推理 → 前景掩膜约束的点云 → COLMAP 导出 → GGGS 数据集`。

- 本阶段只做 VGGT-Ω 前端与数据适配；**不修改 GGGS 的 Gaussian 优化/曝光/损失，也不实现 NKSR**。
- 命令行驱动，**不依赖 Gradio**。

---

## 1. 环境安装

需要 CUDA GPU（实测 RTX A6000 48GB / torch 2.10.0+cu128）。

```bash
# 项目根目录
cd ForeSplat_VGGT_Omega_Frontend

# 虚拟环境（复用宿主已装好的 torch/torchvision，python 3.12）
uv venv --python /usr/bin/python3 --system-site-packages .venv
source .venv/bin/activate

# 适配层依赖（pycolmap 必须是 3.10.x：Image(points2D=...)/ListPoint2D/Track/add_point3D API）
uv pip install --python .venv/bin/python "pycolmap==3.10.0" trimesh plyfile opencv-python scipy matplotlib requests
```

版本约束（实测）：

| 包 | 版本 | 原因 |
|---|---|---|
| torch / torchvision | 2.10.0+cu128 / 0.25.0 | VGGT-Ω 需要 fp16 autocast |
| pycolmap | **3.10.0**（不要 4.x） | `4.x` 用 `image_id` + frame/rig 结构且无 `ListPoint2D`；`3.11+` 的 `cam_from_world` 变只读；社区 `save_colmap()` API 只匹配 `3.10.x` |
| numpy / scipy / opencv / trimesh / plyfile | 任意近期 | COLMAP 导出 + PLY + 掩膜 |

### GGGS 依赖（仅验证用，可选）

验证脚本会 `sys.path` 指向一个 GGGS 克隆并导入 `scene.dataset_readers`。
GGGS 需要 `open3d`、`plyfile`、`trimesh`，以及训练路径的 CUDA 扩展
`simple-knn`（`submodules/simple-knn`）。只做**数据读取验证**时不需要编译扩展，
脚本会自动为缺失的 `open3d` / `simple_knn._C` 打桩。
若要真正跑 `train.py`，需在 GGGS 仓库里按其 README 编译 `submodules/*`
（CUDA 12.6 需给 `simple_knn.cu` 补 `#include <cfloat>`，这是上游已知问题）。

---

## 2. 模型权重

下载（ModelScope 镜像，与 HuggingFace `facebook/VGGT-Omega` 同一文件）：

```bash
mkdir -p weights && curl -L -o weights/vggt_omega_1b_512.pt \
  "https://modelscope.cn/api/v1/datasets/hengyelyu/vggt-omega-weights/repo?Revision=master&FilePath=vggt_omega_1b_512.pt"
```

校验（4,576,706,117 bytes）：

```bash
sha256sum weights/vggt_omega_1b_512.pt
# 期望 c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934
```

加载方式：`torch.load(pt)` → `VGGTOmega().load_state_dict(sd, strict=False)`；
实测 **missing=0, unexpected=0**，参数量 1.14B。

---

## 3. 运行

### 3.1 主流程：一键导出

```bash
python tools/run_vggt_omega_gggs_export.py \
    --images-dir /path/to/rap_fsam3/前景图 \
    --masks-dir  /path/to/rap_fsam3/最终掩膜 \
    --checkpoint weights/vggt_omega_1b_512.pt \
    --output-dir /path/to/GGGS_Dataset \
    --image-resolution 512 \
    --preprocess-mode balanced \
    --confidence-percentile 20 \
    --max-points 1000000
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--image-resolution` | 512 | VGGT-Ω 输入边长（token 数近似 `res/16` 的平方） |
| `--preprocess-mode` | balanced | `balanced` / `max_size`，与社区一致 |
| `--patch-size` | 16 | 必须整除 `--image-resolution` |
| `--confidence-percentile` | 20 | 只在**前景点**上取分位数 |
| `--max-points` | 1000000 | 点云上限（可复现随机下采样，`--seed`） |
| `--frame-stride` / `--max-frames` | 1 / None | 仅做视图子集，**不改顺序** |
| `--filter-depth-edges` | 关 | 深度边缘过滤（默认关：保护叶缘点） |
| `--depth-edge-rtol` | 0.03 | 与社区一致 |
| `--mask-threshold` | 127 | 掩膜二值化阈值 |
| `--foreground-background` | black | RGBA 合成背景色（与 RAP-FSAM3 一致） |
| `--image-prefix-to-strip` / `--mask-prefix-to-strip` | None | RAP-FSAM3 目录用 `crop_xxxx.png` / `mask_xxxx.png` 时配对用 |
| `--no-align` | — | 不施加社区场景对齐矩阵 |
| `--save-predictions` | 关 | 把 depth/conf/pose 存到 `debug/predictions.npz` |
| `--gggs-repo` | None | 指定 GGGS 克隆，导出后自动跑 `readColmapSceneInfo` 验证 |
| `--device` | auto | `cuda` / `cpu` |

RAP-FSAM3 输出目录与本工具假设结构不同时，只需 `--images-dir` / `--masks-dir`
指定目录（必要时加前缀剥离参数），**不改分割算法本身、不重跑分割**。

### 3.2 几何诊断（强烈建议每次导出后跑）

```bash
python tools/diagnose_reconstruction.py \
    --dataset-dir /path/to/GGGS_Dataset \
    --gggs-repo /path/to/GGGS
```

输出：逐帧焦距一致性、相机基线矩阵、跨视角深度一致性（中位相对误差与内点率）。

### 3.3 只跑单元测试（不需要权重）

```bash
python -m pytest tests/ -q      # 19 passed
```

覆盖：RGB 路径与社区 loader 逐元素一致、Mask 与 RGB 几何一致（裁剪/缩放/padding/padding 区域为零）、
仿射矩阵 `A` 与 PIL 重采样质心吻合、内参回映射、COLMAP 导出（单观测 track、
`K_original = A^-1 K_model`、观测与点在各自相机下的重投影一致性、对齐与不对齐两种模式）、
`crop_ / mask_` 前缀配对与顺序保持。

---

## 4. 代码结构

```
ForeSplat_VGGT_Omega_Frontend/
├── vggt_omega/                     社区代码（未改网络/权重/推理逻辑）
│   └── utils/load_fn.py            仅新增：掩膜感知预处理 + 仿射变换工具
├── visual_util.py                  扩展社区 save_colmap()：前景掩膜 + 仿射逆变换
├── foresplat_adapter/
│   ├── dataset.py                  图像/掩膜目录配对（保留顺序、前缀剥离）
│   ├── inference.py                独立推理（无 Gradio）
│   ├── export.py                   GGGS 目录装配（COLMAP + PLY + metadata + debug）
│   ├── ply.py                      二进制 PLY 写出
│   └── verify.py                   用 GGGS readColmapSceneInfo 做回读验证
├── tools/
│   ├── run_vggt_omega_gggs_export.py   主入口（预处理+推理+导出+GGGS 验证）
│   ├── diagnose_reconstruction.py      几何诊断（焦距一致性/基线/跨视角深度）
│   └── visualise_export.py             点云三视图 + 掩膜叠加预览
├── tests/test_preprocessing_and_export.py
├── docs/VGGT_OMEGA_TO_GGGS.md      转换语义说明（重点阅读）
├── docs/RUN_REPORT.md              真实运行结果与可视化
├── docs/README_VGGT_OMEGA_GGGS_FRONTEND.md   本文（安装/权重/命令/结构）
└── weights/                        权重下载目录
```

---

## 5. 兼容性要点（实测踩坑记录）

1. **pycolmap 版本**：社区 `save_colmap()` 需要 `pycolmap==3.10.x`；4.x/3.11+ API 不兼容。
2. **事后回写失效**：pycolmap ≥ 3.x 中 `points3D[id].xyz = ...` 拿到副本，
   社区版"先加点再对齐"的写法会静默不生效；本适配层把对齐提前到创建时施加。
3. **相机主点约定**：VGGT-Ω 用 `(W/2, H/2)`；不能改成 `(W-1)/2`，否则与其反投影不自洽。
4. **GGGS 只认 PINHOLE**，且相机 `width/height` 必须与 `images/` 下图像尺寸一致。
5. **points3D.ply 需带法线字段**（`nx,ny,nz`），否则 GGGS 的 `fetchPly()` 直接抛异常。

---

## 6. 可视化

```bash
python tools/visualise_export.py --dataset-dir /path/to/GGGS_Dataset
```

生成 `debug/preview_pointcloud.png`（前景点云三视图）与 `debug/preview_mask_overlay.png`
（前景 RGB + 掩膜叠加）。

---

## 7. 全序列批处理（Phase 2）

按植物批处理：**全序列单次联合推理 → 多视角融合 → GGGS 数据集**。
详见 `docs/FULL_SEQUENCE_BATCH.md`。

```bash
# 1) 数据发现 + 损坏帧隔离 + 生成 batch_manifest.csv
python tools/build_batch_manifest.py \
    --root /data/fj/F2DMAS/03-SAM \
    --output-root runs/full_sequence_batch \
    --manifest runs/full_sequence_batch/batch_manifest.csv

# 2) 批处理（GPU 1 only；最大规模植物优先；--resume 断点续跑）
CUDA_VISIBLE_DEVICES=1 python tools/run_all_plant_datasets.py \
    --manifest runs/full_sequence_batch/batch_manifest.csv \
    --output-root runs/full_sequence_batch \
    --checkpoint weights/vggt_omega_1b_512.pt \
    --gggs-repo gggs_community \
    --order largest_first --resume
```

关键约束：只允许 GPU 1（`CUDA_VISIBLE_DEVICES=1`）；依赖只用 uv（禁 conda）；
OOM 沿 `balanced@512 → max_size@512 → ... → max_size@384` 阶梯降级，**绝不丢帧**；
损坏帧（03-SAM 实测每株 1–15 个）隔离登记后剔除，其余可用帧全部处理。
