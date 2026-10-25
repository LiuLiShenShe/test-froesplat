# 依赖管理迁移：conda → uv

**生效日期**：2026-10-21
**适用范围**：`F2DMAS` 仓库中本项目自有的代码树（不含第三方源码目录）

---

## 1. 迁移范围

### 已删除的 conda 环境文件

| 文件 | 说明 |
|---|---|
| `ForeSplat_real_release/environment.yml` | 仅含 `numpy/pillow/pyyaml` 的检查脚本环境 |
| `.tmp_foresplat_github/environment.yml` | 公开仓库副本的检查脚本环境 |
| `2d-gaussian-splatting-m2m3-floor40/environment.yml` | 生产 2DGS 训练树（M2M3 + Floor40） |
| `2d-gaussian-splatting-main/environment.yml` | 基线 2DGS 训练树 |

### 未处理的部分（说明）

- `SuGaR-main/`、`第三方源码/`、`instant-ngp/`、`segment-anything/` 等**第三方源码**目录保留其原有 conda/pip 文件。它们不是本项目自有代码，且本仓库未对其执行训练，保持原样以便追溯上游官方安装方式。
- `ForeSplat_real_release/requirements.txt`、`yolov8_seg/requirements.txt`、`exg_otsu/requirements.txt`、`plugins/gaussian_compression/requirements.txt`、`2d-gaussian-splatting-*/scripts/eval_tnt/requirements.txt` 等 `requirements.txt` **保留未删**，作为无 uv 时的回退安装方式与依赖来源记录。它们不再是推荐路径。

---

## 2. 新增的 uv 文件

| 位置 | 文件 | 用途 |
|---|---|---|
| `2d-gaussian-splatting-m2m3-floor40/` | `pyproject.toml`、`uv.lock`、`.python-version`、`setup_uv.sh` | 生产 2DGS 训练环境 |
| `2d-gaussian-splatting-main/` | 同上 | 基线 2DGS 训练环境 |
| `ForeSplat_real_release/` | `pyproject.toml`、`uv.lock`、`.python-version` | 发布包检查脚本 |
| `.tmp_foresplat_github/` | 同上 | 公开仓库副本 |
| `plugins/gaussian_compression/` | `pyproject.toml`、`uv.lock` | 后训练剪枝插件 |
| `yolov8_seg/` | `pyproject.toml`、`uv.lock` | YOLOv8 分割基线 |
| `exg_otsu/` | `pyproject.toml`、`uv.lock` | ExG+Otsu 分割基线 |

---

## 3. 使用方法

### 生产 / 基线 2DGS 训练树

```bash
cd 2d-gaussian-splatting-m2m3-floor40   # 或 2d-gaussian-splatting-main
./setup_uv.sh
source .venv/bin/activate
python train.py -s <scene> -m <output> ...
```

`setup_uv.sh` 做三件事：`uv python install 3.8`（确保有 3.8 解释器）、`uv sync`（安装锁定依赖）、`uv pip install --no-build-isolation ./submodules/diff-surfel-rasterization ./submodules/simple-knn`（编译两个 CUDA 扩展）。

> 两个 CUDA 扩展的 `setup.py` 会 `import torch`，因此**不能**放在 `[project].dependencies` 里由 uv 隔离构建（会因找不到 torch 而失败）。必须在 `uv sync` 之后单独安装。`pyproject.toml` 的注释里已写明这一点。

### 其余各树

```bash
cd <对应目录>
uv sync          # 首次自动按 .python-version 安装解释器并创建 .venv
uv run python <脚本>.py
```

---

## 4. conda 依赖 → uv 依赖对照

### `2d-gaussian-splatting-*/environment.yml`

| 原 conda 依赖 | 迁移后 | 说明 |
|---|---|---|
| `python=3.8.18` | `.python-version` = `3.8` + `requires-python = ">=3.8,<3.9"` | 实际解析为 CPython 3.8.20 |
| `pytorch=2.0.0` / `torchvision=0.15.0` / `torchaudio=2.0.0`（来自 `pytorch` + `nvidia` 频道） | `torch==2.0.0` / `torchvision==0.15.0` / `torchaudio==2.0.0`，经 `[tool.uv.sources]` 指向 `https://download.pytorch.org/whl/cu118` | 保持 CUDA 11.8 轮子，与原环境一致；锁定结果为 `2.0.0+cu118` / `0.15.0+cu118` |
| `typing_extensions=4.9.0` | `typing-extensions==4.9.0` | 精确一致 |
| `pillow=10.2.0` | `pillow==10.2.0` | 精确一致 |
| `pip=23.3.1` | 不再需要 | pip 不再是安装途径 |
| `ffmpeg=4.2.2` | **不迁移，见 §5** | conda 包，uv 无法安装系统二进制 |
| `open3d==0.18.0` | 同 | 精确一致 |
| `mediapy==1.1.2` / `lpips==0.1.4` / `scikit-image==0.21.0` / `tqdm==4.66.2` / `trimesh==4.3.2` | 同 | 精确一致 |
| `plyfile` / `opencv-python`（未固定版本） | `plyfile` 未固定；`opencv-python` 加 `>=4.8,<5` 约束 | 当前解析：`plyfile 1.1.3`、`opencv-python 4.14.0.94`。约束 `<5` 是必要的：不加约束会解析到 `opencv-python 5.0.0.93`，与 numpy 1.24 / Python 3.8 的既有运行环境不一致 |
| `numpy`（未固定） | `numpy>=1.24,<2` | 锁定 `numpy 1.24.4`，保持 numpy 1.x，避免 torch 2.0 与 numpy 2.x 的 ABI 冲突 |
| `submodules/diff-surfel-rasterization`、`submodules/simple-knn` | 见 §3 的手动安装命令 | 构建隔离问题，见上 |
| `defaults` 频道 | 不需要 | 所有依赖都来自 PyPI 或 PyTorch 索引 |

### `ForeSplat_real_release/environment.yml` 与 `.tmp_foresplat_github/environment.yml`

| 原 conda 依赖 | 迁移后 |
|---|---|
| `python=3.10` | `.python-version` = `3.10` + `requires-python = ">=3.10,<3.11"` |
| `numpy` / `pillow` / `pyyaml`（未固定） | `numpy>=1.24` / `pillow>=10.0` / `pyyaml>=6.0`（保持宽松区间） |
| `pip` | 不再需要 |
| `.tmp_foresplat_github` 的 `pip: opencv-python>=4.8` | `opencv-python>=4.8` |

---

## 5. 系统级依赖（uv 无法替代）

| 依赖 | 原来源 | 迁移后要求 |
|---|---|---|
| `ffmpeg 4.2.2` | conda-forge | 系统级安装。本机当前为 **ffmpeg 6.1.1**（`ffmpeg -version`），与原固定的 4.2.2 不同。本项目的 ffmpeg 用法（`scripts/run_batch_ffmpeg_2dgs.sh`、`prepare_filtered_inputs.py`）只做视频帧提取，6.1.1 兼容；若需与历史环境严格一致，请单独安装 4.2.2 二进制。 |
| CUDA Toolkit / nvcc | conda `nvidia` 频道 + `cudatoolkit` | 系统级 NVIDIA 驱动与 CUDA Toolkit。本机已有 `/usr/local/cuda-12.6`，`nvcc` 12.6。编译两个 CUDA 扩展需要它；运行 `torch 2.0.0+cu118` 需要驱动支持 CUDA 11.8（当前驱动 590.48 向后兼容）。 |
| COLMAP | 系统二进制 | 独立安装，与 Python 环境无关。 |

---

## 6. 复现冻结实验的方式

原冻结实验（2026-02 至 2026-10）是在 `surfel_splatting`（conda）与 `foresplat`（conda）环境中运行的。**不要尝试用 uv 环境去"复现"这些历史数值**——uv 解析出的 Python 3.8.20、numpy 1.24.4、opencv 4.14 等与原 conda 环境（Python 3.8.18 等）并非逐位相同，重新训练会产生不同数值。

uv 环境的定位是：

1. **新实验**（例如 `01_FINAL_METHOD_DEFINITION.md` / `07_FINAL_DECISION_REPORT.md` 中建议补做的 V1 正式运行、A5 第二场景）——直接从 uv 环境运行，并在提交时记录 `uv.lock` 的内容哈希。
2. **代码调试与回归测试**——同一套锁定依赖即可。
3. **发布与交付**——`pyproject.toml` + `uv.lock` 是可审计的依赖快照，比 `environment.yml`（未锁定传递依赖）更强。

历史实验的可追溯性由 `MANUSCRIPT_EVIDENCE_REGISTRY.csv` 的 `Source_File` 列与各 `command.txt` 记录保证，不由 uv 承担。

---

## 7. 已验证事项

- `uv lock` 在 7 个目录全部成功解析。
- `uv sync --dry-run` 在生产 2DGS 树上确认 101 个包可安装，torch 解析为 `2.0.0+cu118`。
- `setup_uv.sh` 通过 `bash -n` 语法检查。
- `git status` 显示仅新增/修改上述文件，未触碰生产代码、实验配置或冻结数据。

## 8. 已知取舍

1. **opencv-python 需加 `<5` 上限**：PyPI 已有 5.x 版本，不加约束会改变解析结果。
2. **numpy 需加 `<2` 上限**：torch 2.0 未针对 numpy 2.x 编译。
3. **CUDA 扩展必须单独安装**：见 §3。
4. **ffmpeg 不再被版本固定**：如需严格一致，须自行管理系统 ffmpeg。
