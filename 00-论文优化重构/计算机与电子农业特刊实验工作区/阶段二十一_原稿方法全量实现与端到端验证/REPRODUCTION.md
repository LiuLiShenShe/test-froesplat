# Phase 21 — 复现说明

所有命令都在 `/data/fj/F2DMAS` 下执行。解释器统一使用
`2d-gaussian-splatting-m2m3-floor40/venv/bin/python`（torch 2.5.1+cu121，
自带 `diff_surfel_rasterization`、`simple_knn`、`scipy`、`cv2`、`skimage`）。

```bash
export GS=/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40
export PY=$GS/venv/bin/python
export W=/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十一_原稿方法全量实现与端到端验证
```

---

## 1. 单元 / 契约测试（约 20 s，无需 GPU）

```bash
cd "$W/tests" && python3 -m pytest . -q --no-header -p no:cacheprovider
```

54 项：SG-MPS+R1（10）、M2M3-GO（14，含 τ_p 解析模式、densify 选择、
split 属性张量、prune/densify 同轮索引重映射）、FFT（5）、掩膜同一性（5）、
前景指标（6，含 LPIPS 后端解析）、GCFC（7）、禁用路径 AST 不变量（3）、
端到端 runner 契约（4）。

历史回归（阶段 11–18，150 项）：

```bash
cd /data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区
python3 -m pytest 01-算法模块修改与验证 -q --no-header -p no:cacheprovider
```

---

## 2. FFT 公式对照（Eq.(1) vs legacy）

```bash
cd "$W"
for f in legacy paper; do
  python3 /data/fj/F2DMAS/02-FFT/blur_filter.py \
    --input_dir "/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/02-fft_frames/KongQueZhuYu" \
    --output_dir "runs/fft_formula/$f" --fft_formula "$f" --blur_percentile 25
done
```

产出 `runs/fft_formula/{legacy,paper}/filter_log.json`；
逐帧对照表由 `metrics/fft_formula_kqz.csv` 给出（210 帧，33 帧判定不同）。

---

## 3. RAP-FSAM3 选择机制消融（SG-MPS × 排名策略）

```bash
cd "$W" && python3 scripts/run_sg_mps_ablation.py --run-name sg_mps_ablation
```

6 个配置（S0 base / S1 r1 / S2 heuristic 门控 / S3 τ=0.25 / S4 τ=0.10 /
S5 τ=0.25+contrast），复用冻结的 `E2E3_KongQueZhuYu_GT5_A1_多提示词选择`
候选缓存（P1–P5），不跑 SAM3 推理，CPU 约 15 s/帧。
产物：

- `runs/sg_mps_ablation/<config>/`（完整 RAP 输出，含 `语义门控评分.csv`）
- `metrics/sg_mps_ablation_summary.csv`、`..._per_frame.csv`
- `logs/sg_mps_<config>.txt`

---

## 4. 历史回放一致性验证

```bash
cd "$W"
python3 - <<'EOF'
import hashlib
from pathlib import Path
A=Path("../阶段二十C_FIX_候选元数据与精简细化验证/runs/V0/Easy21/最终掩膜")
B=Path("runs/replay_check/Easy21_V0/最终掩膜")
h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
same=sum(1 for f in sorted(A.glob("*.png")) if (B/f.name).exists() and h(f)==h(B/f.name))
print("identical", same, "of", len(list(A.glob('*.png'))))
EOF
```

回放命令（Phase21 代码，`ranking_policy=base`）：

```bash
python3 "$W/scripts/run_pipeline.py" --job "$W/configs/smoke_rap_only.json" --force
```

---

## 5. GCFC 后端

单元测试：`cd "$W/tests" && python3 -m pytest test_phase21_gcfc.py -v`。

在 RAP 输出中使用（需要与掩膜匹配的 COLMAP `sparse/0` 束）：

```bash
python3 "数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py" \
  --input_dir <帧目录> --output_dir <输出目录> \
  --reuse_existing_candidates --candidate_source_dir <候选缓存> \
  --use_corrective_geometry --corrective_geometry_backend colmap_tracks_gaussian \
  --geometry_gaussian_sigma 3.0 --geometry_support_threshold 0.15 \
  --geometry_boundary_band_px 12 --geometry_min_support_ratio 0.35 \
  --geometry_correct_max_delta_ratio 0.08 --geometry_correct_min_iou 0.75
```

报告字段 `支撑场比例G` 记录 `G_i`；`接受=0/原因=support_gate` 表示支撑不足未修正。

---

## 6. 前景物体 2DGS + M2M3-GO 消融（M0–M4）

```bash
cd "$W"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_m2m3_ablation.py \
  --sample WanNianQing1 --group C_R1 \
  --variants M0,M1,M2,M3,M4 --iterations 30000 --resolution 2 \
  --run-name m2m3_ablation --port-base 6030
```

每档位生成一个 job JSON（`configs/m2m3_ablation/`）并交给
`scripts/run_pipeline.py` 执行 `train → eval`。产物：

- `runs/m2m3_ablation/<sample>_<group>_<variant>/model/`（含 `mask_provenance.json`）
- `runs/m2m3_ablation/<...>/eval/foreground_test.json`（前景指标逐视角）
- `runs/pipeline/<job>/run_manifest.json`（git commit、命令、耗时、掩膜哈希）
- `logs/pipeline/<job>/train.log`、`eval.log`

汇总（只读已落盘产物；缺结果标 `MISSING_EVAL`，不补数）：

```bash
python3 scripts/summarize_m2m3_ablation.py --run-name m2m3_ablation
# -> metrics/m2m3_ablation_summary.csv / metrics/m2m3_ablation_summary.md
```

无注入控制组（论文 Table 6 的“不做前景物体注入”一行）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_m2m3_ablation.py \
  --sample WanNianQing1 --group C_R1 --variants C0 \
  --iterations 30000 --resolution 2 --run-name m2m3_ablation --port-base 6100
```

`C0` 与 M0–M4 共用同一输入、分辨率、迭代数与评价器，但**不加**前景初始化、
前景 RGB 监督、掩膜/背景不透明度损失与视角质量加权，即普通全场景 2DGS。

重跑被中断的档位前，先把半成品目录与 manifest 移到 `archive_*`（保留失败证据），
再加 `--force`；否则 `run_pipeline.py` 会拒绝覆盖已有 manifest：

```bash
mv runs/m2m3_ablation/<job> runs/m2m3_ablation/archive_<job>_<reason>
mv logs/pipeline/<job> logs/pipeline/archive_<job>_<reason>
mv runs/pipeline/<job>/run_manifest.json runs/pipeline/archive_<job>_<reason>/
python3 scripts/run_m2m3_ablation.py ... --force
```

`/data` 为机械盘；本阶段期间被其它租户压到 ~1 MB/s（203 段导入曾耗时 1 h 48 min），
213 视角加载 8–70 min 不等；块设备健康时 30k 迭代约 25 min/档位。
两台 GPU 并行跑不同档位时用不同的 `--port-base`（避免 2DGS network_gui
端口 6009 冲突），例如 GPU0 用 `--port-base 6030`、GPU1 用 `--port-base 6050`。
但**内存不允许并行**：125 GB 被其它会话占用时并行两档会触发内核 OOM killer，
务必串行（`scripts/run_ablation_chain.sh`、`scripts/run_m2_after_chain.sh`、
`scripts/run_c0_after_m2.sh` 就是为此写的串行接力脚本）。

---

## 7. 前景指标单独评价已有 checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 $PY "$GS/scripts/evaluate_foreground.py" \
  -m <model_path> -s <dataset_root> --images images_rgba --resolution 2 --eval \
  --iteration <iter> --mask_dir <masks> --split test --boundary_ignore 2 \
  --with_lpips --output_json <out.json>
```

`summary.lpips_backend` 记录实际使用的 LPIPS 实现。仓库自带
`lpipsPyTorch/modules/lpips.py` 已损坏（内容是 TURN 日志乱码，`import` 报
`SyntaxError`），因此这里回退到 venv 内的 pip 包 `lpips`（0.1.4），字段值为
`lpips-pip(vgg)`。若换到完好的 `lpipsPyTorch` 副本，字段会变成
`lpipsPyTorch`，两个实现都对相同输入给出可比数值。

---

## 8. 端到端 runner

job JSON 见 `configs/smoke_rap_only.json`（单阶段示例）与
`configs/m2m3_ablation/*.json`（train+eval）。四阶段
`fft / rap / train / eval` 可任意组合；dry-run 只打印命令：

```bash
python3 "$W/scripts/run_pipeline.py" --job <job.json> --dry-run
python3 "$W/scripts/run_pipeline.py" --job <job.json> --output-root <run_root>
```

上游失败或缺字段 → 下游阶段在 manifest 中记为 `BLOCKED` 并中止该阶段，
不会静默替换为基线模型或基线掩膜。
