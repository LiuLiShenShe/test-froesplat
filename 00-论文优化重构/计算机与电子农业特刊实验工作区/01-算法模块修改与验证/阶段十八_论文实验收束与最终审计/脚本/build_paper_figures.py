#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 18 — build_paper_figures.py
论文用图构建脚本 | Paper-ready figure builder

Deterministic, no internet, no inference, READ-ONLY on historical evidence.
Bilingual labels: English primary, 中文 second line.

Outputs (figures/ subdirectory under this script's parent):
  Figure_A_failure_mechanism.pdf / .png
  Figure_B_ranking_illustration.pdf / .png
  Figure_C_easy_vs_hard_distribution.pdf / .png
  Figure_D_A6A7_ablation.pdf / .png

Sources:
  - Raw instance masks, RGB images, GT masks, P16/P17A assets (read-only)
  - Phase17C ranking_benchmark.csv (computed R1 primary formula — allowed CSV-column arithmetic)
  - Phase12 P6_全部21帧.csv, Phase17C policy_comparison.csv (Figure C distributions)
  - Phase14.1 manifest (Figure D bar chart)
"""
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import fontManager, FontProperties
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PHASE18_DIR = os.path.dirname(HERE)
FIG_DIR = os.path.join(PHASE18_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BASE = "/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证"
P17A = os.path.join(BASE, "阶段十七A_10GT困难样本Pilot与无GT测试审计")
P16  = os.path.join(BASE, "阶段十六_HardCase_GT构建与锁定")
P141 = os.path.join(BASE, "阶段十四点一_A7有效性封口")
P12  = os.path.join(BASE, "阶段十二_GT_v2_QA与P6正式验收/指标")
P17C = os.path.join(BASE, "阶段十七C_候选排序修复开发")

# CJK font registration
fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "rgb":    "#FFFFFF",
    "gt":     "#2ECC71",
    "r0":     "#E74C3C",
    "oracle": "#3498DB",
    "r1":     "#3498DB",
    "oracle_r1": "#3498DB",
}
DPI = 200

# ---------------------------------------------------------------------------
# paths (failure frames)
# ---------------------------------------------------------------------------
FAIL_FRAMES = [
    ("DouBanLv2", 39, "0039"),
    ("DouBanLv2", 40, "0040"),
    ("DouBanLv2", 41, "0041"),
]

def rgb_path(sample, num):
    return os.path.join(P17A, "00_input_trackA", f"{sample}_{num:04d}.jpg")

def gt_path(sample, num):
    return os.path.join(P16, "GT_potted_clean_challenge", sample,
                        f"mask_potted_clean_{num:04d}.png")

def a5c_mask_path(sample, num):
    return os.path.join(P17A, "02_trackA_variants", "P00_Control",
                        "A5c_final_mask", f"mask_{sample}_{num:04d}.png")

RAWINST = os.path.join(P17A, "02_trackA_variants", "P00_Control",
                        "候选掩膜", "raw_instance_P6")

def raw_mask_path(candidate_id):
    return os.path.join(RAWINST, f"mask_{candidate_id}.png")

# ---------------------------------------------------------------------------
# Figure A — Failure mechanism
# ---------------------------------------------------------------------------
print("Figure A: failure mechanism...")

# benchmark data (instance roles per failure frame)
bench = pd.read_csv(os.path.join(P17C, "Phase17C_ranking_benchmark.csv"))
bench["frame_s"] = bench["frame"].astype(int).astype(str).str.zfill(4)
bench["has_gt"] = bench["has_gt"].astype(int)

# policy comparison (F1 values for annotations)
pc = pd.read_csv(os.path.join(P17C, "Phase17C_policy_comparison.csv"))
pc["frame_s"] = pc["frame"].astype(str).str.zfill(4)

def get_failure_frame_info(sample, frame_str, frame_int):
    """Return dict with baseline_id, oracle_id, baseline_F1, oracle_F1."""
    b = bench[(bench["source"] == "DEV10") & (bench["frame_s"] == frame_str) & (bench["has_gt"] == 1)]
    oracle_row  = b[b["oracle_candidate"] == 1]
    baseline_row = b[b["baseline_selected"] == 1]
    # F1 values from policy_comparison R0
    pc_row = pc[(pc["source"] == "DEV10") & (pc["policy"] == "R0") & (pc["frame_s"] == frame_str)].iloc[0]
    return {
        "baseline_id": f"{sample}_{frame_int:04d}_{int(baseline_row.iloc[0]['instance_id']):02d}",
        "oracle_id":   f"{sample}_{frame_int:04d}_{int(oracle_row.iloc[0]['instance_id']):02d}",
        "baseline_F1": float(pc_row["baseline_F1"]),
        "oracle_F1":   float(pc_row["oracle_F1"]),
    }

def load_mask_binary(path):
    img = mpimg.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return (img > 127) if img.max() > 1 else (img > 0.5)

fig, axes = plt.subplots(len(FAIL_FRAMES), 5, figsize=(15, 3.3 * len(FAIL_FRAMES)))
for row_i, (sample, frame_int, frame_str) in enumerate(FAIL_FRAMES):
    finfo = get_failure_frame_info(sample, frame_str, frame_int)
    rgb = mpimg.imread(rgb_path(sample, frame_int))
    gt  = load_mask_binary(gt_path(sample, frame_int))
    r0  = load_mask_binary(raw_mask_path(finfo["baseline_id"]))
    oc  = load_mask_binary(raw_mask_path(finfo["oracle_id"]))

    panels = [
        ("RGB\nRGB图像",                       rgb,           None),
        ("Ground Truth\n真值",                  gt,            COLORS["gt"]),
        (f"R0 selected (F1={finfo['baseline_F1']:.4f})\n基线误选", r0,  COLORS["r0"]),
        (f"Oracle (F1={finfo['oracle_F1']:.4f})\n最优候选",          oc,  COLORS["oracle"]),
        (f"R1 selected (=oracle)\nR1修复选择",    oc,  COLORS["oracle_r1"]),
    ]

    for col_i, (title, data, c) in enumerate(panels):
        ax = axes[row_i, col_i]
        if c is None:      # RGB
            ax.imshow(data)
        else:              # binary mask on WHITE background
            h, w = data.shape[:2]
            canvas = np.ones((h, w, 3), dtype=np.float64)
            # mask region → colored
            mask_rgb = np.array([int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)], dtype=np.float64)/255
            canvas[data] = mask_rgb * 0.7
            # background → light grey
            canvas[~data] = 0.92
            ax.imshow(np.clip(canvas, 0, 1))
        ax.set_title(title, fontsize=8, linespacing=1.3)
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle("Figure A — Failure Mechanism: DouBanLv2 0039/0040/0041\n"
             "图A — 失败机制：对比基线误选 vs R1修复选择",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "Figure_A_failure_mechanism.png"), dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "Figure_A_failure_mechanism.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Figure A OK")

# ---------------------------------------------------------------------------
# Figure B — Candidate ranking illustration (0039 only, 2 GT-scope candidates)
# ---------------------------------------------------------------------------
print("Figure B: ranking illustration...")

FRAME_INT = 39
b39 = bench[(bench["source"] == "DEV10") & (bench["frame"] == FRAME_INT) & (bench["has_gt"] == 1)].copy()
# compute R1 primary = total_score - q_contrast / 5.5
b39["R1_primary"] = b39["total_score"] - b39["q_contrast"] / 5.5

roles = []
for _, r in b39.iterrows():
    if r["oracle_candidate"]:
        roles.append("Oracle (whole-plant)\n最优候选 (完整植株)")
    elif r["baseline_selected"]:
        roles.append("R0 distractor\n基线误选杂散实例")
    else:
        roles.append("other\n其他候选")

bar_metrics = [
    ("q_contrast\n对比度", "q_contrast", 0.6),
    ("q_area\n面积",       "q_area",     0.5),
    ("q_comp\n连通域",     "q_comp",     0.4),
    ("q_edge\n边界",       "q_edge",     0.3),
    ("total_score\n基线总分", "total_score", 0.4),
    ("R1 primary\nR1主评分", "R1_primary",  0.4),
]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(roles))
width = 0.12
for mi, (ylabel, col, _) in enumerate(bar_metrics):
    vals = b39[col].values
    bars = ax.bar(x + mi*width - width*len(bar_metrics)/2, vals, width, label=ylabel, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(roles, fontsize=8, linespacing=1.3)
ax.set_ylabel("Score\n得分", fontsize=9)
ax.set_title("Figure B — Candidate Feature Comparison (DouBanLv2 0039)\n"
             "图B — 候选特征对比 (0039)", fontsize=10)
ax.legend(fontsize=7, ncol=3, loc="upper left")
ax.axhline(y=0, color="grey", linewidth=0.5)

# Add ranking annotation
oldest_rank = 2  # distractor has HIGHER total_score
ax.annotate("R0 selects distractor\nR0选择了杂散实例",
            xy=(x[1] + width*4 - width*len(bar_metrics)/2, b39.iloc[1]["total_score"]),
            xytext=(1.2, 0.68), fontsize=7, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
ax.annotate("R1 selects oracle\nR1选择了最优候选",
            xy=(x[0] + width*4 - width*len(bar_metrics)/2, b39.iloc[0]["R1_primary"]),
            xytext=(-0.3, 0.54), fontsize=7, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue", lw=0.8))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "Figure_B_ranking_illustration.png"), dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "Figure_B_ranking_illustration.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Figure B OK")

# ---------------------------------------------------------------------------
# Figure C — Easy vs Hard distribution
# ---------------------------------------------------------------------------
print("Figure C: easy vs hard distribution...")

p6_frames = pd.read_csv(os.path.join(P12, "P6_全部21帧.csv"))
easy21_f1 = p6_frames["f1"].values

dev10_r0  = pc[(pc["source"] == "DEV10") & (pc["policy"] == "R0")]["new_F1"].values
dev10_r1  = pc[(pc["source"] == "DEV10") & (pc["policy"] == "R1")]["new_F1"].values

data_list = [easy21_f1, dev10_r0, dev10_r1]
labels    = ["Easy21 baseline\nEasy21基线",
             "GT10 baseline (R0)\nGT10基线",
             "GT10 R1\nGT10修复"]
colors    = ["#95A5A6", "#E74C3C", "#3498DB"]

fig, ax = plt.subplots(figsize=(6, 4))
vp = ax.violinplot(data_list, positions=range(len(data_list)), showmeans=True, showmedians=True)
for i, (body, col) in enumerate(zip(vp["bodies"], colors)):
    body.set_facecolor(col)
    body.set_alpha(0.5)
for attr in ["cmeans", "cmedians"]:
    vp[attr].set_color("black")

ax.set_xticks(range(len(data_list)))
ax.set_xticklabels(labels, fontsize=8, linespacing=1.3)
ax.set_ylabel("F1 Score", fontsize=9)
ax.set_title("Figure C — Easy vs Hard Distribution (descriptive)\n"
             "图C — 易/难集分布 (描述性)", fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.axhline(y=0, color="grey", linewidth=0.5)

# Annotate means
for i, d in enumerate(data_list):
    mean = np.mean(d)
    ax.text(i + 0.18, mean + 0.02, f"μ={mean:.4f}", fontsize=7, ha="center")

# Note: descriptive only — no significance stars per protocol §19
ax.text(0.5, -0.12, "Descriptive only — no significance test performed\n仅描述性展示 — 未进行显著性检验",
        transform=ax.transAxes, ha="center", fontsize=7, color="grey")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "Figure_C_easy_vs_hard_distribution.png"), dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "Figure_C_easy_vs_hard_distribution.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Figure C OK")

# ---------------------------------------------------------------------------
# Figure D — A6/A7 ablation (Easy21 F1 means)
# ---------------------------------------------------------------------------
print("Figure D: A6/A7 ablation...")

m141 = pd.read_json(os.path.join(P141, "Phase14_1_manifest.json"), typ="series")
variants = m141["variants"]
v00 = float(variants["V00"]["f1_mean"])
v10 = float(variants["V10"]["f1_mean"])
v01 = float(variants["V01_R1"]["f1_mean"])
v11 = float(variants["V11_R1"]["f1_mean"])

bars_data = [
    ("V00\nControl\n对照",     v00, "#95A5A6"),
    ("V10\nA6 ON\nA6开启",     v10, "#E67E22"),
    ("V01\nA7 ON\nA7开启",     v01, "#9B59B6"),
    ("V11\nA6+A7 ON\nA6+A7开启", v11, "#E74C3C"),
]

fig, ax = plt.subplots(figsize=(5, 4))
x = np.arange(len(bars_data))
heights = [b[1] for b in bars_data]
cols    = [b[2] for b in bars_data]
ax.bar(x, heights, color=cols, alpha=0.8, width=0.5)

ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bars_data], fontsize=8, linespacing=1.3)
ax.set_ylabel("Mean F1\n平均F1", fontsize=9)
ax.set_title("Figure D — A6/A7 Ablation on Easy21 (descriptive)\n"
             "图D — A6/A7 消融 (描述性)", fontsize=10)
ax.set_ylim(0.976, 0.986)

# Annotate each bar
for i, (_, h, _) in enumerate(bars_data):
    ax.text(i, h + 0.0001, f"{h:.4f}", ha="center", va="bottom", fontsize=8)

ax.text(0.5, -0.14, "A6: E_A6=−0.0028, 1 material regression (ChangShouHua2_0075)\n"
        "A7: E_A7≈0 (valid exposure after Phase14.1 lifecycle fix)\n"
        "Neither provides consistent measurable benefit → disabled by default",
        transform=ax.transAxes, ha="center", fontsize=6.5, color="#555555")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "Figure_D_A6A7_ablation.png"), dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "Figure_D_A6A7_ablation.pdf"), bbox_inches="tight")
plt.close(fig)
print("  Figure D OK")

print("\nAll figures written to:", FIG_DIR)
print("  4 PDF + 4 PNG (8 files total)")
