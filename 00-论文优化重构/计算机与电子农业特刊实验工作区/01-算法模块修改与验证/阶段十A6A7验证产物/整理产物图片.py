#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理 RAP-FSAM3 v2（A6+A7+证据引导SPNP）四样本 P2/P6 对比产物图片。

产物分类：
  - 成功产物：F1 >= 0.9 且 GT 口径干净
  - GT 混标：F1 高但 GT 标注口径需人工核对（不计入失败）
  - 失败产物：F1 < 0.6，SAM3 候选级失败（管道未能纠正）

输出目录（位于阶段十A6A7验证产物/）：
  - 成功产物/        —— 每帧三联图（P2 | GT轮廓 | P6）
  - 失败产物/        —— 每帧三联图 + 失败模式说明
  - GT混标待核/      —— 每帧三联图 + 混标说明
  - 冒烟证据图/      —— 从 /tmp 固化的 A6/A7/证据引导SPNP 三联/四联图
"""

import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ── 路径配置 ──────────────────────────────────────────────────────
BASE_MASK = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/03-分割Mask/05-RAP-FSAM3掩膜")
GT_DIR = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/03-分割Mask/01-gt_masks")
INPUT_DIR = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/02-fft_frames")
OUTPUT_ROOT = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十A6A7验证产物")

# ── 样本与 GT 帧 ─────────────────────────────────────────────────
SAMPLES = ["CaoMei1", "ChangShouHua2", "DouBanLv1", "XianKeLai1"]
GT_FRAMES = {
    "CaoMei1": ["0000", "0025", "0050", "0075", "0100"],
    "ChangShouHua2": ["0000", "0025", "0050", "0075", "0100"],
    "DouBanLv1": ["0000", "0025", "0050", "0075", "0100"],
    "XianKeLai1": ["0000"],
}

# ── 分类表（来自实验三_P2去盆vsP6带盆_四样本16GT对比.csv + 逐帧表） ──
# (category, note)
CLASSIFICATION = {
    "CaoMei1_0000": ("gt_mixed", "F1=0.928，GT去盆/带盆混标，需人工核对"),
    "CaoMei1_0025": ("success", "F1=0.955，去盆干净"),
    "CaoMei1_0050": ("success", "F1=0.969，去盆干净"),
    "CaoMei1_0075": ("gt_mixed", "F1=0.951，掩膜含盆但GT去盆→F1虚高，需核对"),
    "CaoMei1_0100": ("pipeline_fail", "F1=0.229，SAM3候选失败(只割到盆丢植株)"),
    "ChangShouHua2_0000": ("success", "F1=0.821，去盆基本干净"),
    "ChangShouHua2_0025": ("success", "F1=0.978，去盆干净"),
    "ChangShouHua2_0050": ("success", "F1=0.984，去盆干净"),
    "ChangShouHua2_0075": ("gt_mixed", "F1=0.758，GT口径混标，需核对"),
    "ChangShouHua2_0100": ("pipeline_fail", "F1=0.436，GT含盆+深色叶漏分"),
    "DouBanLv1_0000": ("pipeline_fail", "F1=0.481，GT口径混标(标了盆)"),
    "DouBanLv1_0025": ("success", "F1=0.974，去盆干净"),
    "DouBanLv1_0050": ("success", "F1=0.986，去盆干净"),
    "DouBanLv1_0075": ("success", "F1=0.977，去盆干净"),
    "DouBanLv1_0100": ("success", "F1=0.970，去盆干净"),
    "XianKeLai1_0000": ("success", "F1=0.971，A6A7SPNP提升最大帧"),
}

# ── 工具函数 ──────────────────────────────────────────────────────
def load_gt_mask(sample: str, frame: str) -> np.ndarray:
    """从 labelme JSON 栅格化 GT 二值掩膜（0/255）。"""
    json_path = GT_DIR / sample / f"{frame}.json"
    with open(json_path) as f:
        data = json.load(f)
    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data["shapes"]:
        if shape.get("label") != "1":
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        # 填充多边形（linestrip 视为闭合多边形）
        cv2_fill_polygon(mask, pts)
    return (mask > 0).astype(np.uint8) * 255


def cv2_fill_polygon(mask: np.ndarray, pts: np.ndarray):
    """用 cv2 填充多边形到 mask（原地修改）。"""
    import cv2
    cv2.fillPoly(mask, [pts], 255)


def load_mask_png(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"))
    return (img > 127).astype(np.uint8) * 255


def make_overlay(orig: np.ndarray, mask: np.ndarray, color=(0, 180, 80), alpha=0.35) -> np.ndarray:
    """生成半透明叠加图（与脚本 save_overlay 风格一致）。"""
    rgb = orig.astype(np.float32)
    m = (mask > 0)[..., None]
    overlay = rgb * (1.0 - alpha * m) + np.array(color, dtype=np.float32) * (alpha * m)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_gt_contour(overlay: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """在叠加图上绘制 GT 轮廓（绿色实线），便于人工核查。"""
    import cv2
    out = overlay.copy()
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), thickness=3)
    return out


def build_panel(orig: np.ndarray, p2_mask: np.ndarray, gt_mask: np.ndarray,
                p6_mask: np.ndarray, scale_w: int = 1920) -> np.ndarray:
    """生成三联图：P2叠加 | GT轮廓(绿) | P6叠加。"""
    h, w = orig.shape[:2]
    scale = scale_w / w
    new_h = int(h * scale)
    # resize
    orig_s = np.array(Image.fromarray(orig).resize((scale_w, new_h)))
    p2_s = np.array(Image.fromarray(p2_mask).resize((scale_w, new_h)))
    gt_s = np.array(Image.fromarray(gt_mask).resize((scale_w, new_h)))
    p6_s = np.array(Image.fromarray(p6_mask).resize((scale_w, new_h)))

    p2_ov = make_overlay(orig_s, p2_s)
    p6_ov = make_overlay(orig_s, p6_s)
    gt_ov = make_overlay(orig_s, p2_s)  # 底图用 P2 叠加
    gt_ov = draw_gt_contour(gt_ov, gt_s)

    panel = np.hstack([p2_ov, gt_ov, p6_ov])
    return panel


def add_label(img: np.ndarray, text: str, pos=(10, 10), color=(255, 255, 255)) -> np.ndarray:
    """在图像上添加文字标签。"""
    out = img.copy()
    draw = ImageDraw.Draw(Image.fromarray(out))
    try:
        font = ImageDraw.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageDraw.truetype(ImageDraw.getfont().path, 28) if hasattr(ImageDraw, 'getfont') else None
    draw.text(pos, text, fill=color, font=font)
    return np.array(out)


# ── 主流程 ────────────────────────────────────────────────────────
def main():
    # 创建输出目录
    out_success = OUTPUT_ROOT / "成功产物"
    out_fail = OUTPUT_ROOT / "失败产物"
    out_mixed = OUTPUT_ROOT / "GT混标待核"
    out_smoke = OUTPUT_ROOT / "冒烟证据图"
    for d in [out_success, out_fail, out_mixed, out_smoke]:
        d.mkdir(parents=True, exist_ok=True)

    summary = {"success": [], "gt_mixed": [], "pipeline_fail": []}

    for sample in SAMPLES:
        frames = GT_FRAMES[sample]
        for frame in frames:
            key = f"{sample}_{frame}"
            if key not in CLASSIFICATION:
                continue
            category, note = CLASSIFICATION[key]

            # 路径
            p2_dir = BASE_MASK / f"E1_{sample}_A6A7_证据引导SPNP_101帧"
            p6_dir = BASE_MASK / f"E1_{sample}_P6带盆整株_101帧"
            orig_path = INPUT_DIR / sample / f"{frame}.jpg"

            if not (p2_dir / "最终掩膜" / f"mask_{frame}.png").exists():
                print(f"[跳过] {key}: P2 最终掩膜缺失")
                continue
            if not (p6_dir / "最终掩膜" / f"mask_{frame}.png").exists():
                print(f"[跳过] {key}: P6 最终掩膜缺失")
                continue
            if not orig_path.exists():
                print(f"[跳过] {key}: 原图缺失 {orig_path}")
                continue

            # 加载
            orig = np.array(Image.open(orig_path).convert("RGB"))
            p2_mask = load_mask_png(p2_dir / "最终掩膜" / f"mask_{frame}.png")
            p6_mask = load_mask_png(p6_dir / "最终掩膜" / f"mask_{frame}.png")
            gt_mask = load_gt_mask(sample, frame)

            # 生成三联图
            panel = build_panel(orig, p2_mask, gt_mask, p6_mask)
            panel = add_label(panel, f"{sample} {frame}  [{category}]", pos=(10, 10))

            # 输出文件名
            fname = f"{sample}_{frame}_P2|GT|P6.png"

            if category == "success":
                dst = out_success / fname
            elif category == "gt_mixed":
                dst = out_mixed / fname
            else:  # pipeline_fail
                dst = out_fail / fname

            Image.fromarray(panel).save(dst)
            summary[category].append((key, note, str(dst)))

            print(f"[完成] {key:25s} -> {category:14s} {dst.name}")

    # ── 固化 /tmp 冒烟证据图 ──────────────────────────────────────
    tmp_files = {
        "final_0000_3panel.png": "A6粘连切割_三联图_0000(GT帧,基线F1=0.9742→0.9787)",
        "final_0005_3panel.png": "A6粘连切割_三联图_0005(邻株切割)",
        "final_0006_3panel.png": "A6粘连切割_三联图_0006(邻株切割)",
        "final_0007_3panel.png": "A6粘连切割_三联图_0007(邻株切割)",
        "final_0008_3panel.png": "A6粘连切割_三联图_0008(邻株切割)",
        "final_0009_3panel.png": "A6粘连切割_三联图_0009(邻株切割)",
        "a7_0000_4panel.png": "A6+A7记忆传播_四联图_0000(A1s|共识|记忆|最终)",
        "a7_0005_4panel.png": "A6+A7记忆传播_四联图_0005(粘连帧最干净)",
        "a7_0006_4panel.png": "A6+A7记忆传播_四联图_0006",
        "a7_0007_4panel.png": "A6+A7记忆传播_四联图_0007",
        "a7_0008_4panel.png": "A6+A7记忆传播_四联图_0008",
        "a7_0009_4panel.png": "A6+A7记忆传播_四联图_0009",
        "evspnp_0006_3panel.png": "证据引导SPNP_三联图_0006(共识|SPNP细化|最终)",
        "evspnp_0007_3panel.png": "证据引导SPNP_三联图_0007(花盆剔除+邻株消失)",
        "evspnp_0008_3panel.png": "证据引导SPNP_三联图_0008",
        "evspnp_0009_3panel.png": "证据引导SPNP_三联图_0009",
        "bridge_0005_3panel.png": "粘连桥切割诊断_三联图_0005",
        "bridge_0006_3panel.png": "粘连桥切割诊断_三联图_0006",
        "bridge_0007_3panel.png": "粘连桥切割诊断_三联图_0007",
        "bridge_0009_3panel.png": "粘连桥切割诊断_三联图_0009",
    }
    for fname, desc in tmp_files.items():
        src = Path("/tmp") / fname
        if src.exists():
            dst = out_smoke / fname
            shutil.copy2(src, dst)
            print(f"[固化] {fname:30s} -> 冒烟证据图/{fname}")
        else:
            print(f"[缺失] /tmp/{fname}")

    # ── 写 README ─────────────────────────────────────────────────
    readme = build_readme(summary)
    (OUTPUT_ROOT / "阶段十A6A7产物整理README.md").write_text(readme, encoding="utf-8")
    print("\n[完成] README 已写入:", OUTPUT_ROOT / "阶段十A6A7产物整理README.md")


def build_readme(summary: dict) -> str:
    lines = []
    lines.append("# 阶段十 A6/A7 + 证据引导 SPNP 四样本产物整理\n")
    lines.append("整理日期：2026-08-27\n")
    lines.append("## 目录结构\n")
    lines.append("```")
    lines.append("阶段十A6A7验证产物/")
    lines.append("├── 成功产物/          # F1>=0.9 且 GT 口径干净，可直接用于论文")
    lines.append("├── GT混标待核/        # F1 高但 GT 标注口径需人工核对")
    lines.append("├── 失败产物/          # F1<0.6，SAM3 候选级失败（管道无法纠正）")
    lines.append("├── 冒烟证据图/        # KongQueZhuYu 10帧冒烟 A6/A7/证据引导SPNP 三联/四联图")
    lines.append("└── 阶段十A6A7产物整理README.md")
    lines.append("```\n")

    lines.append("## 成功产物（%d 帧）\n" % len(summary["success"]))
    lines.append("| 样本_帧 | 备注 | 图片 |")
    lines.append("| --- | --- | --- |")
    for key, note, path in summary["success"]:
        lines.append(f"| {key} | {note} | `成功产物/{Path(path).name}` |")
    lines.append("")

    lines.append("## GT 混标待核（%d 帧）\n" % len(summary["gt_mixed"]))
    lines.append("> 这些帧 F1 数值不低，但 GT 标注口径（去盆 vs 带盆）存在混标，需人工确认后再决定是否纳入主表。\n")
    lines.append("| 样本_帧 | 备注 | 图片 |")
    lines.append("| --- | --- | --- |")
    for key, note, path in summary["gt_mixed"]:
        lines.append(f"| {key} | {note} | `GT混标待核/{Path(path).name}` |")
    lines.append("")

    lines.append("## 失败产物（%d 帧）\n" % len(summary["pipeline_fail"]))
    lines.append("> 这些帧为 SAM3 候选级失败（前端分割即失败，A6/A7/SPNP 无法纠正），属模型能力边界，非管道 bug。\n")
    lines.append("| 样本_帧 | 备注 | 图片 |")
    lines.append("| --- | --- | --- |")
    for key, note, path in summary["pipeline_fail"]:
        lines.append(f"| {key} | {note} | `失败产物/{Path(path).name}` |")
    lines.append("")

    lines.append("## 冒烟证据图（KongQueZhuYu 10 帧）\n")
    lines.append("- `final_*_3panel.png`：A1s选中 | A6共识投票 | 最终掩膜（粘连切割验证）")
    lines.append("- `a7_*_4panel.png`：A1s选中 | A6共识 | A7记忆传播 | 最终掩膜")
    lines.append("- `evspnp_*_3panel.png`：A6共识 | 证据引导SPNP细化 | 最终掩膜（花盆剔除+邻株消失）")
    lines.append("- `bridge_*_3panel.png`：粘连桥切割诊断三联图")
    lines.append("")
    lines.append("## 读图约定\n")
    lines.append("- 三联图（P2|GT|P6）：左=P2去盆最终掩膜叠加，中=P2叠加+GT绿色轮廓，右=P6带盆最终掩膜叠加")
    lines.append("- 四联图（A1s|共识|记忆|最终）：从左到右即管线顺序，红色半透明为掩膜覆盖区")
    lines.append("- 真正用于论文和 3D 重建的是最右图（或三联图右图）")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
