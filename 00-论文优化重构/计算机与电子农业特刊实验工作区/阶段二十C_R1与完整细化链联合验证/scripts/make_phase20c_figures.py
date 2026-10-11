#!/usr/bin/env python3
"""Create traceable draft comparison figures from real Phase 20C masks."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十C_R1与完整细化链联合验证"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def panel(image: Image.Image, title: str, subtitle: str = "", size: tuple[int, int] = (280, 520)) -> Image.Image:
    canvas = Image.new("RGB", (size[0], size[1] + 64), "white")
    canvas.paste(fit(image, size), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, size[1], size[0], size[1] + 64), fill=(245, 245, 245))
    draw.text((8, size[1] + 6), title, fill="black")
    if subtitle:
        draw.text((8, size[1] + 30), subtitle, fill=(50, 50, 50))
    return canvas


def blocked(title: str, subtitle: str) -> Image.Image:
    image = Image.new("RGB", (280, 520), (238, 238, 238))
    draw = ImageDraw.Draw(image)
    draw.multiline_text((20, 230), f"{title}\n{subtitle}", fill=(100, 100, 100), align="center")
    return image


def open_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L").convert("RGB")


def save_grid(path: Path, panels: list[Image.Image], columns: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows_count = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * rows_count), "white")
    for index, image in enumerate(panels):
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    canvas.save(path)


def main() -> int:
    manifest = rows(WORK / "configs/PHASE20C_INPUT_MANIFEST.csv")
    manifest_by_stem = {row["stem"]: row for row in manifest}
    frame_metrics = rows(WORK / "metrics/frame_metrics.csv") if (WORK / "metrics/frame_metrics.csv").exists() else []
    metrics = {(row["dataset"], row["group"], row["stem"]): row for row in frame_metrics}
    out = WORK / "figures/generated"

    hard_stems = [
        "DouBanLv2_0039",
        "DouBanLv2_0040",
        "DouBanLv2_0041",
        "BaiZhang_0033",
        "BaiZhang_0034",
        "BaiZhang_0035",
        "BaiZhang_0036",
        "BaiZhang_0037",
    ]
    for stem in hard_stems:
        record = manifest_by_stem[stem]
        rgb = Image.open(ROOT / record["input_path"])
        gt = open_mask(ROOT / record["gt_path"])
        panels = [panel(rgb, "RGB", stem), panel(gt, "GT", "potted_clean")]
        for group in ("V0", "V1"):
            mask = WORK / "runs" / group / record["dataset"] / f"最终掩膜/mask_{stem}.png"
            row = metrics.get((record["dataset"], group, stem), {})
            panels.append(panel(open_mask(mask), group, f"F1={float(row.get('final_f1', 0)):.4f}"))
        for group in ("V2", "V3"):
            panels.append(panel(blocked(group, "GT10 COLMAP blocked"), group, "not run"))
        save_grid(out / f"GT10_{stem}_RGB_GT_V0_V1_V2_V3.png", panels)

    easy_stem = "CaoMei1_0000"
    record = manifest_by_stem[easy_stem]
    rgb = Image.open(ROOT / record["input_path"])
    gt = open_mask(ROOT / record["gt_path"])
    panels = [panel(rgb, "RGB", easy_stem), panel(gt, "GT", "potted_clean")]
    for group in ("V0", "V1", "V2", "V3"):
        mask = WORK / "runs" / group / record["dataset"] / f"最终掩膜/mask_{easy_stem}.png"
        row = metrics.get((record["dataset"], group, easy_stem), {})
        panels.append(panel(open_mask(mask), group, f"F1={float(row.get('final_f1', 0)):.4f}"))
    save_grid(out / f"Easy21_{easy_stem}_RGB_GT_V0_V1_V2_V3.png", panels)

    stage_stems = ["CaoMei1_0000", "DouBanLv1_0050"]
    for stem in stage_stems:
        record = manifest_by_stem[stem]
        rgb = Image.open(ROOT / record["input_path"])
        output = WORK / "runs/V3/Easy21"
        paths = {
            "Selected": output / f"选择后掩膜/mask_{stem}.png",
            "SPNP": output / f"正负提示细化掩膜/mask_{stem}.png",
            "Residual": output / f"残差修复掩膜/mask_{stem}.png",
            "GCFC": output / f"几何修正掩膜/mask_{stem}.png",
            "Final": output / f"最终掩膜/mask_{stem}.png",
        }
        panels = [panel(rgb, "RGB", stem)]
        for title, path in paths.items():
            panels.append(panel(open_mask(path), title, "real output"))
        save_grid(out / f"Easy21_{stem}_refinement_stages.png", panels, columns=6)
    print(f"figures={len(list(out.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
