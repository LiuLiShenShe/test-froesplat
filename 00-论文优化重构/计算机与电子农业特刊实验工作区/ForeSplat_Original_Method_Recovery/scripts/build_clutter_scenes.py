#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build controlled "cluttered flowerbed" scenes for the foreground-selection test.

Motivation
----------
The dev set is composed of *single* potted plants on a clean bench.  In that
setting the SG-MPS gate has nothing to suppress (no neighbour plants, no table
clutter), and an oracle sweep shows the whole candidate pool's best answer is
the single-prompt P2 mask in 16/16 frames -- so the gate can only lose, never
win.  The scenario the method is actually meant for is a *garden with many
plants where only the front-most one is the target*.

This script composites REAL plant crops (source RGB + the existing potted-plant
GT mask, so the cut-out is exact) into a multi-plant scene with a known depth
order.  The foreground plant's mask is the target GT *by construction*, which
makes the gate's neighbour/background/centre priors a measurable quantity
instead of a visual impression.

Layout (canvas 2160x3840, matching the source frames):
    back-left  plant  scale 0.44, dimmed + blurred   (far)
    back-right plant  scale 0.44, dimmed + blurred   (far)
    mid        plant  scale 0.56, centre, slight blur
    FRONT      plant  scale 0.78, centre-bottom, sharp  <-- target GT

Everything is deterministic given the seed, so the scene can be regenerated.

Usage:
    /home/test/.venvs/sam3/bin/python scripts/build_clutter_scenes.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
GT_SPLIT = (ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/"
            "S20-RAP-FSAM3掩膜生成与验证/GT口径拆分审计/gt_masks_split_v2")
OUT = Path(os.environ.get("CLUTTER_ROOT", str(RECOVERY / "runs/clutter_scene")))
INPUTS = OUT / "inputs"
GT_OUT = OUT / "gt"

SCENES = {
    "CaoMei1": ["0000", "0025", "0050", "0075", "0100"],
    "ChangShouHua2": ["0000", "0025", "0050", "0075", "0100"],
    "DouBanLv1": ["0000", "0025", "0050", "0075", "0100"],
    "XianKeLai1": ["0000"],
}
CANVAS = (2160, 3840)   # w, h (matches dev_inputs)


def load_rgba(scene: str, stem: str) -> Image.Image | None:
    img_path = RECOVERY / "dev_inputs" / scene / f"{stem}.jpg"
    gt_path = GT_SPLIT / scene / f"mask_potted_plant_{stem}.png"
    if not (img_path.exists() and gt_path.exists()):
        return None
    rgb = Image.open(img_path).convert("RGB")
    gt = Image.open(gt_path).convert("L")
    if gt.size != rgb.size:
        gt = gt.resize(rgb.size, Image.NEAREST)
    mask = np.array(gt) > 127
    if mask.sum() < 0.02 * mask.size:
        return None
    rgba = np.dstack([np.array(rgb), (mask * 255).astype(np.uint8)])
    ys, xs = np.where(mask)
    crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return Image.fromarray(crop, "RGBA")


def tint(img: Image.Image, brightness: float, blur: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    arr[..., :3] = np.clip(arr[..., :3] * brightness, 0, 255)
    out = Image.fromarray(arr.astype(np.uint8), "RGBA")
    if blur > 0:
        # Blur RGB only so the alpha edge stays hard for compositing.
        rgb = out.convert("RGB").filter(ImageFilter.GaussianBlur(blur))
        out = Image.merge("RGBA", (*rgb.split(), out.split()[3]))
    return out


def paste_scaled(canvas: Image.Image, sprite: Image.Image, scale: float,
                 center_x: float, bottom_y: float
                 ) -> tuple[Image.Image, tuple[int, int, int, int], Image.Image]:
    """Scale the sprite to `scale * canvas height` and paste bottom-anchored.

    Returns (composed canvas, paste box, resized sprite) -- the resized sprite is
    needed by the caller to derive the target mask at the final scale.
    """
    cw, ch = canvas.size
    target_h = max(8, int(ch * scale))
    ratio = target_h / sprite.height
    target_w = max(8, int(sprite.width * ratio))
    sprite = sprite.resize((target_w, target_h), Image.LANCZOS)
    left = int(center_x * cw - target_w / 2)
    top = int(bottom_y * ch - target_h)
    left = max(0, min(left, cw - target_w))
    top = max(0, min(top, ch - target_h))
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(sprite, (left, top), sprite)
    composed = Image.alpha_composite(canvas.convert("RGBA"), layer)
    return composed, (left, top, target_w, target_h), sprite


def garden_background(seed: int) -> Image.Image:
    """A soft green-brown vertical gradient, so the canvas is not pure black."""
    rng = np.random.default_rng(seed)
    w, h = CANVAS
    top = np.array([150, 175, 140], dtype=np.float32)   # pale sky-green
    bottom = np.array([70, 95, 55], dtype=np.float32)   # soil green-brown
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    grad = (top[None, :] * (1 - t) + bottom[None, :] * t)[:, None, :]
    img = np.repeat(grad, w, axis=1)
    img += rng.normal(0, 3.0, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def build_one(name: str, fg: tuple[str, str], mids: list[tuple[str, str]],
              backs: list[tuple[str, str]], seed: int,
              hard: bool = False) -> dict | None:
    rng = random.Random(seed)
    fg_img = load_rgba(*fg)
    if fg_img is None:
        return None
    canvas = garden_background(seed)
    layers: list[tuple[str, np.ndarray]] = []
    order = [(s, "back") for s in backs] + [(s, "mid") for s in mids] + [(fg, "front")]
    for (scene, stem), role in order:
        sprite = load_rgba(scene, stem)
        if sprite is None:
            if role == "front":
                return None
            continue
        if role == "back":
            if hard:
                # Boundary probe: the *distractor* is bigger than the target and
                # sits higher in the frame, so "largest instance" no longer
                # equals "front-most target".
                sprite = tint(sprite, brightness=0.78, blur=4.0)
                canvas, box, placed = paste_scaled(canvas, sprite, scale=0.95,
                                                   center_x=rng.choice([0.18, 0.82]),
                                                   bottom_y=0.96)
            else:
                sprite = tint(sprite, brightness=0.70, blur=5.0)
                canvas, box, placed = paste_scaled(canvas, sprite, scale=0.44,
                                                   center_x=rng.choice([0.10, 0.90]),
                                                   bottom_y=rng.uniform(0.60, 0.66))
        elif role == "mid":
            sprite = tint(sprite, brightness=0.84, blur=2.5)
            canvas, box, placed = paste_scaled(
                canvas, sprite, scale=0.70 if hard else 0.56,
                center_x=rng.choice([0.25, 0.75]), bottom_y=0.80)
        else:
            canvas, box, placed = paste_scaled(
                canvas, sprite, scale=0.62 if hard else 0.78,
                center_x=0.30 if hard else 0.50, bottom_y=0.99)
        left, top, tw, th = box
        alpha = np.array(placed.split()[3]) > 127
        m = np.zeros((CANVAS[1], CANVAS[0]), dtype=bool)
        region = m[top:top + th, left:left + tw]
        region[alpha[:region.shape[0], :region.shape[1]]] = True
        layers.append((role, m))
    if not layers or layers[-1][0] != "front":
        return None
    fg_mask = (layers[-1][1] * 255).astype(np.uint8)
    # A neighbour's *visible* pixels are what the prediction can wrongfully
    # claim; the part hidden behind a nearer plant is not a false positive.
    neigh: list[tuple[str, np.ndarray]] = []
    for i, (role, m) in enumerate(layers[:-1]):
        occluded = np.zeros_like(m)
        for _, nearer in layers[i + 1:]:
            occluded |= nearer
        visible = m & ~occluded
        if visible.any():
            neigh.append((f"{role}_{i}", (visible * 255).astype(np.uint8)))
    INPUTS.mkdir(parents=True, exist_ok=True)
    GT_OUT.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(INPUTS / f"{name}.jpg", quality=95)
    Image.fromarray(fg_mask, "L").save(GT_OUT / f"mask_{name}.png")
    # Neighbour masks let the probe quantify leakage into the *non-target*
    # plants, which is the failure mode the gate is supposed to prevent.
    for idx, (tag, nm) in enumerate(neigh):
        Image.fromarray(nm, "L").save(GT_OUT / f"neigh_{name}_{idx}_{tag}.png")
    return {"name": name, "fg_source": f"{fg[0]}/{fg[1]}",
            "fg_ratio": float((fg_mask > 127).mean()),
            "neighbours": [t for t, _ in neigh]}


def main() -> int:
    hard = "--hard" in sys.argv
    if hard:
        specs = [
            ("clutter_hard_01", ("CaoMei1", "0025"), [("DouBanLv1", "0050")],
             [("ChangShouHua2", "0100"), ("XianKeLai1", "0000")], 55),
            ("clutter_hard_02", ("DouBanLv1", "0075"), [("XianKeLai1", "0000")],
             [("CaoMei1", "0100"), ("ChangShouHua2", "0000")], 66),
            ("clutter_hard_03", ("ChangShouHua2", "0050"), [("CaoMei1", "0075")],
             [("DouBanLv1", "0025"), ("XianKeLai1", "0000")], 77),
        ]
    else:
        specs = [
            ("clutter_01", ("CaoMei1", "0000"), [("DouBanLv1", "0025")],
             [("ChangShouHua2", "0050"), ("XianKeLai1", "0000")], 11),
            ("clutter_02", ("ChangShouHua2", "0025"), [("CaoMei1", "0050")],
             [("DouBanLv1", "0000"), ("CaoMei1", "0100")], 22),
            ("clutter_03", ("DouBanLv1", "0050"), [("XianKeLai1", "0000")],
             [("CaoMei1", "0025"), ("ChangShouHua2", "0075")], 33),
            ("clutter_04", ("XianKeLai1", "0000"), [("ChangShouHua2", "0100")],
             [("DouBanLv1", "0075"), ("CaoMei1", "0075")], 44),
        ]
    built = []
    for name, fg, mids, backs, seed in specs:
        info = build_one(name, fg, mids, backs, seed, hard=hard)
        if info:
            built.append(info)
            print(f"[built] {name}: fg={info['fg_source']} fg_ratio={info['fg_ratio']:.3f}")
        else:
            print(f"[skip] {name}: missing source")
    (OUT / "scene_manifest.json").write_text(
        json.dumps({"canvas": CANVAS, "scenes": built,
                    "note": "foreground plant mask is the target GT by construction"},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(built)} scenes -> {INPUTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
