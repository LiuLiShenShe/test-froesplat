#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train few-shot supervised segmentation references for experiment one.

By default, the script uses the fixed S23 six-frame GT subset and writes
predictions into the same ``method_masks`` directory used by the external
baseline evaluator. ``--bench-root`` can point it at the four-sample
representative benchmark without changing the training protocol.

Evaluation protocol:
- Two sequence-level folds.
- Train on all GT frames from one sequence and predict the held-out sequence.
- No ImageNet/backbone pretraining is used, because the current torchvision
  install is not importable in this environment.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
PAPER_ROOT = WORKSPACE_DIR.parent
DATA_ROOT = PAPER_ROOT / "数据管理"
BENCH_ROOT = DATA_ROOT / "05-评测结果/S23_Experiment1_VFM_Benchmark"
METHOD_MASK_ROOT = BENCH_ROOT / "method_masks"
RUN_ROOT = DATA_ROOT / "05-评测结果/S23_Experiment1_FewShot_Supervised"

METHODS = {
    "unet": {
        "method_dir": "UNet_fewshot_seqcv",
        "display": "U-Net",
        "description": "random-initialized compact U-Net",
    },
    "deeplab": {
        "method_dir": "DeepLabV3PlusLite_fewshot_seqcv",
        "display": "DeepLabv3+ lite",
        "description": "random-initialized ASPP encoder-decoder",
    },
}


@dataclass(frozen=True)
class FrameRecord:
    sample: str
    frame: str
    image: Path
    gt_mask: Path

    @property
    def stem(self) -> str:
        return f"{self.sample}_{self.frame}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest() -> list[FrameRecord]:
    rows = read_csv(BENCH_ROOT / "manifest.csv")
    return [
        FrameRecord(
            sample=row["sample"],
            frame=row["frame"],
            image=Path(row["image"]),
            gt_mask=Path(row["gt_mask"]),
        )
        for row in rows
    ]


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img > 127


def resize_image(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    out = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return out > 0


def normalize_image(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return (arr - mean) / std


def augment_pair(rgb: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    out_rgb = rgb.copy()
    out_mask = mask.copy()

    if rng.random() < 0.5:
        out_rgb = np.ascontiguousarray(out_rgb[:, ::-1])
        out_mask = np.ascontiguousarray(out_mask[:, ::-1])

    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.90, 1.10)
    tx = rng.uniform(-0.04, 0.04) * w
    ty = rng.uniform(-0.04, 0.04) * h
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    mat[:, 2] += (tx, ty)
    out_rgb = cv2.warpAffine(
        out_rgb,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    out_mask = cv2.warpAffine(
        out_mask.astype(np.uint8),
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    contrast = rng.uniform(0.82, 1.18)
    brightness = rng.uniform(-0.08, 0.08)
    out_rgb = np.clip(out_rgb.astype(np.float32) / 255.0 * contrast + brightness, 0.0, 1.0)
    if rng.random() < 0.3:
        gamma = rng.uniform(0.85, 1.20)
        out_rgb = np.power(out_rgb, gamma)
    return (out_rgb * 255.0).astype(np.uint8), out_mask


class FewShotDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[FrameRecord],
        *,
        width: int,
        height: int,
        augment: bool,
        augment_multiplier: int,
        seed: int,
    ) -> None:
        self.records = records
        self.augment = augment
        self.augment_multiplier = max(1, augment_multiplier)
        self.rng = random.Random(seed)
        self.cache: list[tuple[np.ndarray, np.ndarray]] = []
        for record in records:
            rgb = resize_image(read_rgb(record.image), width, height)
            mask = resize_mask(read_mask(record.gt_mask), width, height)
            self.cache.append((rgb, mask))

    def __len__(self) -> int:
        return len(self.records) * self.augment_multiplier

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, mask = self.cache[index % len(self.cache)]
        if self.augment:
            rgb, mask = augment_pair(rgb, mask, self.rng)
        x = torch.from_numpy(normalize_image(rgb).transpose(2, 0, 1)).float()
        y = torch.from_numpy(mask.astype(np.float32)[None, :, :]).float()
        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.enc1 = DoubleConv(3, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 12)
        self.up4 = nn.ConvTranspose2d(base * 12, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
                ConvBNReLU(in_ch, out_ch, dilation=2),
                ConvBNReLU(in_ch, out_ch, dilation=4),
                ConvBNReLU(in_ch, out_ch, dilation=8),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(self.branches), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabV3PlusLite(nn.Module):
    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, base, stride=2),
            ConvBNReLU(base, base),
        )
        self.low = ConvBNReLU(base, base * 2, stride=2)
        self.enc2 = ConvBNReLU(base * 2, base * 4, stride=2)
        self.enc3 = ConvBNReLU(base * 4, base * 6, dilation=2)
        self.enc4 = ConvBNReLU(base * 6, base * 8, dilation=4)
        self.aspp = ASPP(base * 8, base * 4)
        self.low_proj = nn.Sequential(
            nn.Conv2d(base * 2, base, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            DoubleConv(base * 5, base * 3),
            nn.Conv2d(base * 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.stem(x)
        low = self.low(x)
        x = self.enc2(low)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.aspp(x)
        x = torch.nn.functional.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, self.low_proj(low)], dim=1))
        return torch.nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


def build_model(model_key: str) -> nn.Module:
    if model_key == "unet":
        return UNetSmall()
    if model_key == "deeplab":
        return DeepLabV3PlusLite()
    raise ValueError(f"Unsupported model: {model_key}")


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + 1.0
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return 1.0 - (num / den).mean()


def estimate_pos_weight(records: list[FrameRecord], width: int, height: int) -> float:
    pos = 0
    total = 0
    for record in records:
        mask = resize_mask(read_mask(record.gt_mask), width, height)
        pos += int(mask.sum())
        total += int(mask.size)
    neg = max(1, total - pos)
    pos = max(1, pos)
    return float(np.clip(neg / pos, 0.5, 8.0))


def train_one_fold(
    *,
    model_key: str,
    train_records: list[FrameRecord],
    test_records: list[FrameRecord],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, list[dict[str, object]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = build_model(model_key).to(device)
    pos_weight = torch.tensor(estimate_pos_weight(train_records, args.target_width, args.target_height), device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    dataset = FewShotDataset(
        train_records,
        width=args.target_width,
        height=args.target_height,
        augment=True,
        augment_multiplier=args.augment_multiplier,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    rows: list[dict[str, object]] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            rows.append(
                {
                    "model": METHODS[model_key]["display"],
                    "test_sample": ",".join(sorted({r.sample for r in test_records})),
                    "train_samples": ",".join(sorted({r.sample for r in train_records})),
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_sec": time.time() - start,
                }
            )
    return model, rows


def predict_record(
    model: nn.Module,
    record: FrameRecord,
    *,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict[str, object]:
    rgb_full = read_rgb(record.image)
    h_full, w_full = rgb_full.shape[:2]
    rgb = resize_image(rgb_full, args.target_width, args.target_height)
    x_np = normalize_image(rgb).transpose(2, 0, 1)[None, :, :, :]
    x = torch.from_numpy(x_np).float().to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        logits_flip = torch.flip(model(torch.flip(x, dims=[3])), dims=[3])
        prob = torch.sigmoid((logits + logits_flip) * 0.5)[0, 0].detach().cpu().numpy()
    prob_full = cv2.resize(prob, (w_full, h_full), interpolation=cv2.INTER_LINEAR)
    mask = prob_full >= args.threshold
    out_path = out_dir / f"mask_{record.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), (mask.astype(np.uint8) * 255))
    return {
        "sample": record.sample,
        "frame": record.frame,
        "image": str(record.image),
        "gt_mask": str(record.gt_mask),
        "pred_mask": str(out_path),
        "threshold": args.threshold,
        "foreground_ratio": float(mask.mean()),
        "prob_mean": float(prob_full.mean()),
    }


def run_model(model_key: str, args: argparse.Namespace, frames: list[FrameRecord], device: torch.device) -> None:
    method_dir = METHODS[model_key]["method_dir"]
    out_dir = METHOD_MASK_ROOT / method_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = RUN_ROOT / method_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    samples = sorted({record.sample for record in frames})
    train_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    for fold_idx, test_sample in enumerate(samples):
        train_records = [record for record in frames if record.sample != test_sample]
        test_records = [record for record in frames if record.sample == test_sample]
        if not train_records or not test_records:
            continue
        seed = args.seed + fold_idx * 97 + (0 if model_key == "unet" else 997)
        model, fold_train_rows = train_one_fold(
            model_key=model_key,
            train_records=train_records,
            test_records=test_records,
            args=args,
            device=device,
            seed=seed,
        )
        train_rows.extend(fold_train_rows)
        for record in test_records:
            row = predict_record(model, record, args=args, device=device, out_dir=out_dir)
            row.update(
                {
                    "model": METHODS[model_key]["display"],
                    "method_dir": method_dir,
                    "fold": fold_idx + 1,
                    "train_samples": ",".join(sorted({r.sample for r in train_records})),
                    "train_frames": len(train_records),
                    "protocol": "leave-one-sequence-out few-shot training on benchmark manifest",
                    "weights": "random initialization; no external pretraining",
                }
            )
            pred_rows.append(row)

    write_csv(
        log_dir / "training_log.csv",
        ["model", "test_sample", "train_samples", "epoch", "loss", "lr", "elapsed_sec"],
        train_rows,
    )
    write_csv(
        out_dir / "run_log.csv",
        [
            "model",
            "method_dir",
            "fold",
            "sample",
            "frame",
            "train_samples",
            "train_frames",
            "protocol",
            "weights",
            "image",
            "gt_mask",
            "pred_mask",
            "threshold",
            "foreground_ratio",
            "prob_mean",
        ],
        pred_rows,
    )
    write_csv(
        log_dir / "prediction_log.csv",
        [
            "model",
            "method_dir",
            "fold",
            "sample",
            "frame",
            "train_samples",
            "train_frames",
            "protocol",
            "weights",
            "image",
            "gt_mask",
            "pred_mask",
            "threshold",
            "foreground_ratio",
            "prob_mean",
        ],
        pred_rows,
    )


def main() -> int:
    global BENCH_ROOT, METHOD_MASK_ROOT, RUN_ROOT

    parser = argparse.ArgumentParser(description="Train few-shot supervised references for experiment one.")
    parser.add_argument("--bench-root", type=Path, default=BENCH_ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=sorted(METHODS), default=["unet", "deeplab"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--augment-multiplier", type=int, default=16)
    parser.add_argument("--target-width", type=int, default=384)
    parser.add_argument("--target-height", type=int, default=672)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    BENCH_ROOT = args.bench_root.resolve()
    METHOD_MASK_ROOT = BENCH_ROOT / "method_masks"
    RUN_ROOT = args.run_root.resolve() if args.run_root is not None else BENCH_ROOT / "fewshot_runs"

    frames = load_manifest()
    if not frames:
        raise RuntimeError(f"Benchmark manifest is empty or missing: {BENCH_ROOT / 'manifest.csv'}")
    device = torch.device(args.device)
    print(f"Using device={device}; frames={len(frames)}; models={','.join(args.models)}")
    for model_key in args.models:
        print(f"Training {METHODS[model_key]['display']} ({METHODS[model_key]['description']})")
        run_model(model_key, args, frames, device)
    print(f"Wrote supervised masks under {METHOD_MASK_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
