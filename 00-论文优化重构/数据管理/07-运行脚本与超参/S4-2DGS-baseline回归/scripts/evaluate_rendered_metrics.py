#!/usr/bin/env python3
"""Evaluate rendered 2DGS images without importing the broken local lpipsPyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm


def psnr(render: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = ((render - gt) ** 2).view(render.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    values = [torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / float(2 * sigma**2))) for x in range(window_size)]
    gauss = torch.stack(values)
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    one_d = gaussian(window_size, 1.5).unsqueeze(1)
    two_d = one_d.mm(one_d.t()).float().unsqueeze(0).unsqueeze(0)
    return two_d.expand(channel, 1, window_size, window_size).contiguous().to(device=device, dtype=dtype)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    import torch.nn.functional as F

    channel = img1.size(-3)
    window = create_window(window_size, channel, img1.device, img1.dtype)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return tf.to_tensor(image).unsqueeze(0).to(device)[:, :3, :, :]


def evaluate_method(method_dir: Path, device: torch.device, use_lpips: bool) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    image_names = sorted(p.name for p in renders_dir.iterdir() if p.is_file())
    if not image_names:
        raise FileNotFoundError(f"No rendered images found in {renders_dir}")

    lpips_model = None
    if use_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    per_view: dict[str, dict[str, float]] = {}
    ssim_values: list[float] = []
    psnr_values: list[float] = []
    lpips_values: list[float] = []

    for name in tqdm(image_names, desc=f"metrics {method_dir.name}"):
        render = load_rgb(renders_dir / name, device)
        gt = load_rgb(gt_dir / name, device)
        ssim_value = float(ssim(render, gt).detach().cpu())
        psnr_value = float(psnr(render, gt).mean().detach().cpu())
        row = {"SSIM": ssim_value, "PSNR": psnr_value}
        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)
        if lpips_model is not None:
            # lpips expects [-1, 1].
            lpips_value = float(lpips_model(render * 2.0 - 1.0, gt * 2.0 - 1.0).mean().detach().cpu())
            row["LPIPS"] = lpips_value
            lpips_values.append(lpips_value)
        per_view[name] = row

    summary = {
        "SSIM": sum(ssim_values) / len(ssim_values),
        "PSNR": sum(psnr_values) / len(psnr_values),
    }
    if lpips_values:
        summary["LPIPS"] = sum(lpips_values) / len(lpips_values)
    return summary, per_view


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_lpips", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_dir = args.model_path / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    full: dict[str, dict[str, float]] = {}
    per_view: dict[str, dict[str, dict[str, float]]] = {}
    for method_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        summary, method_per_view = evaluate_method(method_dir, device, use_lpips=not args.no_lpips)
        full[method_dir.name] = summary
        per_view[method_dir.name] = method_per_view
        print(method_dir.name, json.dumps(summary, ensure_ascii=False))

    (args.model_path / "results.json").write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.model_path / "per_view.json").write_text(json.dumps(per_view, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
