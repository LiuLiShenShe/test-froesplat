#!/usr/bin/env python3
"""Compute PSNR, SSIM, and LPIPS from rendered 2DGS test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def image_to_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(image.size[1], image.size[0], 3).permute(2, 0, 1).contiguous().float().div(255.0)
    return data.unsqueeze(0).to(device)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = ((pred - gt) ** 2).reshape(pred.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    window_1d = gaussian_window(window_size, 1.5, device, dtype).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channel = pred.size(-3)
    window = create_window(window_size, channel, pred.device, pred.dtype)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(gt, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def load_lpips(net: str, device: torch.device) -> torch.nn.Module:
    import lpips

    model = lpips.LPIPS(net=net, verbose=False).to(device)
    model.eval()
    return model


def compute(model_dir: Path, iteration: int, lpips_net: str, device_name: str) -> dict[str, object]:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    method = f"ours_{iteration}"
    method_dir = model_dir / "test" / method
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if not render_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing render/gt directories under {method_dir}")

    render_files = sorted(path for path in render_dir.iterdir() if path.is_file())
    if not render_files:
        raise FileNotFoundError(f"No rendered images found in {render_dir}")

    lpips_model = load_lpips(lpips_net, device)
    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    per_view = {"SSIM": {}, "PSNR": {}, "LPIPS": {}}

    with torch.no_grad():
        for index, render_path in enumerate(render_files, start=1):
            gt_path = gt_dir / render_path.name
            if not gt_path.is_file():
                raise FileNotFoundError(f"Missing GT image for {render_path.name}: {gt_path}")

            render_tensor = image_to_tensor(render_path, device)
            gt_tensor = image_to_tensor(gt_path, device)
            if render_tensor.shape != gt_tensor.shape:
                raise ValueError(f"Shape mismatch for {render_path.name}: {render_tensor.shape} vs {gt_tensor.shape}")

            ssim_value = ssim(render_tensor, gt_tensor).item()
            psnr_value = psnr(render_tensor, gt_tensor).item()
            lpips_value = lpips_model(render_tensor, gt_tensor, normalize=True).item()

            ssims.append(ssim_value)
            psnrs.append(psnr_value)
            lpipss.append(lpips_value)
            per_view["SSIM"][render_path.name] = ssim_value
            per_view["PSNR"][render_path.name] = psnr_value
            per_view["LPIPS"][render_path.name] = lpips_value

            if index == 1 or index == len(render_files) or index % 10 == 0:
                print(f"  metrics {index}/{len(render_files)} {render_path.name}")

    summary = {
        "SSIM": float(torch.tensor(ssims).mean().item()),
        "PSNR": float(torch.tensor(psnrs).mean().item()),
        "LPIPS": float(torch.tensor(lpipss).mean().item()),
    }
    (model_dir / "results.json").write_text(
        json.dumps({method: summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_dir / "per_view.json").write_text(
        json.dumps({method: per_view}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  SSIM : {summary['SSIM']:.7f}")
    print(f"  PSNR : {summary['PSNR']:.7f}")
    print(f"  LPIPS: {summary['LPIPS']:.7f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model-path", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    compute(args.model_path, args.iteration, args.lpips_net, args.device)


if __name__ == "__main__":
    main()
