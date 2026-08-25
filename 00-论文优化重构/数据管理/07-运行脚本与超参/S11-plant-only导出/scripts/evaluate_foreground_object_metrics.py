#!/usr/bin/env python3
"""Evaluate rendered images on the mask foreground object region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm


def load_eval_image_map(source_path: Path, images_dir: str = "images", llffhold: int = 8) -> dict[str, str]:
    image_root = source_path / images_dir
    image_names = sorted(
        p.name
        for p in image_root.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    test_names = [name for idx, name in enumerate(image_names) if idx % llffhold == 0]
    return {f"{idx:05d}.png": Path(name).stem for idx, name in enumerate(test_names)}


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    values = [
        torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / float(2 * sigma**2)))
        for x in range(window_size)
    ]
    gauss = torch.stack(values)
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    one_d = gaussian(window_size, 1.5).unsqueeze(1)
    two_d = one_d.mm(one_d.t()).float().unsqueeze(0).unsqueeze(0)
    return two_d.expand(channel, 1, window_size, window_size).contiguous().to(device=device, dtype=dtype)


def masked_psnr(render: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0) * render.shape[1]
    mse = (((render - gt) ** 2) * mask).sum() / denom
    return 20 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(1.0e-12)))


def masked_ssim(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    import torch.nn.functional as F

    channel = img1.size(1)
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
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    weight = mask.expand_as(ssim_map)
    return (ssim_map * weight).sum() / weight.sum().clamp_min(1.0)


def leakage_metrics(render: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    outside = 1.0 - mask
    inside = mask
    render_gray = render.mean(dim=1, keepdim=True)
    outside_denom = outside.sum().clamp_min(1.0)
    inside_denom = inside.sum().clamp_min(1.0)
    outside_energy = (render_gray * outside).sum() / outside_denom
    inside_energy = (render_gray * inside).sum() / inside_denom
    outside_nonblack = ((render_gray > 0.03).float() * outside).sum() / outside_denom
    leakage_ratio = outside_energy / inside_energy.clamp_min(1.0e-6)
    return {
        "outside_energy": float(outside_energy.detach().cpu()),
        "inside_energy": float(inside_energy.detach().cpu()),
        "outside_nonblack_ratio": float(outside_nonblack.detach().cpu()),
        "leakage_energy_ratio": float(leakage_ratio.detach().cpu()),
    }


def crop_to_mask_bbox(
    render: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    padding: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_2d = mask[0, 0] > 0.5
    coords = torch.nonzero(mask_2d, as_tuple=False)
    if coords.numel() == 0:
        return render, gt, mask
    y0 = max(int(coords[:, 0].min().item()) - padding, 0)
    y1 = min(int(coords[:, 0].max().item()) + padding + 1, render.shape[-2])
    x0 = max(int(coords[:, 1].min().item()) - padding, 0)
    x1 = min(int(coords[:, 1].max().item()) + padding + 1, render.shape[-1])
    return render[:, :, y0:y1, x0:x1], gt[:, :, y0:y1, x0:x1], mask[:, :, y0:y1, x0:x1]


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    return tf.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def load_mask(path: Path, size: tuple[int, int], threshold: int, device: torch.device) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return (tf.to_tensor(mask).unsqueeze(0).to(device) >= (threshold / 255.0)).float()


def mask_path_for(mask_dir: Path, pattern: str, image_name: str) -> Path:
    stem = Path(image_name).stem
    path = mask_dir / pattern.format(stem=stem, image_name=image_name)
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing mask for {image_name}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--llffhold", default=8, type=int)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--mask-pattern", default="mask_{stem}.png")
    parser.add_argument("--mask-threshold", default=127, type=int)
    parser.add_argument("--method", default="ours_30000")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    method_dir = args.model_path / "test" / args.method
    renders_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    image_names = sorted(p.name for p in renders_dir.iterdir() if p.is_file())
    image_map = load_eval_image_map(args.source_path, args.images_dir, args.llffhold)

    lpips_model = None
    if not args.no_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    per_view: dict[str, dict[str, float]] = {}
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    lpips_crop_values: list[float] = []
    ssim_crop_values: list[float] = []
    mask_ratios: list[float] = []
    outside_energy_values: list[float] = []
    inside_energy_values: list[float] = []
    outside_nonblack_values: list[float] = []
    leakage_ratio_values: list[float] = []

    for name in tqdm(image_names, desc=f"foreground metrics {args.method}"):
        render = load_rgb(renders_dir / name, device)
        gt = load_rgb(gt_dir / name, device)
        mask = load_mask(
            mask_path_for(args.mask_dir, args.mask_pattern, image_map.get(name, Path(name).stem)),
            (render.shape[-1], render.shape[-2]),
            args.mask_threshold,
            device,
        )
        psnr_value = float(masked_psnr(render, gt, mask).detach().cpu())
        ssim_value = float(masked_ssim(render, gt, mask).detach().cpu())
        row = {
            "PSNR_fg": psnr_value,
            "SSIM_fg": ssim_value,
            "mask_ratio": float(mask.mean().detach().cpu()),
        }
        row.update(leakage_metrics(render, mask))
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        mask_ratios.append(row["mask_ratio"])
        outside_energy_values.append(row["outside_energy"])
        inside_energy_values.append(row["inside_energy"])
        outside_nonblack_values.append(row["outside_nonblack_ratio"])
        leakage_ratio_values.append(row["leakage_energy_ratio"])
        if lpips_model is not None:
            render_fg = render * mask
            gt_fg = gt * mask
            lpips_value = float(
                lpips_model(render_fg * 2.0 - 1.0, gt_fg * 2.0 - 1.0).mean().detach().cpu()
            )
            row["LPIPS_fg_black_bg"] = lpips_value
            lpips_values.append(lpips_value)
            render_crop, gt_crop, mask_crop = crop_to_mask_bbox(render, gt, mask)
            render_crop_fg = render_crop * mask_crop
            gt_crop_fg = gt_crop * mask_crop
            lpips_crop = float(
                lpips_model(render_crop_fg * 2.0 - 1.0, gt_crop_fg * 2.0 - 1.0)
                .mean()
                .detach()
                .cpu()
            )
            ssim_crop = float(masked_ssim(render_crop, gt_crop, mask_crop).detach().cpu())
            row["LPIPS_fg_crop"] = lpips_crop
            row["SSIM_fg_crop"] = ssim_crop
            lpips_crop_values.append(lpips_crop)
            ssim_crop_values.append(ssim_crop)
        per_view[name] = row

    summary = {
        "PSNR_fg": sum(psnr_values) / len(psnr_values),
        "SSIM_fg": sum(ssim_values) / len(ssim_values),
        "mask_ratio_mean": sum(mask_ratios) / len(mask_ratios),
        "outside_energy_mean": sum(outside_energy_values) / len(outside_energy_values),
        "inside_energy_mean": sum(inside_energy_values) / len(inside_energy_values),
        "outside_nonblack_ratio_mean": sum(outside_nonblack_values) / len(outside_nonblack_values),
        "leakage_energy_ratio_mean": sum(leakage_ratio_values) / len(leakage_ratio_values),
        "num_images": len(image_names),
        "eval_scope": "mask_foreground_object",
    }
    if lpips_values:
        summary["LPIPS_fg_black_bg"] = sum(lpips_values) / len(lpips_values)
    if lpips_crop_values:
        summary["LPIPS_fg_crop"] = sum(lpips_crop_values) / len(lpips_crop_values)
        summary["SSIM_fg_crop"] = sum(ssim_crop_values) / len(ssim_crop_values)

    out = {
        args.method: summary,
        "per_view": per_view,
    }
    out_path = args.model_path / "foreground_object_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
