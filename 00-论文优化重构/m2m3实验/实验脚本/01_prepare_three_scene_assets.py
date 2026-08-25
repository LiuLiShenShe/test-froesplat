#!/usr/bin/env python3
"""Prepare stable mask and optional view-weight assets for M2M3 pilot runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def split_scenes(value: str) -> set[str] | None:
    value = value.strip()
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def read_manifest(path: Path, selected: set[str] | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if selected is None:
        return rows
    kept = [row for row in rows if row["scene_id"] in selected]
    missing = sorted(selected - {row["scene_id"] for row in kept})
    if missing:
        raise SystemExit(f"Unknown scene(s) in manifest: {', '.join(missing)}")
    return kept


def image_files(source_dir: Path) -> list[Path]:
    image_dir = source_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {image_dir}")
    files = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No input images found in: {image_dir}")
    return files


def require_colmap(source_dir: Path) -> None:
    sparse_dir = source_dir / "sparse" / "0"
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        path = sparse_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing COLMAP file: {path}")


def mask_candidates(mask_dir: Path, stem: str) -> Iterable[Path]:
    stems = [stem]
    if stem.startswith("crop_"):
        stems.append(stem[len("crop_") :])
    for item in stems:
        for suffix in MASK_SUFFIXES:
            yield mask_dir / f"mask_{item}{suffix}"
            yield mask_dir / f"{item}{suffix}"
            yield mask_dir / "masks" / f"mask_{item}{suffix}"
            yield mask_dir / "masks" / f"{item}{suffix}"


def resolve_mask(mask_dir: Path, stem: str) -> Path | None:
    for candidate in mask_candidates(mask_dir, stem):
        if candidate.is_file():
            return candidate
    return None


def validate_image_file(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - report bad input assets without hiding the path.
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def validate_mask_file(
    path: Path,
    threshold: int = 128,
    min_foreground_ratio: float = 0.001,
    max_foreground_ratio: float = 0.95,
) -> tuple[bool, str, dict[str, object]]:
    try:
        with Image.open(path) as image:
            mask = image.convert("L")
            hist = mask.histogram()
            foreground = sum(hist[threshold:])
            total = mask.width * mask.height
    except Exception as exc:  # noqa: BLE001 - report bad input assets without hiding the path.
        return False, f"{type(exc).__name__}: {exc}", {}

    ratio = foreground / total if total else 0.0
    details = {
        "width": mask.width,
        "height": mask.height,
        "foreground_pixels": foreground,
        "foreground_ratio": ratio,
    }
    if total == 0:
        return False, "empty image dimensions", details
    if ratio < min_foreground_ratio:
        return False, f"foreground_ratio below {min_foreground_ratio}: {ratio:.6f}", details
    if ratio > max_foreground_ratio:
        return False, f"foreground_ratio above {max_foreground_ratio}: {ratio:.6f}", details
    return True, "", details


def link_or_copy(src: Path, dst: Path, copy_files: bool, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if force:
            dst.unlink()
        else:
            return
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def normalize_key(value: str) -> str:
    value = Path(str(value).strip()).stem
    return value


def load_weights(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    weights: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return weights
        name_key = next((k for k in ("image_name", "name", "image", "filename", "stem") if k in reader.fieldnames), reader.fieldnames[0])
        weight_key = next((k for k in ("weight", "view_weight") if k in reader.fieldnames), None)
        if weight_key is None:
            return weights
        for row in reader:
            key = normalize_key(row.get(name_key, ""))
            value = str(row.get(weight_key, "")).strip()
            if key and value:
                weights[key] = value
    return weights


def write_normalized_weights(raw_path: Path, target_path: Path, stems: list[str]) -> dict[str, object]:
    if not raw_path.is_file():
        return {"enabled": False, "raw_path": str(raw_path), "prepared_path": "", "matched": 0, "missing": len(stems)}
    raw = load_weights(raw_path)
    rows = []
    missing = []
    for stem in stems:
        keys = [stem]
        if stem.startswith("crop_"):
            keys.append(stem[len("crop_") :])
        value = next((raw[key] for key in keys if key in raw), None)
        if value is None:
            missing.append(stem)
            continue
        rows.append({"image_name": stem, "weight": value})
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "enabled": True,
        "raw_path": str(raw_path),
        "prepared_path": str(target_path),
        "matched": len(rows),
        "missing": len(missing),
        "missing_examples": missing[:10],
    }


def summarize_weights(raw_path: Path, target_path: Path, stems: list[str]) -> dict[str, object]:
    if not raw_path.is_file():
        return {"enabled": False, "raw_path": str(raw_path), "prepared_path": "", "matched": 0, "missing": len(stems)}
    raw = load_weights(raw_path)
    missing = []
    matched = 0
    for stem in stems:
        keys = [stem]
        if stem.startswith("crop_"):
            keys.append(stem[len("crop_") :])
        if any(key in raw for key in keys):
            matched += 1
        else:
            missing.append(stem)
    return {
        "enabled": True,
        "raw_path": str(raw_path),
        "prepared_path": str(target_path),
        "matched": matched,
        "missing": len(missing),
        "missing_examples": missing[:10],
    }


def prepare_scene(
    row: dict[str, str],
    output_root: Path,
    copy_files: bool,
    force: bool,
    check_only: bool,
    allow_missing_masks: bool,
) -> dict[str, object]:
    scene_id = row["scene_id"]
    source_dir = Path(row["source_dir"])
    raw_mask_dir = Path(row["raw_mask_dir"])
    raw_weight = Path(row["raw_view_weight"]) if row.get("raw_view_weight") else None

    require_colmap(source_dir)
    images = image_files(source_dir)
    stems = [p.stem for p in images]

    invalid_images: list[dict[str, str]] = []
    for image in images:
        ok, error = validate_image_file(image)
        if not ok:
            invalid_images.append({"stem": image.stem, "path": str(image), "error": error})

    missing_masks: list[str] = []
    invalid_masks: list[dict[str, str]] = []
    resolved_masks: list[tuple[str, Path]] = []
    mask_ratios: list[float] = []
    for stem in stems:
        mask = resolve_mask(raw_mask_dir, stem)
        if mask is None:
            missing_masks.append(stem)
        else:
            ok, error, details = validate_mask_file(mask)
            if ok:
                resolved_masks.append((stem, mask))
                ratio = details.get("foreground_ratio")
                if isinstance(ratio, float):
                    mask_ratios.append(ratio)
            else:
                missing_masks.append(stem)
                invalid_masks.append({"stem": stem, "path": str(mask), "error": error, **details})

    prepared_mask_dir = output_root / "prepared_masks" / scene_id
    prepared_weight_path = output_root / "prepared_view_weights" / f"{scene_id}_view_weights.csv"
    prepared_gate_path = output_root / "prepared_gate_lists" / f"{scene_id}_mask_matched.txt"

    if not check_only:
        if force and prepared_mask_dir.exists():
            shutil.rmtree(prepared_mask_dir)
        for stem, mask in resolved_masks:
            link_or_copy(mask, prepared_mask_dir / f"mask_{stem}.png", copy_files=copy_files, force=force)
        prepared_gate_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_gate_path.write_text("\n".join(stem for stem, _ in resolved_masks) + "\n", encoding="utf-8")

    weight_summary = {
        "enabled": False,
        "raw_path": str(raw_weight) if raw_weight else "",
        "prepared_path": "",
        "matched": 0,
        "missing": 0,
    }
    if raw_weight:
        if not check_only:
            weight_summary = write_normalized_weights(raw_weight, prepared_weight_path, stems)
        else:
            weight_summary = summarize_weights(raw_weight, prepared_weight_path, stems)

    summary = {
        "scene_id": scene_id,
        "zh_name": row.get("zh_name", ""),
        "source_dir": str(source_dir),
        "raw_mask_dir": str(raw_mask_dir),
        "prepared_mask_dir": str(prepared_mask_dir),
        "prepared_gate_list": str(prepared_gate_path),
        "image_count": len(stems),
        "invalid_image_count": len(invalid_images),
        "invalid_image_examples": invalid_images[:10],
        "mask_count": len(resolved_masks),
        "missing_mask_count": len(missing_masks),
        "missing_mask_examples": missing_masks[:10],
        "invalid_mask_count": len(invalid_masks),
        "invalid_mask_examples": invalid_masks[:10],
        "mask_foreground_ratio_summary": {
            "min": min(mask_ratios) if mask_ratios else "",
            "median": statistics.median(mask_ratios) if mask_ratios else "",
            "max": max(mask_ratios) if mask_ratios else "",
        },
        "view_weight": weight_summary,
        "mask_source_note": row.get("mask_source_note", ""),
    }
    if invalid_images:
        summary["status"] = "invalid_images"
    elif missing_masks:
        summary["status"] = "ok_with_gate" if allow_missing_masks and resolved_masks else "missing_masks"
    elif raw_weight and weight_summary.get("enabled") and weight_summary.get("matched", 0) == 0:
        summary["status"] = "view_weight_unmatched"
    else:
        summary["status"] = "ok"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--copy", action="store_true", help="Copy mask files instead of creating symlinks.")
    parser.add_argument("--force", action="store_true", help="Replace existing prepared symlinks/files.")
    parser.add_argument("--check-only", action="store_true", help="Validate paths without preparing mask links.")
    parser.add_argument(
        "--allow-missing-masks",
        action="store_true",
        help="Allow scenes with partial masks; a matched-view gate list will be generated for training.",
    )
    args = parser.parse_args()

    selected = split_scenes(args.scenes)
    rows = read_manifest(args.manifest, selected)
    summaries = []
    failed = False
    for row in rows:
        try:
            summary = prepare_scene(
                row,
                args.output_root,
                args.copy,
                args.force,
                args.check_only,
                args.allow_missing_masks,
            )
        except Exception as exc:  # noqa: BLE001 - this is a CLI validation report.
            failed = True
            summary = {
                "scene_id": row.get("scene_id", ""),
                "zh_name": row.get("zh_name", ""),
                "status": "error",
                "error": str(exc),
            }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary.get("status") not in ("ok", "ok_with_gate"):
            failed = True

    report_dir = args.output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not args.check_only:
        report_path = report_dir / "asset_preparation_summary.json"
        report_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {report_path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
