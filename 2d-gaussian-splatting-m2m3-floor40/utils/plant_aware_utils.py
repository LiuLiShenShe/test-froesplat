import csv
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image


NONE_VALUES = {"", "none", "None", "NONE", None}


def enabled(value: object) -> bool:
    return value not in NONE_VALUES


def read_name_set(path: str) -> set[str]:
    gate_path = Path(path)
    if not gate_path.exists():
        raise FileNotFoundError(f"View-quality retained list not found: {gate_path}")

    names: list[str] = []
    if gate_path.suffix.lower() == ".json":
        data = json.loads(gate_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("accepted", "retained", "keep", "images", "frames"):
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"Unsupported retained-list JSON format: {gate_path}")
        names = [str(item) for item in data]
    elif gate_path.suffix.lower() == ".csv":
        with gate_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return set()
            name_key = next(
                (k for k in ("image_name", "filename", "file", "name", "stem") if k in reader.fieldnames),
                reader.fieldnames[0],
            )
            keep_key = next(
                (k for k in ("keep", "accepted", "retained", "use") if k in reader.fieldnames),
                None,
            )
            for row in reader:
                if keep_key is not None:
                    keep_value = str(row.get(keep_key, "")).strip().lower()
                    if keep_value in {"0", "false", "no", "reject", "rejected"}:
                        continue
                names.append(str(row.get(name_key, "")))
    else:
        names = [
            line.strip()
            for line in gate_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    return {Path(name).stem for name in names if str(name).strip()}


def filter_by_name_set(items: Iterable[object], keep_names: set[str]) -> list[object]:
    return [item for item in items if Path(getattr(item, "image_name")).stem in keep_names]


def resolve_mask_path(mask_dir: str, image_name: str, pattern: str = "mask_{stem}.png") -> Path:
    root = Path(mask_dir)
    stem = Path(image_name).stem
    candidates = []
    if pattern:
        candidates.append(root / pattern.format(stem=stem, image_name=image_name))
    for suffix in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        candidates.extend(
            [
                root / f"{stem}{suffix}",
                root / f"mask_{stem}{suffix}",
                root / "masks" / f"{stem}{suffix}",
                root / "masks" / f"mask_{stem}{suffix}",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Mask for image '{image_name}' not found in {root}. "
        f"Tried pattern '{pattern}' and common mask names."
    )


def load_mask_tensor(
    mask_dir: str,
    image_name: str,
    resolution: tuple[int, int],
    pattern: str = "mask_{stem}.png",
    threshold: float = 0.5,
) -> torch.Tensor:
    mask_path = resolve_mask_path(mask_dir, image_name, pattern)
    try:
        nearest = Image.Resampling.NEAREST
    except AttributeError:
        nearest = Image.NEAREST
    mask = Image.open(mask_path).convert("L").resize(resolution, nearest)
    tensor = torch.from_numpy(__import__("numpy").array(mask)).float() / 255.0
    return (tensor >= threshold).float().unsqueeze(0)


def erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return -F.max_pool2d(-mask[None], kernel_size=kernel, stride=1, padding=radius)[0]


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask[None], kernel_size=kernel, stride=1, padding=radius)[0]


def stable_mask_region(mask: torch.Tensor, boundary_px: int) -> torch.Tensor:
    if boundary_px <= 0:
        return torch.ones_like(mask)
    dilated = dilate_mask(mask, boundary_px)
    eroded = erode_mask(mask, boundary_px)
    return ((dilated - eroded) < 0.5).float()


def masked_mean(value: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return value.mean()
    denom = weight.sum().clamp_min(1.0)
    return (value * weight).sum() / denom


def ramp_weight(iteration: int, start_iter: int, warmup_iters: int) -> float:
    if iteration < start_iter:
        return 0.0
    if warmup_iters <= 0:
        return 1.0
    return min(1.0, float(iteration - start_iter + 1) / float(warmup_iters))


# ── Phase 21: mask identity / provenance closure ─────────────────────────────

def file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_mask_provenance(
    cameras,
    mask_dir: str,
    mask_pattern: str,
    mask_threshold: float,
    output_path: str,
    source_label: str = "",
    mask_mode: str = "",
) -> dict:
    """Record exactly which mask file backs each training camera.

    The manifest binds camera -> mask path -> file SHA256 -> tensor foreground
    fraction.  It is the closure required so that a run cannot silently train on
    a different mask version than the one produced by RAP-FSAM3.
    """
    import json
    from pathlib import Path

    entries = []
    for cam in cameras:
        tensor = getattr(cam, "gt_alpha_mask", None)
        if tensor is None:
            continue
        try:
            mask_path = resolve_mask_path(mask_dir, cam.image_name, mask_pattern)
        except FileNotFoundError as error:
            entries.append({
                "image": cam.image_name,
                "mask_path": None,
                "sha256": None,
                "status": "MISSING",
                "error": str(error),
            })
            continue
        entries.append({
            "image": cam.image_name,
            "mask_path": str(mask_path),
            "sha256": file_sha256(mask_path),
            "height": int(tensor.shape[-2]),
            "width": int(tensor.shape[-1]),
            "foreground_fraction": float(tensor.float().mean().item()),
            "threshold": float(mask_threshold),
            "status": "OK",
        })
    manifest = {
        "schema_version": 1,
        "source_label": source_label,
        "mask_mode": mask_mode,
        "mask_dir": str(Path(mask_dir)),
        "mask_pattern": mask_pattern,
        "mask_threshold": float(mask_threshold),
        "camera_count": len(entries),
        "missing_count": sum(1 for entry in entries if entry["status"] != "OK"),
        "entries": entries,
    }
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def verify_mask_provenance(manifest: dict, expected: dict) -> None:
    """Raise if the actual mask hashes differ from the expected manifest.

    `expected` maps image name -> sha256 (or a full provenance manifest dict).
    """
    if isinstance(expected, dict) and "entries" in expected:
        expected = {entry["image"]: entry.get("sha256") for entry in expected["entries"]}
    actual = {entry["image"]: entry.get("sha256") for entry in manifest.get("entries", [])}
    mismatches = []
    for image, expected_hash in expected.items():
        if expected_hash is None:
            continue
        if actual.get(image) != expected_hash:
            mismatches.append(f"{image}: expected {expected_hash}, got {actual.get(image)}")
    if mismatches:
        raise ValueError(
            "Mask provenance mismatch; refusing to train on a different mask version:\n  "
            + "\n  ".join(mismatches[:10])
            + (f"\n  ... and {len(mismatches) - 10} more" if len(mismatches) > 10 else "")
        )
