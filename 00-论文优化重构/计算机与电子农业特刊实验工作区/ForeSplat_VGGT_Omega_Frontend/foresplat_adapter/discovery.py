"""Batch data discovery for the ForeSplat VGGT-Omega full-sequence pipeline.

Scans a root folder that contains one sub-directory per plant.  Each plant
directory holds the SAM3 segmentation outputs of one acquisition sequence::

    <root>/<Plant>/
        crop_XXXX.png      foreground RGB (black background)
        mask_XXXX.png      binary foreground mask
        segmentation_log.json   (optional, ignored here)

Each plant is one independent scene; a plant is never mixed with another.

Discovery is *tolerant* but never silent: frames whose files fail the PNG
magic / PIL structural check are **quarantined** (recorded with a reason),
and only the surviving paired frames are forwarded to the pipeline.  This is
the only situation in which frames are excluded - usable frames are never
sub-sampled or dropped (no striding, no windowing).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from PIL import Image

IMAGE_GLOB_PREFIX = "crop_"
MASK_GLOB_PREFIX = "mask_"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass
class FrameScan:
    """One paired frame with its integrity status."""

    stem: str
    image_path: str
    mask_path: str
    valid: bool
    reason: str = ""


@dataclass
class PlantScan:
    """Discovery result for one plant directory (= one scene)."""

    plant_id: str
    plant_dir: str
    images_dir: str
    masks_dir: str
    frames: list[FrameScan] = field(default_factory=list)
    image_resolution: tuple[int, int] | None = None
    prefix_mapping: str = "crop_->mask_"

    @property
    def total_images(self) -> int:
        return len(self.frames)

    @property
    def valid_frames(self) -> list[FrameScan]:
        return [f for f in self.frames if f.valid]

    @property
    def matched_images(self) -> int:
        return len(self.valid_frames)

    @property
    def quarantined(self) -> list[FrameScan]:
        return [f for f in self.frames if not f.valid]


def _png_magic_ok(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(8) == PNG_MAGIC
    except OSError:
        return False


def _validate_image(path: str) -> str | None:
    """Return a failure reason, or None if the PNG is structurally valid."""
    if not _png_magic_ok(path):
        return "corrupt or not a PNG (bad magic)"
    try:
        with Image.open(path) as img:
            img.verify()  # structural check (does not decode pixels)
    except Exception as exc:  # noqa: BLE001 - report any decode failure
        return f"corrupt image data ({type(exc).__name__})"
    return None


def scan_plant(
    plant_id: str,
    plant_dir: str,
    image_prefix: str = IMAGE_GLOB_PREFIX,
    mask_prefix: str = MASK_GLOB_PREFIX,
) -> PlantScan:
    """Pair ``crop_XXXX`` images with ``mask_XXXX`` masks inside one plant dir."""
    images_dir = masks_dir = plant_dir
    digits = lambda name, prefix: (  # noqa: E731
        name[len(prefix) : -4] if name.startswith(prefix) and name.endswith(".png") else None
    )

    images: dict[str, str] = {}
    masks: dict[str, str] = {}
    for name in sorted(os.listdir(plant_dir)):
        path = os.path.join(plant_dir, name)
        if not os.path.isfile(path) or not name.endswith(".png"):
            continue
        key = digits(name, image_prefix)
        if key is not None and key not in images:
            images[key] = path
            continue
        key = digits(name, mask_prefix)
        if key is not None and key not in masks:
            masks[key] = path

    scan = PlantScan(
        plant_id=plant_id,
        plant_dir=plant_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
    )

    for stem in sorted(set(images) & set(masks)):
        reason = _validate_image(images[stem]) or _validate_image(masks[stem])
        scan.frames.append(
            FrameScan(
                stem=stem,
                image_path=images[stem],
                mask_path=masks[stem],
                valid=reason is None,
                reason=reason or "",
            )
        )

    for stem in sorted(set(masks) - set(images)):
        scan.frames.append(
            FrameScan(stem=stem, image_path="", mask_path=masks[stem], valid=False,
                      reason="mask without matching foreground image")
        )
    for stem in sorted(set(images) - set(masks)):
        scan.frames.append(
            FrameScan(stem=stem, image_path=images[stem], mask_path="", valid=False,
                      reason="image without matching mask")
        )

    scan.frames.sort(key=lambda f: f.stem)
    if scan.valid_frames:
        with Image.open(scan.valid_frames[0].image_path) as img:
            scan.image_resolution = (img.width, img.height)
    return scan


def discover_plants(
    root_dir: str,
    image_prefix: str = IMAGE_GLOB_PREFIX,
    mask_prefix: str = MASK_GLOB_PREFIX,
) -> list[PlantScan]:
    """Scan every sub-directory of ``root_dir`` as one plant scene."""
    plants = []
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        scan = scan_plant(name, path, image_prefix=image_prefix, mask_prefix=mask_prefix)
        if scan.frames:
            plants.append(scan)
    return plants


def write_manifest(plants: list[PlantScan], csv_path: str, output_root: str) -> None:
    """Write the batch manifest (one row per plant)."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "plant_id", "images_dir", "masks_dir", "total_images",
                "matched_images", "quarantined_frames", "image_resolution",
                "prefix_mapping", "output_dir", "processing_status",
            ]
        )
        for plant in plants:
            status = "ready" if plant.matched_images >= 2 else "insufficient_frames"
            writer.writerow(
                [
                    plant.plant_id,
                    plant.images_dir,
                    plant.masks_dir,
                    plant.total_images,
                    plant.matched_images,
                    len(plant.quarantined),
                    f"{plant.image_resolution[0]}x{plant.image_resolution[1]}" if plant.image_resolution else "",
                    plant.prefix_mapping,
                    os.path.join(output_root, plant.plant_id),
                    status,
                ]
            )


def write_quarantine_csvs(plants: list[PlantScan], output_root: str) -> None:
    """Write ``metadata/quarantined_frames.csv`` per plant (created lazily)."""
    for plant in plants:
        bad = plant.quarantined
        if not bad:
            continue
        out_dir = os.path.join(output_root, plant.plant_id, "metadata")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "quarantined_frames.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stem", "image_path", "mask_path", "reason"])
            for frame in bad:
                writer.writerow([frame.stem, frame.image_path, frame.mask_path, frame.reason])