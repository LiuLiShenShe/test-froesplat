"""Input dataset discovery for the ForeSplat VGGT-Omega frontend adapter.

Expects the RAP-FSAM3 output layout::

    input_dataset/
        images/   000001.png ...   (RAP-FSAM3 processed foreground RGB, or RGBA)
        masks/    000001.png ...   (binary foreground mask, same stem as the image)

Image files are paired with mask files by identical stem (extension may
differ).  RGBA foreground images are composited over black by the loader
(``_load_foreground_rgb_image``) so the RGB input reproduces the original
RAP-FSAM3 foreground background form; the *independent* binary mask is always
kept and is never replaced by a "black pixel" heuristic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class FramePaths:
    """One frame: a RAP-FSAM3 foreground image paired with its binary mask."""

    index: int
    image_path: str
    mask_path: str
    image_name: str

    @property
    def stem(self) -> str:
        return os.path.splitext(self.image_name)[0]


def _list_dir(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(
        n for n in os.listdir(directory) if not n.startswith(".") and os.path.isfile(os.path.join(directory, n))
    )


def _strip_prefix(stem: str, prefix: str | None) -> str:
    if prefix and stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _tail_digits(stem: str) -> str | None:
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return digits or None


def _build_mask_index(masks_dir: str, mask_prefix: str | None) -> tuple[dict[str, str], dict[str, list[str]]]:
    by_key: dict[str, str] = {}
    by_tail: dict[str, list[str]] = {}
    for name in _list_dir(masks_dir):
        if os.path.splitext(name)[1].lower() not in MASK_EXTENSIONS:
            continue
        stem = _strip_prefix(os.path.splitext(name)[0], mask_prefix)
        by_key[stem] = os.path.join(masks_dir, name)
        tail = _tail_digits(stem)
        if tail is not None:
            by_tail.setdefault(tail, []).append(os.path.join(masks_dir, name))
    return by_key, by_tail


def collect_frames(
    images_dir: str,
    masks_dir: str,
    max_frames: int | None = None,
    frame_stride: int = 1,
    image_prefix: str | None = None,
    mask_prefix: str | None = None,
) -> list[FramePaths]:
    """Pair ``images_dir`` with ``masks_dir`` by file stem, preserving view order.

    ``image_prefix`` / ``mask_prefix`` are stripped from the stems before pairing
    (e.g. RAP-FSAM3 writes ``crop_0000.png`` next to ``mask_0000.png``: pass
    ``--image-prefix-to-strip crop_ --mask-prefix-to-strip mask_``).  If the exact
    stem still does not match, pairing falls back to the trailing digit run
    (``0000``) as long as it is unique among the masks.

    ``frame_stride`` and ``max_frames`` only *subset* the sequence - the
    original relative order of the selected views is never changed, so VGGT-Omega
    sees the same viewpoint ordering as the input directory.
    """
    if frame_stride <= 0:
        raise ValueError("frame_stride must be >= 1")

    images = [n for n in _list_dir(images_dir) if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS]
    if not images:
        raise ValueError(f"No images found in {images_dir}")

    mask_by_key, mask_by_tail = _build_mask_index(masks_dir, mask_prefix)

    frames: list[FramePaths] = []
    for name in images:
        stem = _strip_prefix(os.path.splitext(name)[0], image_prefix)
        mask_path = mask_by_key.get(stem)
        if mask_path is None:
            tail = _tail_digits(stem)
            candidates = mask_by_tail.get(tail, []) if tail is not None else []
            if len(candidates) == 1:
                mask_path = candidates[0]
        if mask_path is None:
            raise ValueError(
                f"No mask found for image {name!r} (pairing key {stem!r}) in {masks_dir}. "
                "The ForeSplat adapter requires an independent RAP-FSAM3 binary mask per image "
                "(black-background heuristics are not used). Use --image-prefix-to-strip / "
                "--mask-prefix-to-strip when the two directories use different name prefixes."
            )
        frames.append(
            FramePaths(
                index=len(frames),
                image_path=os.path.join(images_dir, name),
                mask_path=mask_path,
                image_name=name,
            )
        )

    if frame_stride > 1:
        frames = frames[::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    if len(frames) < 2:
        raise ValueError("VGGT-Omega needs at least 2 frames to estimate camera poses")
    for i, frame in enumerate(frames):
        frame.index = i
    return frames


def foreground_ratios(mask_paths: list[str], mask_threshold: int = 127) -> list[float]:
    """Foreground pixel ratio of each mask at its *original* resolution."""
    ratios = []
    for path in mask_paths:
        with Image.open(path) as mask:
            if mask.mode not in ("L", "1"):
                mask = mask.convert("L")
            array = np.asarray(mask)
        ratios.append(float((array > mask_threshold).mean()))
    return ratios


def load_mask_binary(mask_path: str, mask_threshold: int = 127) -> np.ndarray:
    """Load a mask as a ``uint8`` array of 0/255 at its original resolution."""
    with Image.open(mask_path) as mask:
        if mask.mode not in ("L", "1"):
            mask = mask.convert("L")
        return (np.asarray(mask) > mask_threshold).astype(np.uint8) * 255
