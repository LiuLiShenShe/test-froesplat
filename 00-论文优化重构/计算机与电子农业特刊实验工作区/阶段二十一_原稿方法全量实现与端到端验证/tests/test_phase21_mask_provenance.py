#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — mask identity / provenance closure tests.

  T1 manifest binds camera -> path -> SHA256 -> foreground fraction
  T2 verification passes for identical hashes
  T3 verification fails loudly on a different mask version
  T4 missing masks are recorded, never silently ignored
  T5 resolve_mask_path honours the configured pattern
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TRAINER = Path("/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40")
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from utils import plant_aware_utils as pu  # noqa: E402


def make_camera(name, height=8, width=8, foreground=True):
    mask = torch.zeros((1, height, width))
    if foreground:
        mask[:, 2:6, 2:6] = 1.0
    return SimpleNamespace(image_name=name, gt_alpha_mask=mask)


def write_mask(path: Path, value: int = 255):
    from PIL import Image
    array = np.zeros((8, 8), dtype=np.uint8)
    array[2:6, 2:6] = value
    Image.fromarray(array).save(path)


def test_t1_manifest_hashes(tmp_path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    write_mask(mask_dir / "mask_a.png")
    write_mask(mask_dir / "mask_b.png", value=128)
    cameras = [make_camera("a.jpg"), make_camera("b.jpg")]
    manifest = pu.write_mask_provenance(
        cameras, str(mask_dir), "mask_{stem}.png", 0.5,
        str(tmp_path / "manifest.json"))
    assert manifest["camera_count"] == 2
    assert manifest["missing_count"] == 0
    assert all(len(entry["sha256"]) == 64 for entry in manifest["entries"])
    assert manifest["entries"][0]["foreground_fraction"] == pytest.approx(16 / 64)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["entries"][0]["image"] == "a.jpg"


def test_t2_verification_passes(tmp_path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    write_mask(mask_dir / "mask_a.png")
    cameras = [make_camera("a.jpg")]
    manifest = pu.write_mask_provenance(cameras, str(mask_dir), "mask_{stem}.png",
                                        0.5, "")
    pu.verify_mask_provenance(manifest, manifest)  # must not raise


def test_t3_verification_fails_on_new_version(tmp_path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    write_mask(mask_dir / "mask_a.png")
    cameras = [make_camera("a.jpg")]
    first = pu.write_mask_provenance(cameras, str(mask_dir), "mask_{stem}.png", 0.5, "")
    # rewrite with different content -> different hash
    write_mask(mask_dir / "mask_a.png", value=64)
    second = pu.write_mask_provenance(cameras, str(mask_dir), "mask_{stem}.png", 0.5, "")
    with pytest.raises(ValueError, match="provenance mismatch"):
        pu.verify_mask_provenance(second, first)


def test_t4_missing_mask_is_recorded(tmp_path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    write_mask(mask_dir / "mask_a.png")
    cameras = [make_camera("a.jpg"), make_camera("missing.jpg")]
    manifest = pu.write_mask_provenance(cameras, str(mask_dir), "mask_{stem}.png", 0.5, "")
    assert manifest["missing_count"] == 1
    missing = [entry for entry in manifest["entries"] if entry["status"] == "MISSING"]
    assert missing[0]["image"] == "missing.jpg"


def test_t5_resolve_pattern(tmp_path):
    (tmp_path / "mask_x.png").write_bytes(b"fake")
    resolved = pu.resolve_mask_path(str(tmp_path), "x.jpg", "mask_{stem}.png")
    assert resolved.name == "mask_x.png"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
