from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache_metadata(image_path: Path, source_dir: Path, metadata_dir: Path | None = None):
    roots = [metadata_dir] if metadata_dir is not None else [source_dir / "候选掩膜", source_dir]
    for root in roots:
        path = root / f"候选元数据_{image_path.stem}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if payload.get("schema_version") != 1 or payload.get("image_name") != image_path.name:
                raise ValueError(f"Candidate metadata image/schema mismatch: {path}")
            image_hash = payload.get("image_sha256", "")
            if image_hash and image_hash != file_sha256(image_path):
                raise ValueError(f"Candidate metadata RGB SHA256 mismatch: {path}")
            indexed = {}
            orders = set()
            for entry in payload["candidates"]:
                key = (entry["prompt_id"], int(entry["instance_id"]))
                if key in indexed:
                    raise ValueError(f"Duplicate candidate identity {key}: {path}")
                try:
                    score = float(entry["sam_score"])
                    threshold = float(entry["mask_threshold"])
                    box = tuple(entry["box"]) if entry["box"] is not None else None
                    order = entry["candidate_order"]
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Invalid SAM confidence")
                    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                        raise ValueError("Invalid mask threshold")
                    legacy_or_empty = entry.get("prompt_mode") == "legacy_union" or entry.get("raw_detection_count") == 0
                    if box is None and not legacy_or_empty:
                        raise ValueError("Missing candidate box")
                    if box is not None and (len(box) != 4 or any(not isinstance(value, int) for value in box)):
                        raise ValueError("Invalid candidate box")
                    if not isinstance(order, int) or order < 0 or order in orders:
                        raise ValueError("Invalid/duplicate candidate order")
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid candidate metadata {key}: {path}: {error}") from error
                orders.add(order)
                indexed[key] = {**entry, "sam_score": score, "mask_threshold": threshold,
                                "box": box, "image_sha256": image_hash,
                                "source_image_name": payload["image_name"], "metadata_path": str(path)}
            return indexed
    for root in roots:
        path = root / f"候选评分明细_{image_path.stem}.csv"
        if not path.exists():
            continue
        indexed = {}
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["图像"] != image_path.name:
                    raise ValueError(f"Historical candidate metadata image mismatch: {path}")
                key = (row["提示词编号"], int(row["实例编号"]))
                if key in indexed:
                    raise ValueError(f"Duplicate candidate identity {key}: {path}")
                try:
                    score = float(row["SAM3分数"])
                    box = tuple(int(value) for value in row["外接框"].split(",")) if row["外接框"] else None
                    threshold = float(row["掩膜阈值"])
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Invalid historical confidence/box")
                    if box is None and row["候选模式"] != "legacy_union":
                        raise ValueError("Missing historical candidate box")
                    if box is not None and len(box) != 4:
                        raise ValueError("Invalid historical box")
                    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                        raise ValueError("Invalid historical mask threshold")
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid historical metadata {key}: {path}") from error
                indexed[key] = {"sam_score": score, "box": box, "mask_threshold": threshold,
                                "candidate_order": key[1], "source_image_name": row["图像"],
                                "prompt_mode": row["候选模式"], "metadata_path": str(path)}
        return indexed
    return {}


def metadata_for_mask(indexed, prompt_id: str, instance_id: int, mask_path: Path, strict: bool):
    entry = indexed.get((prompt_id, instance_id))
    if entry and entry.get("mask_sha256") and entry["mask_sha256"] != file_sha256(mask_path):
        raise ValueError(f"Candidate mask SHA256 mismatch: {mask_path}")
    verified = bool(entry and entry.get("mask_sha256") and entry.get("image_sha256") and entry.get("box") is not None
                    and entry.get("metadata_status", "VERIFIED") in ("VERIFIED", "GENERATED"))
    if not verified:
        message = f"METADATA_UNVERIFIED: {prompt_id}/{instance_id} at {mask_path}; score-equivalent replay is not certified"
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return entry or {}, "VERIFIED" if verified else "METADATA_UNVERIFIED"
