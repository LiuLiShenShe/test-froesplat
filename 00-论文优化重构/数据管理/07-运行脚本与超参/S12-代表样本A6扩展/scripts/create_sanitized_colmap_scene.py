#!/usr/bin/env python3
"""Create a COLMAP scene copy that excludes unreadable image files.

The original final_locked scene is left untouched. The output scene keeps the
same sparse model except that images.bin is rewritten without invalid images.
Readable images are symlinked into the sanitized scene.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

from PIL import Image


def read_next_bytes(fid, num_bytes: int, fmt: str):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def read_images_binary(path: Path) -> dict[int, dict]:
    images = {}
    with path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = int(props[0])
            qvec = tuple(float(v) for v in props[1:5])
            tvec = tuple(float(v) for v in props[5:8])
            camera_id = int(props[8])
            name_bytes = bytearray()
            ch = read_next_bytes(fid, 1, "c")[0]
            while ch != b"\x00":
                name_bytes.extend(ch)
                ch = read_next_bytes(fid, 1, "c")[0]
            name = name_bytes.decode("utf-8")
            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            raw = read_next_bytes(fid, 24 * num_points2d, "ddq" * num_points2d)
            xys = [(float(raw[i]), float(raw[i + 1])) for i in range(0, len(raw), 3)]
            point3d_ids = [int(raw[i]) for i in range(2, len(raw), 3)]
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "xys": xys,
                "point3D_ids": point3d_ids,
            }
    return images


def write_images_binary(images: dict[int, dict], path: Path) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image_id in sorted(images):
            image = images[image_id]
            fid.write(
                struct.pack(
                    "<idddddddi",
                    int(image["id"]),
                    *image["qvec"],
                    *image["tvec"],
                    int(image["camera_id"]),
                )
            )
            fid.write(image["name"].encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image["xys"])))
            for (x, y), point_id in zip(image["xys"], image["point3D_ids"]):
                fid.write(struct.pack("<ddq", float(x), float(y), int(point_id)))


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def relink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--output-scene", required=True, type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--mask-pattern", default="mask_{stem}.png")
    args = parser.parse_args()

    src = args.source_scene
    out = args.output_scene
    src_sparse = src / "sparse" / "0"
    out_sparse = out / "sparse" / "0"
    out_images = out / "images"
    out_sparse.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    images = read_images_binary(src_sparse / "images.bin")
    kept = {}
    dropped = []
    mask_errors = []

    for image_id, image in images.items():
        image_path = src / "images" / image["name"]
        if not image_path.exists():
            dropped.append({"id": image_id, "name": image["name"], "reason": "missing_image"})
            continue
        if not is_readable_image(image_path):
            dropped.append({"id": image_id, "name": image["name"], "reason": "unreadable_image"})
            continue
        if args.mask_dir is not None:
            stem = Path(image["name"]).stem
            mask_path = args.mask_dir / args.mask_pattern.format(stem=stem, image_name=image["name"])
            if not mask_path.exists():
                dropped.append({"id": image_id, "name": image["name"], "reason": "missing_mask"})
                continue
            if not is_readable_image(mask_path):
                mask_errors.append({"id": image_id, "name": image["name"], "mask": str(mask_path)})
                dropped.append({"id": image_id, "name": image["name"], "reason": "unreadable_mask"})
                continue
        kept[image_id] = image
        relink(image_path, out_images / image["name"])

    for name in ("cameras.bin", "points3D.bin", "points3D.ply"):
        src_file = src_sparse / name
        if src_file.exists():
            shutil.copy2(src_file, out_sparse / name)
    write_images_binary(kept, out_sparse / "images.bin")

    report = {
        "source_scene": str(src),
        "output_scene": str(out),
        "images_before": len(images),
        "images_after": len(kept),
        "dropped_count": len(dropped),
        "dropped": dropped,
        "mask_errors_count": len(mask_errors),
        "mask_errors": mask_errors,
    }
    (out / "sanitize_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not kept:
        raise RuntimeError("No readable images remain after sanitizing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
