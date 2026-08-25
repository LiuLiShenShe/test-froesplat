#!/usr/bin/env python3
"""Extract or index RGB frames for the ForeSplat workflow."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def copy_existing_images(image_dir: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for idx, src in enumerate(list_images(image_dir)):
        dst = output / f"{idx:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def extract_video_frames(video: Path, output: Path, fps: float) -> list[Path]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV is required for video extraction. Install opencv-python or use --image_dir.") from exc

    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    stride = max(1, int(round(source_fps / fps)))
    saved: list[Path] = []
    frame_index = 0
    save_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride == 0:
            dst = output / f"{save_index:06d}.jpg"
            cv2.imwrite(str(dst), frame)
            saved.append(dst)
            save_index += 1
        frame_index += 1

    cap.release()
    return saved


def write_manifest(frames: list[Path], output: Path) -> None:
    manifest = output / "frame_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "file_name"])
        writer.writeheader()
        for idx, frame in enumerate(frames):
            writer.writerow({"frame_id": idx, "file_name": frame.name})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Input RGB video.")
    source.add_argument("--image_dir", type=Path, help="Existing image directory to index/copy.")
    parser.add_argument("--output", type=Path, required=True, help="Output frame directory.")
    parser.add_argument("--fps", type=float, default=3.0, help="Sampling FPS for video input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_dir:
        frames = copy_existing_images(args.image_dir, args.output)
    else:
        frames = extract_video_frames(args.video, args.output, args.fps)
    write_manifest(frames, args.output)
    print(f"Wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
