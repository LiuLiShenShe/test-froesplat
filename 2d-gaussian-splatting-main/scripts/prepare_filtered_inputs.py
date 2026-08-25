#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def rebuild_input_dir(input_dir: Path, valid_files: list[Path]) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for existing in input_dir.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
        elif existing.is_dir():
            raise RuntimeError(f"unexpected directory inside managed input dir: {existing}")

    for source_file in valid_files:
        target = input_dir / source_file.name
        target.symlink_to(source_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare filtered 2DGS scene inputs")
    parser.add_argument("--input_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--scene", action="append", default=[], help="Optional scene name, repeatable")
    args = parser.parse_args()

    requested = set(args.scene)
    scenes = [
        path
        for path in sorted(args.input_root.iterdir())
        if path.is_dir() and path.name != "ffmpeg_bin" and (not requested or path.name in requested)
    ]

    global_rows: list[dict[str, object]] = []
    for scene_path in scenes:
        scene_output = args.output_root / scene_path.name
        input_dir = scene_output / "input"
        valid_files: list[Path] = []
        bad_files: list[str] = []

        image_files = sorted([path for path in scene_path.iterdir() if path.is_file()])
        for image_file in image_files:
            if is_valid_image(image_file):
                valid_files.append(image_file)
            else:
                bad_files.append(image_file.name)

        rebuild_input_dir(input_dir, valid_files)

        excluded_path = scene_output / "excluded_bad_images.txt"
        excluded_path.parent.mkdir(parents=True, exist_ok=True)
        excluded_path.write_text("\n".join(bad_files) + ("\n" if bad_files else ""), encoding="utf-8")

        summary = {
            "scene": scene_path.name,
            "source_dir": str(scene_path),
            "output_dir": str(scene_output),
            "total_files": len(image_files),
            "valid_files": len(valid_files),
            "bad_files": len(bad_files),
            "bad_image_names": bad_files,
        }
        (scene_output / "input_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        global_rows.append(summary)

    (args.output_root / "input_preparation_summary.json").write_text(
        json.dumps(global_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
