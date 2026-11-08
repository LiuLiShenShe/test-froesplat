# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preprocess NuScenes dataset into per-scene HDF5 files with depth maps.

Generates ``scene.h5`` per scene containing CAM_FRONT images, sparse LiDAR depth maps,
camera extrinsics, and intrinsics.

Output goes to ``{target_dir}/{split-suffix}/{scene}/scene.h5`` (e.g.
``.../trainval/scene-0013/scene.h5``), which is exactly what ``NuScenesH5``
reads with ``root_path=target_dir``, ``split=<split-suffix>``.

Usage:
    # Generate the evaluation set: the 50 scenes baked into the loader
    # (NUSCENES_EVAL_SCENES). Consumed by test_datasets/nuscenes.yaml (scenes: "eval").
    python -m dvlt.scripts.preprocess.nuscenes_h5.preprocess \
        --data_root datasets/nuscenes_raw \
        --target_dir datasets/test/nuscenes \
        --split v1.0-trainval \
        --scene_split eval \
        --lidarseg_root datasets/nuscenes_raw/v1.0-trainval \
        --workers 32

    # Process all 10 scenes in v1.0-mini
    python -m dvlt.scripts.preprocess.nuscenes_h5.preprocess \
        --data_root datasets/nuscenes_raw \
        --target_dir datasets/test/nuscenes \
        --split v1.0-mini \
        --num_scenes 10 \
        --workers 32

    # Process specific scenes
    python -m dvlt.scripts.preprocess.nuscenes_h5.preprocess \
        --data_root datasets/nuscenes_raw \
        --target_dir datasets/test/nuscenes \
        --split v1.0-mini \
        --scene_ids 0 1 2

    # Process the full official validation split from v1.0-trainval
    python -m dvlt.scripts.preprocess.nuscenes_h5.preprocess \
        --data_root datasets/nuscenes_raw \
        --target_dir datasets/test/nuscenes \
        --split v1.0-trainval \
        --scene_split val \
        --lidarseg_root datasets/nuscenes_raw/v1.0-trainval \
        --workers 32
"""

import argparse

from dvlt.data.sources.datasets.nuscenes_h5 import NUSCENES_EVAL_SCENES
from dvlt.scripts.preprocess.nuscenes_h5.processor import NuScenesProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="NuScenes dataset preprocessing")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of raw NuScenes data (contains v1.0-mini/, v1.0-trainval/, etc.)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="v1.0-mini",
        help="NuScenes split to process (e.g. v1.0-mini, v1.0-trainval)",
    )
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="Specific scene IDs to process (space-separated integers). Takes priority over start_idx/num_scenes.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start scene index (used when scene_ids not provided)",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=10,
        help="Number of scenes to process (used when scene_ids not provided)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for scene processing",
    )
    parser.add_argument(
        "--lidarseg_root",
        type=str,
        default=None,
        help=(
            "Path to directory containing lidarseg.json and category.json. "
            "Optional: when provided, noise (class 0) and ego-vehicle (class 31) LiDAR "
            "points are filtered out from depth maps. Without it, depth maps still work "
            "but use all LiDAR points. Example: datasets/nuscenes_mini/labels/"
        ),
    )
    parser.add_argument(
        "--scene_split",
        type=str,
        default=None,
        choices=["eval", "train", "val", "test", "mini_train", "mini_val", "train_detect", "train_track"],
        help=(
            "Only process scenes in this split. 'eval' is the loader's baked-in "
            "NUSCENES_EVAL_SCENES (the 50 scenes used by test_datasets/nuscenes.yaml); "
            "the others come from nuscenes.utils.splits (e.g. 'val' for the full "
            "validation set of v1.0-trainval). Takes priority over scene_ids/start_idx/num_scenes."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.scene_split is not None:
        from nuscenes import NuScenes

        nusc = NuScenes(version=args.split, dataroot=args.data_root, verbose=False)
        name_to_idx = {s["name"]: i for i, s in enumerate(nusc.scene)}

        if args.scene_split == "eval":
            split_scene_names = set(NUSCENES_EVAL_SCENES)
        else:
            from nuscenes.utils.splits import create_splits_scenes

            scene_splits = create_splits_scenes()
            if args.scene_split not in scene_splits:
                raise ValueError(
                    f"Unknown scene_split: {args.scene_split}. Choose from {['eval'] + list(scene_splits.keys())}."
                )
            split_scene_names = set(scene_splits[args.scene_split])

        scene_ids_list = sorted([name_to_idx[n] for n in split_scene_names if n in name_to_idx])
        missing = sorted(split_scene_names - set(name_to_idx))
        if missing:
            print(f"Warning: {len(missing)} '{args.scene_split}' scene(s) not in {args.split}: {', '.join(missing)}")
        del nusc
        print(f"Processing {len(scene_ids_list)} scenes from split '{args.scene_split}'.")
    elif args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    else:
        scene_ids_list = list(range(args.start_idx, args.start_idx + args.num_scenes))

    scene_ids_list = [int(sid) for sid in scene_ids_list]

    processor = NuScenesProcessor(
        load_dir=args.data_root,
        save_dir=args.target_dir,
        split=args.split,
        process_id_list=scene_ids_list,
        workers=args.workers,
        lidarseg_root=args.lidarseg_root,
    )
    processor.convert()


if __name__ == "__main__":
    main()
