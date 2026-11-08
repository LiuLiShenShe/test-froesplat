#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the ETH3D high-resolution multi-view DSLR data needed for evaluation:
#   - undistorted DSLR images + calibration files
#   - distorted (THIN_PRISM_FISHEYE) calibration, needed to undistort GT depth
#   - per-scene ground-truth depth
#
# Usage:
#   bash src/dvlt/scripts/preprocess/eth3d/download.sh <dest_dir>
#   e.g. bash src/dvlt/scripts/preprocess/eth3d/download.sh ${user.data_root}/test/eth3d
#
# Requires: wget, and 7zip/p7zip (the `7z` command).
# After downloading, undistort the depth maps with:
#   python -m dvlt.scripts.preprocess.eth3d.undistort_depth --root <dest_dir>
set -euo pipefail

DEST_DIR="${1:?Usage: download.sh <dest_dir>}"

SEVENZIP="$(command -v 7z || command -v 7za || command -v 7zz || true)"
if [ -z "${SEVENZIP}" ]; then
    echo "Error: 7z not found. Install p7zip, e.g.:" >&2
    echo "  conda install -c conda-forge p7zip   # or" >&2
    echo "  sudo apt-get install p7zip-full" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

# Archives to fully extract: undistorted images + calibration, then per-scene GT depth.
archives=("multi_view_training_dslr_undistorted.7z")
scenes=("courtyard" "delivery_area" "electro" "facade" "kicker" "meadow" "office" "pipes" "playground" "relief" "relief_2" "terrace" "terrains")
for scene in "${scenes[@]}"; do
    archives+=("${scene}_dslr_depth.7z")
done

# Download (resumable) and extract. Archives are kept until all have been
# extracted, so an interrupted run can resume without re-downloading.
for a in "${archives[@]}"; do
    wget -c "https://www.eth3d.net/data/${a}"
    "${SEVENZIP}" x "${a}" -aoa -bsp1
done

# The distorted-jpg archive is large (~4.7 GB) but only its tiny
# dslr_calibration_jpg/{cameras.txt,images.txt} files are needed (the
# THIN_PRISM_FISHEYE params for undistorting the GT depth). Download it and
# extract only those calibration folders, not the full-res distorted images.
jpg_archive="multi_view_training_dslr_jpg.7z"
wget -c "https://www.eth3d.net/data/${jpg_archive}"
"${SEVENZIP}" x "${jpg_archive}" -aoa -bsp1 "*/dslr_calibration_jpg/*"

# All archives extracted successfully: clean them up.
rm -f "${archives[@]}" "${jpg_archive}"

echo "ETH3D download complete in ${DEST_DIR}"
echo "Next: python -m dvlt.scripts.preprocess.eth3d.undistort_depth --root ${DEST_DIR}"
