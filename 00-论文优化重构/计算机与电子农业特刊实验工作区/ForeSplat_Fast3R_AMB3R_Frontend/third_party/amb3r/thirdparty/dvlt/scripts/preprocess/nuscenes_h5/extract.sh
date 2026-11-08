#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Extract NuScenes archives into one root in the correct order. The lidarseg
# archive must be extracted LAST: it and the meta archive both ship a
# v1.0-*/category.json, and only the lidarseg one has the class `index` fields
# the devkit needs.
#
# Usage: extract.sh <archives_dir> [dest_root]   (dest defaults to archives_dir)
set -euo pipefail

SRC="${1:?usage: extract.sh <archives_dir> [dest_root]}"
DEST="${2:-$SRC}"
mkdir -p "$DEST"

extract() {  # $1 = tar flags, $2 = glob
    for f in $2; do
        [ -e "$f" ] || continue
        echo "Extracting $(basename "$f")..."
        tar "$1" "$f" -C "$DEST"
    done
}

# meta + blobs (order between these does not matter), lidarseg LAST.
extract xzf "$SRC/v1.0-*_meta.tgz"
extract xzf "$SRC/v1.0-mini.tgz"
extract xzf "$SRC/v1.0-*_blobs.tgz"
extract xjf "$SRC/nuScenes-lidarseg-*.tar.bz2"

echo "Done -> $DEST"
