#!/usr/bin/env bash
# Create or refresh the uv-managed environment for this 2DGS tree.
#
#   ./setup_uv.sh
#
# conda is no longer used. See ../UV_MIGRATION.md for the mapping from the
# former environment.yml and for the system-level dependencies (ffmpeg, CUDA).
set -euo pipefail
cd "$(dirname "$0")"

uv python install 3.8
uv sync

# The two CUDA extensions import torch inside setup.py, so they must be built
# against the torch that `uv sync` just installed, not in an isolated build env.
uv pip install --no-build-isolation \
    ./submodules/diff-surfel-rasterization \
    ./submodules/simple-knn

echo "Environment ready. Activate with: source .venv/bin/activate"
