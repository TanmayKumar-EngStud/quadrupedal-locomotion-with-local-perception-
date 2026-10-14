#!/usr/bin/env bash
set -euo pipefail
# Run in the repository root on Linux with an NVIDIA driver already installed.
command -v nvidia-smi >/dev/null || { echo 'An NVIDIA CUDA instance is required.'; exit 1; }
nvidia-smi
command -v uv >/dev/null || python3 -m pip install 'uv>=0.8,<1'
uv sync --extra cloud --frozen
uv run --extra cloud python scripts/cloud_preflight.py
uv run --extra cloud quad-train --config configs/preflight.json
