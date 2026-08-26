#!/usr/bin/env bash
# Launch activation capture for a model key from config.MODELS, on the GPU
# class the registry assigns it. Usage: bash run_capture.sh <key> [args...]
set -euo pipefail
cd "$(dirname "$0")/.."
KEY="$1"; shift || true
GPU=$(.venv/bin/python -c "import config; print(config.MODELS['$KEY']['gpu'])")
CAPTURE_GPU="$GPU" modal run capture/modal_capture.py --model "$KEY" "$@"
