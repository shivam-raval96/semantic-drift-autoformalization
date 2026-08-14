#!/usr/bin/env bash
# Launch the verification gate for a model key from config.VERIFY, on the
# GPU class the registry assigns it. Usage: bash run_verify.sh <key> [args...]
set -euo pipefail
cd "$(dirname "$0")/.."
KEY="$1"; shift || true
GPU=$(.venv/bin/python -c "import config; print(config.VERIFY['models']['$KEY']['gpu'])")
VERIFY_GPU="$GPU" modal run behavior/modal_verify.py --model "$KEY" "$@"
