#!/usr/bin/env bash
# One-time (idempotent) local setup: venv + deps + modal auth check.
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null 2>&1 || {
  echo "uv not found — install it first: https://docs.astral.sh/uv/"; exit 1; }

[ -d .venv ] || uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -q -r requirements.txt

modal profile current >/dev/null 2>&1 || {
  echo "Modal is not authenticated — run: modal setup"; exit 1; }

python -c "import config; print('config OK:', config.ROOT)"
echo "Done"
