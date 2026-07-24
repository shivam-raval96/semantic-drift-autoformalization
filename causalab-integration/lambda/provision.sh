#!/usr/bin/env bash
# Provision a fresh Lambda instance for the etp_implication 8B runs.
# Idempotent; run as the default user on Ubuntu-based Lambda images.
#
# Usage:
#   export HF_TOKEN=hf_...          # required (gated Llama; license accepted for 3.1)
#   ./provision.sh                  # expects this repo alongside, see below
#
# Source of the task: either (a) the semantic-drift-autoformalization repo
# is cloned next to this script's parent (branch etp-causalab), or (b) a
# bundle tarball made by make_bundle.sh was scp'd and unpacked here.
set -euo pipefail

CAUSALAB_REF="e433ccef06f638917da97b0d5316b4dae641ff90"   # pinned commit
CAUSALAB_DIR="$HOME/causalab"

[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set" >&2; exit 1; }

# 1. causalab at the pinned commit
if [ ! -d "$CAUSALAB_DIR/.git" ]; then
  git clone https://github.com/goodfire-ai/causalab "$CAUSALAB_DIR"
fi
git -C "$CAUSALAB_DIR" fetch --quiet
git -C "$CAUSALAB_DIR" checkout --quiet "$CAUSALAB_REF"
echo "causalab @ $(git -C "$CAUSALAB_DIR" rev-parse --short HEAD) (pinned)"

# 2. python env (Lambda images ship CUDA torch; venv keeps it clean)
if [ ! -x "$CAUSALAB_DIR/.venv/bin/python" ]; then
  python3 -m venv "$CAUSALAB_DIR/.venv"
fi
"$CAUSALAB_DIR/.venv/bin/pip" install -q -e "$CAUSALAB_DIR"
"$CAUSALAB_DIR/.venv/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available - wrong instance or torch build"
print("torch", torch.__version__, "| cuda ok:", torch.cuda.get_device_name(0))
PY

# 3. install the task (repo checkout or unpacked bundle, whichever exists)
HERE="$(cd "$(dirname "$0")" && pwd)"
INTEGRATION="$(dirname "$HERE")"          # .../causalab-integration
[ -d "$INTEGRATION/tasks/etp_implication" ] || { echo "causalab-integration not found next to lambda/" >&2; exit 1; }
"$INTEGRATION/install.sh" "$CAUSALAB_DIR"

# 4. verify the task loads and the data hash is the expected v5
"$CAUSALAB_DIR/.venv/bin/python" - <<PY
import hashlib, sys
sys.path.insert(0, "$CAUSALAB_DIR")
from causalab.tasks.loader import load_task
t = load_task("etp_implication")
blob = open("$CAUSALAB_DIR/causalab/tasks/etp_implication/data/etp_pairs.json", "rb").read()
h = hashlib.sha256(blob).hexdigest()[:16]
print("task loads OK | data sha", h)
assert h == "5e7f69b0e0e27c25", f"data hash mismatch: {h} (expected v5 5e7f69b0e0e27c25)"
PY

echo "PROVISIONED. Next: ./run_8b.sh"
