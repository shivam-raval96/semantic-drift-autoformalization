#!/usr/bin/env bash
# Install the etp_implication task into a causalab checkout.
# Usage: ./install.sh /path/to/causalab
set -euo pipefail
DEST="${1:?usage: ./install.sh /path/to/causalab}"
[ -d "$DEST/causalab/tasks" ] || { echo "not a causalab checkout: $DEST" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
cp -r "$HERE/tasks/etp_implication" "$DEST/causalab/tasks/"
cp "$HERE/configs/task/etp_implication.yaml" "$DEST/causalab/configs/task/"
mkdir -p "$DEST/causalab/configs/runners/etp_implication"
cp "$HERE/configs/runners/etp_implication/etp_8b_pipeline.yaml" "$DEST/causalab/configs/runners/etp_implication/"
echo "installed etp_implication into $DEST"
echo "smoke test: cd $DEST && python3 -c \"import sys; sys.path.insert(0,'.'); from causalab.tasks.loader import load_task; print(load_task('etp_implication'))\""
