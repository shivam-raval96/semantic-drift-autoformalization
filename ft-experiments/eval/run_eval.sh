#!/usr/bin/env bash
# Eval entry point — base or FT-checkpoint, one runner for both.
#
#   bash eval/run_eval.sh 8b story                                  # base
#   bash eval/run_eval.sh 8b two-stage --limit 5                    # smoke
#   bash eval/run_eval.sh 8b story --adapter /models/checkpoints/phase5a-r16-all/final --adapter-rank 16
#
# Base = no --adapter (structural fairness, zero code duplication).
# If --adapter is given and no --out-tag, results go to runs/ft-v1/.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: run_eval.sh MODEL ARM [extra modal args]}"
ARM="${2:?usage: run_eval.sh MODEL ARM [extra modal args]}"
shift 2

EXTRA=("$@")
if [[ " ${EXTRA[*]} " == *" --adapter "* && " ${EXTRA[*]} " != *" --out-tag "* ]]; then
  EXTRA+=(--out-tag ft-v1)
fi

exec modal run modal_eval.py --model "$MODEL" --arm "$ARM" "${EXTRA[@]}"
