#!/usr/bin/env bash
# Training entry point — presets or explicit sweep points, seeds loopable.
#
#   bash training/run_train.sh 8b --preset smoke                 # Phase 4 smoke
#   bash training/run_train.sh 8b --preset phase5a               # Phase 5a headline
#   bash training/run_train.sh 8b --rank 1 --layer 16 --seed 0   # one sweep point
#   SEEDS="0 1 2" bash training/run_train.sh 8b --rank 1 --layer 16   # seed loop
#
# Only the 8B is a training target today; the model arg is kept for
# future expansion. Config resolution + stamping happens in
# train_lora.py; checkpoints go to the Modal Volume under
# /models/checkpoints/<run_name>/. The old one-shot Phase-5a chain
# (train -> three evals -> comparison) is documented as separate
# commands in training/README.md and eval/README.md.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: run_train.sh MODEL [--preset P | --rank N --layer K --seed S ...]}"
shift
[ "$MODEL" = "8b" ] || { echo "only 8b is a training target (got: $MODEL)"; exit 1; }

if [ -n "${SEEDS:-}" ]; then
  for S in $SEEDS; do
    echo "=== seed $S ==="
    modal run train_lora.py "$@" --seed "$S"
  done
else
  exec modal run train_lora.py "$@"
fi
