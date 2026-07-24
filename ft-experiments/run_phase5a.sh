#!/usr/bin/env bash
# Phase 5a launcher — run it yourself:  bash ft-experiments/run_phase5a.sh
#
# Chain (sequential, all output live in this terminal and tee'd to logs):
#   1. Train grammar-LoRA on Modal A10G — Llama-3.1-8B bf16, r=16, alpha=16,
#      all layers (q,k,v,o,gate,up,down), 3 epochs, batch 4, seq 1024 packed,
#      lr 2e-4 cosine, seed 0. Checkpoints: step 0 + every 10 steps + final,
#      PEFT format, on the shared Volume. Config lives ONLY in
#      modal_train.py (mode=full) and is stamped into
#      runs/ft-v1/train-phase5a-r16-all.json together with the loss curve
#      and both sanity probes.
#   2. Eval the final adapter on all three arms (story / literal /
#      two-stage), frozen protocol, same A10G eval config as the base runs.
#   3. Print the base-vs-FT comparison table (runs/ft-v1/comparison.md).
#
# Any non-zero exit stops the chain immediately.
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="runs/ft-v1/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)

banner() {
  printf '\n============================================================\n'
  printf '  %s\n' "$1"
  printf '============================================================\n'
}

banner "1/3  PHASE 5a TRAINING — Llama-3.1-8B · LoRA r=16 · all layers · 3 epochs"
modal run modal_train.py --mode full 2>&1 | tee "$LOG_DIR/train-$STAMP.log"

ADAPTER=/models/checkpoints/phase5a-r16-all/final
for ARM in story literal two-stage; do
  banner "2/3  FT EVAL — $ARM arm · 777 problems · adapter $ADAPTER"
  modal run modal_base_eval.py --model 8b --arm "$ARM" \
      --adapter "$ADAPTER" --adapter-rank 16 --out-tag ft-v1 \
      2>&1 | tee "$LOG_DIR/eval-8b-$ARM-$STAMP.log"
done

banner "3/3  BASE vs FT COMPARISON"
python3 compare_table.py

echo "Done. Artifacts: runs/ft-v1/ (train JSON + three run dirs + comparison.md); logs in $LOG_DIR"
