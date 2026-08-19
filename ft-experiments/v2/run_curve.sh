#!/bin/bash
# Checkpoint-curve evals: every saved checkpoint of one run, two arms
# (trained grammar A story + never-trained B-far story), limit-200 each.
# Sequential per checkpoint, the two arms in parallel — bounded load.
# Usage: ./run_curve.sh v2-8b-s0 8b [curve-tag]
set -u
RUN=$1; MODEL=$2; TAG=${3:-curve-v2}
cd "$(dirname "$0")/.."
CKPTS=$(modal volume ls harsh-ft-grammar-weights "checkpoints/$RUN" 2>/dev/null \
        | grep -oE "step-[0-9]+" | sort -t- -k2 -n)
for ck in $CKPTS; do
  for arm in story story-bfar; do
    log="runs/ft-v2/logs/curve-$RUN-$ck-$arm.log"
    (modal run eval/modal_eval.py --model "$MODEL" --arm "$arm" --limit 200 \
       --adapter "/models/checkpoints/$RUN/$ck" \
       --out-tag "$TAG-$RUN-$ck" >"$log" 2>&1; echo "EXIT:$?" >>"$log") &
  done
  wait
  echo "$(date +%H:%M:%S) [curve] $RUN $ck done (story + story-bfar)" >> runs/ft-v2/live.log
done
echo "$(date +%H:%M:%S) [curve] $RUN COMPLETE" >> runs/ft-v2/live.log
