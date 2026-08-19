#!/bin/bash
# FT v2 finish orchestrator: runs every remaining GPU phase in order with
# bounded parallelism and one automatic retry per run (evals resume from
# their last chunk, so a retry never repeats finished work).
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=runs/ft-v2/live.log
ev() { echo "$(date +%H:%M:%S) [finish] $*" >> "$LOG"; }

run_retry() {  # run_retry <logname> <cmd...>
  local name=$1; shift
  local log="runs/ft-v2/logs/$name.log"
  "$@" >>"$log" 2>&1 || { ev "$name failed once, retrying"; "$@" >>"$log" 2>&1; }
  echo "EXIT:$?" >> "$log"
}

CK32=/models/checkpoints/v2-qwen3-32b-s0/step-900
CK14=/models/checkpoints/v2-ministral-14b-s0/final

# ---- Phase A: wait for the running 14B cells to land
ev "phase A: waiting for 14B cells"
while :; do
  n=0
  for a in story literal story-bnear story-bfar literal-bnear literal-bfar; do
    [ -f "runs/ft-v2/ministral-14b-$a/summary.json" ] && n=$((n+1))
  done
  [ "$n" = 6 ] && break
  sleep 60
done
ev "phase A done: all 6 FT-14B cells complete"

# ---- Phase B: 32B FT cells at step-900, 3 at a time
ev "phase B: 32B FT cells (adapter step-900)"
i=0
for a in story literal story-bnear story-bfar literal-bnear literal-bfar; do
  run_retry "eval-ft-32b-$a" \
    modal run eval/modal_eval.py --model qwen3-32b --arm "$a" \
      --adapter "$CK32" --out-tag ft-v2 &
  i=$((i+1)); [ $((i % 3)) = 0 ] && wait
done
wait
ev "phase B done: 32B FT cells complete"

# ---- Phase C: 32B representation + format controls
ev "phase C: 32B capture/probe + format controls"
(
  cd ../probe-experiments || exit 1
  export CAPTURE_GPU=A100-80GB
  L=../ft-experiments/runs/ft-v2/logs
  { modal run capture/modal_capture.py --model qwen3-32b --tag q32b-ftv2 \
      --adapter "$CK32" >>"$L/capture-32b-ftv2.log" 2>&1 \
    || modal run capture/modal_capture.py --model qwen3-32b --tag q32b-ftv2 \
      --adapter "$CK32" >>"$L/capture-32b-ftv2.log" 2>&1; } &&
  { modal run probing/remote_probe.py --tag q32b-ftv2 --out probe-32b-ftv2 \
      >>"$L/probe-32b-ftv2.log" 2>&1 \
    || modal run probing/remote_probe.py --tag q32b-ftv2 --out probe-32b-ftv2 \
      >>"$L/probe-32b-ftv2.log" 2>&1; }
) &
run_retry "fmtctl-32b-base" modal run v2/format_control.py --model qwen3-32b
run_retry "fmtctl-32b-ft" modal run v2/format_control.py --model qwen3-32b --adapter "$CK32"
wait
ev "phase C done: 32B capture/probe + controls complete"

# ---- Phase D: trimmed curves (6 checkpoints x 2 arms), one model at a time
ev "phase D: trimmed checkpoint curves"
for spec in "v2-ministral-14b-s0 ministral-14b" "v2-qwen3-32b-s0 qwen3-32b"; do
  set -- $spec; RUN=$1; MODEL=$2
  for ck in step-0 step-100 step-300 step-500 step-700 step-900; do
    modal volume ls harsh-ft-grammar-weights "checkpoints/$RUN" 2>/dev/null | grep -q "$ck" || continue
    for arm in story story-bfar; do
      run_retry "curve-$RUN-$ck-$arm" \
        modal run eval/modal_eval.py --model "$MODEL" --arm "$arm" --limit 200 \
          --adapter "/models/checkpoints/$RUN/$ck" --out-tag "curve-v2-$RUN-$ck" &
    done
    wait
    ev "curve $RUN $ck done"
  done
done
ev "phase D done: curves complete"

ev "ALL PHASES COMPLETE — ready for analysis"
touch runs/ft-v2/FINISH_DONE
