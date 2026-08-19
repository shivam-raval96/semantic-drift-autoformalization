#!/bin/bash
# Live experiment dashboard — refreshes every 15s. Ctrl-C to exit.
cd "$(dirname "$0")/.." || exit 1
while true; do
  clear
  echo "=== FT v2 LIVE — $(date +%H:%M:%S) ==="
  echo
  s=$(grep -c "'loss'" runs/ft-v2/logs/train-v2-32b-r3.log 2>/dev/null)
  if grep -q "EXIT:" runs/ft-v2/logs/train-v2-32b-r3.log 2>/dev/null; then
    echo "32B training : FINISHED"
  else
    echo "32B training : step ~$s / 1041"
  fi
  echo
  echo "FT-14B eval cells (of 777):"
  for a in story literal story-bnear story-bfar literal-bnear literal-bfar; do
    if [ -f "runs/ft-v2/ministral-14b-$a/summary.json" ]; then
      printf "  %-14s DONE\n" "$a"
    else
      printf "  %-14s %s\n" "$a" "$(wc -l < runs/ft-v2/ministral-14b-$a/results.jsonl 2>/dev/null || echo 0)"
    fi
  done
  echo
  echo "FT-32B eval cells:"
  found=0
  for a in story literal story-bnear story-bfar literal-bnear literal-bfar; do
    if [ -d "runs/ft-v2/qwen3-32b-$a" ]; then
      found=1
      if [ -f "runs/ft-v2/qwen3-32b-$a/summary.json" ]; then
        printf "  %-14s DONE\n" "$a"
      else
        printf "  %-14s %s\n" "$a" "$(wc -l < runs/ft-v2/qwen3-32b-$a/results.jsonl 2>/dev/null || echo 0)"
      fi
    fi
  done
  [ "$found" = "0" ] && echo "  (launch after 32B training lands)"
  echo
  echo "--- last events ---"
  tail -5 runs/ft-v2/live.log
  sleep 15
done
