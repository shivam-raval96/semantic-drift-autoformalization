#!/bin/bash
# Live tracker for the FT v2 experiment.
#
#   ./watch.sh            overview: current status + live event stream
#   ./watch.sh <stage>    raw log tail of one stage (shows progress bars)
#   ./watch.sh ls         list available stage logs
#
# Stages appear as they start (e.g. train-8b, train-32b, eval-base-bfar-8b, ...).
cd "$(dirname "$0")/../runs/ft-v2" || exit 1
case "$1" in
  ls) ls -t logs/ 2>/dev/null | sed 's/\.log$//'; exit 0 ;;
  "") clear
      cat STATUS.md 2>/dev/null
      echo
      echo "=== live events (Ctrl-C to exit; ./watch.sh ls for stage logs) ==="
      exec tail -n 30 -f live.log ;;
  *)  exec tail -n 40 -f "logs/$1.log" ;;
esac
