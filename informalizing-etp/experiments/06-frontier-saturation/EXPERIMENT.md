# 06 -- GPT-5.5 frontier saturation check

This experiment tests GPT-5.5 on the same 40 stratified ETP pairs: five
pairs at each total operation count from 1 to 8.

## Results

| Condition | Exact | Correct-swapped | Wrong | Correct |
|---|---:|---:|---:|---:|
| Story -> R.G., reasoning on | 39 | 1 | 0 | 40/40 (100%) |
| Story -> R.G., reasoning off | 30 | 7 | 3 | 37/40 (92.5%) |
| Literal -> R.G., reasoning off | 40 | 0 | 0 | 40/40 (100%) |
| Story -> Literal -> R.G., reasoning off | 40 | 0 | 0 | 40/40 (100%) |

`correct-swapped` is semantically correct: it only exchanges the two sides
of an equality.

**Vacuous-law-excluded view** (added 2026-07-22). Experiment 07
identified pairs containing a vacuous law (E1 `x = x` / E2 `x = y`) as
a measurement hazard, and the convention is now to exclude them
(`experiments/README.md`). The stratified runs keep 24 of 40 pairs
(`python3 filter_vacuous.py experiments/06-frontier-saturation`).
Filtered gpt-5.5 scores: story/off 21/24 (87.5, was 37/40 = 92.5) —
all three original misses survive the filter, so the one non-saturated
cell looks slightly *worse* without its easy vacuous pairs; story/on,
literal/off, and two-stage/off remain perfect at 24/24. The saturation
reading is unchanged.

## Conclusion

The current task is largely saturated for GPT-5.5. Only direct Story -> R.G.
without reasoning produced errors, while the other three conditions reached
100%. Further runs with the same frontier model and setup are unlikely to
add much.

The raw artifacts for all four conditions are in `runs/`.
