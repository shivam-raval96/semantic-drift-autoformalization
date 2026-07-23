# 11 — Majority voting: self-consistency for weak open-weight models

Run 2026-07-22.

## Question

The open-weight models sit at 21–37% correct on the story arm of
experiment 09's balanced synthetic pairs (temperature 0, no-think):
deepseek-chat-v3.1 36.5%, llama-3.3-70b 25.5%, mistral-small-3.2-24b
21.5%, qwen3-32b 21.0%. Does K-sample self-consistency — formalize the
same story K times at temperature 0.7 and keep the most popular
answer — recover accuracy these models lose in a single pass, and how
does the recovery scale in K (vote@1 → vote@3 → vote@5 → vote@7)?
Because the harness grades syntactically, "most popular" is exact and
LLM-free here: answers are pooled by their grading-equivalence class
(`checkform.answer_class_key` — the orbit under the eight accepted
transforms), so a law's swap/dual variants vote together, exactly as
`grade()` treats them. Voting on raw canonical strings instead would
split one law's votes across variants; small models are swap-heavy
(llama: 17 exact vs 33 correct-swapped on this pair set), so the class
pooling is load-bearing.

## Setup

Same pipeline, prompts, grading, pair set, and regime machinery as
experiment 09's story arm; the changed variables are the temperature
(0.7, via the new `benchmark.py --temperature`), the K = 7 repeated
passes, and the post-hoc voting reduction (`voteform.py`, new).

- **Equations**: experiment 09's `data/synthetic-equations.txt`
  (sha256 `7f11f36c…`), 364 synthetic laws in per-equation bins 1–10.
- **Models** (4, via OpenRouter — the open-weight members of the
  default set):
  - `deepseek/deepseek-chat-v3.1`
  - `qwen/qwen3-32b`
  - `meta-llama/llama-3.3-70b-instruct`
  - `mistralai/mistral-small-3.2-24b-instruct`
- **Form**: `--form story` only — the hardest arm for these models,
  and the arm where experiment 09 left the most headroom.
- **Thinking mode**: `--reasoning off`, as in experiments 07–09.
- **Sampling** (seed 0): the byte-identical 200 balanced pairs of
  experiment 09 (`--stratify-eq-ops 20 --eq-bins 1:10`,
  `--label-prefix S`; verified: `samples.jsonl` of every pass diffs
  empty against experiment 09's story run). Vacuous-free by
  construction.
- **Passes**: 7 independent runs of the identical command into
  `runs/run-eqstrat20-s0-story-t0.7-pass{1..7}`, all at temperature
  0.7. Each pass is a normal resumable benchmark run; OpenRouter may
  route different passes (or rows) to different providers — rows
  record `routed_model`/`provider`, and at nonzero temperature this
  provider variance is folded into the sampled variance being
  measured.
- **Voting** (`voteform.py`): for each (pair, model), passes whose
  answer parses cast a ballot — their grading-equivalence class;
  unparseable and api-error passes abstain. vote@k reduces the prefix
  passes 1..k: most votes wins, ties go to the class first seen in
  the earliest pass, and an all-abstain cell falls back to the
  earliest graded row (unparseable over api-error). The emitted row
  is the earliest pass's row from the winning class, verbatim, plus a
  `vote` provenance field; it keeps that pass's own verdict/bucket —
  correctness is class-invariant, so correct rates are unaffected,
  but the exact/swapped split in vote-dir compositions reflects the
  representative passes. vote@1 is pass 1 relabeled.
- **Baselines**: experiment 09's temp-0 story run, copied restricted
  to the 4 models as `runs/…-t0-baseline` (labeled "temp0
  single-pass"); vote@1 doubles as the temp-0.7 single-pass baseline,
  so the report separates the cost of raising temperature from the
  gain of voting.
- **Other**: `max_tokens` 4096, `--concurrency 8`. 200 pairs × 4
  models × 7 passes = 5,600 calls.

## Prompts

Story arm template byte-identical to experiments 02–09:
`formalize_prompt.md` at sha256
`ad33f6de859156b81be0d889abd3c56e4d9275bd855eb6d804d4e8ebcfe4983c`.
Regime wrappers identical to experiment 07 (off regime; qwen3 gets
`/no_think` on every call). Nothing in any prompt mentions repeated
attempts or voting — every pass is oblivious to the others.

## Reproduce

```sh
M4=deepseek/deepseek-chat-v3.1,qwen/qwen3-32b,meta-llama/llama-3.3-70b-instruct,mistralai/mistral-small-3.2-24b-instruct
EQ=experiments/09-synthetic-complexity/data/synthetic-equations.txt

for P in 1 2 3 4 5 6 7; do
  python3 benchmark.py --seed 0 --stratify-eq-ops 20 --eq-bins 1:10 \
      --equations-path $EQ --label-prefix S --form story --reasoning off \
      --temperature 0.7 --concurrency 8 --models "$M4" \
      --out-dir experiments/11-majority-voting/runs/run-eqstrat20-s0-story-t0.7-pass$P
done

python3 voteform.py \
    experiments/11-majority-voting/runs/run-eqstrat20-s0-story-t0.7-pass{1,2,3,4,5,6,7} \
    --ks 1,3,5,7 --out-root experiments/11-majority-voting/runs \
    --baseline experiments/09-synthetic-complexity/runs/run-eqstrat20-s0-story-think-off

python3 charts.py \
    experiments/11-majority-voting/runs/run-eqstrat20-s0-story-t0.7-t0-baseline \
    experiments/11-majority-voting/runs/run-eqstrat20-s0-story-t0.7-vote1 \
    experiments/11-majority-voting/runs/run-eqstrat20-s0-story-t0.7-vote3 \
    experiments/11-majority-voting/runs/run-eqstrat20-s0-story-t0.7-vote5 \
    experiments/11-majority-voting/runs/run-eqstrat20-s0-story-t0.7-vote7 \
    --title "ETP story formalization · majority voting at temperature 0.7" \
    --out experiments/11-majority-voting/report/report.html --pdf
```

## Results

Artifacts: the seven pass directories and the five voteform-derived
directories under `runs/` (each with `summary.md`), and
`report/report.html` / `.pdf` (correct rate by model and condition,
accuracy-vs-complexity with one line per condition, verdict
composition per condition — all on the identical pair set).

All seven passes: 800/800 rows graded, 0 api-errors (one pass needed
2 retried calls), compliance clean. OpenRouter-reported cost for all
5,600 calls: $1.06 (7.29M tokens). Temperature verified live: pass
bucket compositions differ pass to pass (e.g. exact 104–120).

Correct% over the 200 balanced pairs:

| model | temp0 | vote@1 | vote@3 | vote@5 | vote@7 |
|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 36.5 | 34.0 | 37.0 | 37.5 | 37.0 |
| meta-llama/llama-3.3-70b-instruct | 25.5 | 25.5 | 26.0 | 27.0 | 27.5 |
| mistralai/mistral-small-3.2-24b-instruct | 21.5 | 22.0 | 23.5 | 22.5 | 23.0 |
| qwen/qwen3-32b | 21.0 | 20.5 | 22.5 | 22.5 | 23.5 |
| **pooled** | 26.1 | 25.5 | 27.2 | 27.4 | 27.8 |

Cell-level flips, temp-0 baseline → vote@7: 22 gained, 9 lost (net
+13 of 800; qwen +9/−4, mistral +5/−2, deepseek +4/−3, llama +4/−0).

Pooled correct% by per-equation ops bin m (both laws in bin m):

| bin | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| temp0 | 91 | 84 | 56 | 18 | 10 | 0 | 1 | 1 | 0 | 0 |
| vote@7 | 94 | 84 | 66 | 20 | 9 | 1 | 1 | 2 | 0 | 0 |

Vote diagnostics at k = 7 (per (pair, model) cell):

- **Consensus is bimodal.** 159/800 cells are unanimous (7/7 ballots
  in one class); 239 cells have a 1-vote plurality winner (every
  parsed pass answered differently); 93 cells had all seven passes
  unparseable (the only unparseables that survive voting). 222 cells
  needed the earliest-pass tie-break.
- **Agreement predicts correctness almost perfectly.** Correct% of
  the winning class by votes-for-winner: 1/7 → 0.4, 2/7 → 7.1,
  3/7 → 9.9, 4/7 → 19.6, 5/7 → 50.0, 6/7 → 75.8, 7/7 → 93.7.
  Used as a confidence gate: answering only at ≥5/7 consensus gives
  82.8% precision on 30% coverage; ≥7/7 gives 93.7% on 20% —
  against 27% overall.
- **Voting fixes parse failures, not misreadings.** Unparseable
  drops 268 (temp0) → 93 (vote@7) because a single parseable pass
  supplies an answer — but the recovered answers are almost all
  wrong (wrong grows 323 → 485), so the parse-rate recovery barely
  moves the correct rate.

## Conclusions

- **Majority voting helps, but the uplift is small: +2.3 points
  pooled over single-pass at the same temperature (25.5 → 27.8), and
  monotone in K for every model.** This is far from the large
  self-consistency gains reported on chain-of-thought math
  benchmarks: these formalization errors are not sampling noise
  around a correct modal answer. Raising temperature to 0.7 itself
  costs ~0.6 points (vote@1 25.5 vs temp0 26.1); K = 3 already
  recoups it, so the net of "vote@3 vs the deterministic run you
  would have done anyway" is about +1 point at 3× cost.
- **Models are either solid or lost, with a thin flippable middle.**
  The consensus distribution's two modes — unanimous (94% correct)
  and all-answers-distinct (~0%) — mean voting only has leverage
  where a model is partially reliable. That zone is the complexity
  shoulder: the entire pooled gain concentrates in bin 3
  (56 → 66%). In bins 5–10, where single-pass accuracy is ~0–10%,
  seven samples find nothing to amplify — consistent with experiment
  09's finding that deep pairs fail by output-grammar collapse, not
  near-miss misreading.
- **The real value of self-consistency here is calibration, not
  accuracy.** Vote agreement is a steep, monotone, nearly-free
  confidence signal (0.4% correct at 1/7 → 93.7% at 7/7). A
  downstream autoformalization pipeline should use K-sample
  agreement to decide *when to trust* a weak model (or when to
  escalate to a stronger one), rather than to squeeze points out of
  it: ≥5/7 consensus turns a 21–37%-accurate model into an
  ~83%-precision oracle over the 30% of inputs it can actually do.
- Methodological: pooling votes by grading-equivalence class
  mattered mechanically (222 tie-broken cells and every
  swap-variant merge would otherwise fragment; `correct-swapped` is
  ~44% of these models' correct rows), and `voteform.py`'s
  synthetic run dirs let charts.py compare the K-ladder with zero
  changes to the report pipeline. Both are reusable for any future
  multi-sample experiment (best-of-K with an oracle, verifier
  reranking, cross-model ensembles).
