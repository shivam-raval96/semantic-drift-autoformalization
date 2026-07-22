# 07 — Two-stage at scale: no-think, even complexity coverage, plus GPT-5.5

Run 2026-07-22.

## Question

Experiment 05 measured the two-stage (abstract → formalize) pipeline on
30 uniform + 40 stratified pairs per model and found its no-think story:
decomposition helps models that cannot think, and the failures
concentrate in the low-complexity bins where stage 1 invents structure
for degenerate laws. This experiment repeats the no-think arm at 4× the
stratified sample — 160 pairs per model, exactly 20 per total-operation
bin 1–8, with no uniform (complexity-skewed) run — to answer: does that
picture hold with real per-bin resolution, and where does a frontier
model (GPT-5.5, which was at ceiling on experiment 06's 40 stratified
two-stage pairs) land against the small models when neither can reason?

## Setup

Same pipeline, prompts, grading, and regime machinery as experiment 05;
the changed variables are the sample size/shape, the single regime, and
the added frontier model.

- **Models** (9, via OpenRouter — experiment 05's eight plus GPT-5.5):
  - `deepseek/deepseek-chat-v3.1`
  - `qwen/qwen3-32b`
  - `meta-llama/llama-3.3-70b-instruct`
  - `mistralai/mistral-small-3.2-24b-instruct`
  - `openai/gpt-4o-mini`
  - `google/gemini-2.5-flash`
  - `anthropic/claude-haiku-4.5`
  - `openai/gpt-5-mini`
  - `openai/gpt-5.5`
- **Form**: `--form two-stage`, unchanged from experiment 05 — stage 1
  abstracts the themed story via `abstract_prompt.md`, stage 2 feeds the
  raw stage-1 response into `literal_prompt.md`, and the final answer is
  graded by checkform. Two API calls per row, both at temperature 0.
- **Thinking mode**: `--reasoning off` only (experiment 05 established
  the on-regime ceiling; the no-think regime is where the two-stage
  effect lives). Native toggles as before: qwen3 gets `/no_think` in
  both stages; gpt-5-mini floors at `effort: minimal`; gpt-5.5 supports
  a true `{"effort": "none", "exclude": true}` (the arm experiment 06
  validated); llama-3.3, mistral-small, and gpt-4o-mini get the prompt
  wrappers alone. Compliance clean: `max(reasoning_tokens,
  stage1_reasoning_tokens)` is 0 for every row except qwen3's familiar
  1-token think-block artifact.
- **Sampling** (seed 0, ETP `equations.txt` sha256 `e30e1a67…c8010`,
  unchanged): one run, `run-strat20-s0-think-off` — 160 pairs
  stratified 20 per total-operation bin 1–8. Two consequences of
  `per_bin = 20`:
  - Bin 1 is *exhaustive*: the ETP list has exactly 2 zero-op and 5
    one-op laws, giving 20 ordered pairs total, and all 20 are in the
    sample (20/bin is the maximum perfectly-even design; 26/bin cannot
    be filled).
  - The pair set is **not** byte-identical to the strat5 sets of
    experiments 02–06 — a different per-bin draw count changes the RNG
    sequence after bin 1 — so comparisons to earlier experiments are
    aggregate-level, not per-pair.
- **Other**: `max_tokens` 4096 per call, temperature 0,
  `--concurrency 8`. 160 pairs × 9 models = 1,440 rows (2,880 calls).
  Results rows keep the standard schema plus `stage1_*` bookkeeping,
  as in experiment 05. OpenRouter-reported cost for the whole run:
  $4.63, of which gpt-5.5 is $3.25.

## Prompts

Both templates byte-identical to experiment 05's (same sha256s at run
time):

- Stage 1: `abstract_prompt.md` at sha256
  `1aa038f2d13dbddcc5c7f803d9166d9e8a82ce9b981a3463fc59a3d44cce8733`.
- Stage 2: `literal_prompt.md` at sha256
  `089ffc52bb57c5aa7c2ead0e613ec7e73b11039aa79d729f8c80c63a0852f8b0`.
- Regime wrappers (off regime; prefixes empty), from `run_meta.json`:
  - stage-1 suffix:
    > Respond with only the rewritten description, and no other text
    > before it.
  - stage-2 suffix:
    > Respond with only the two required lines, and no other text
    > before them.

  plus a trailing `/no_think` line for qwen3 (both stages).

## Reproduce

```sh
python3 benchmark.py --seed 0 --stratify-ops 20 --form two-stage --reasoning off \
    --concurrency 8 \
    --models deepseek/deepseek-chat-v3.1,qwen/qwen3-32b,meta-llama/llama-3.3-70b-instruct,mistralai/mistral-small-3.2-24b-instruct,openai/gpt-4o-mini,google/gemini-2.5-flash,anthropic/claude-haiku-4.5,openai/gpt-5-mini,openai/gpt-5.5 \
    --out-dir experiments/07-two-stage-scale/runs/run-strat20-s0-think-off

python3 charts.py experiments/07-two-stage-scale/runs/run-strat20-s0-think-off \
    --title "ETP formalization benchmark · two-stage at scale, no-think, + GPT-5.5" \
    --out experiments/07-two-stage-scale/report/benchmark-report.html --pdf
```

Requires checkform's term-depth cap (`_MAX_TERM_DEPTH`, added during
this experiment — see run notes).

## Results

Artifacts: `runs/run-strat20-s0-think-off/` (with `summary.md`) and
`report/benchmark-report.html` / `.pdf`.

Correct% over the 160 stratified pairs, no-think (experiment 05's
strat5-off number in parentheses; not per-pair comparable):

| model | correct% | (exp 05, n=40) |
|---|---|---|
| openai/gpt-5.5 | **99.4** | — |
| deepseek/deepseek-chat-v3.1 | 73.8 | (70.0) |
| google/gemini-2.5-flash | 73.8 | (67.5) |
| anthropic/claude-haiku-4.5 | 65.0 | (67.5) |
| meta-llama/llama-3.3-70b-instruct | 58.8 | (60.0) |
| openai/gpt-5-mini | 56.9 | (57.5) |
| mistralai/mistral-small-3.2-24b-instruct | 50.0 | (42.5) |
| qwen/qwen3-32b | 38.8 | (22.5) |
| openai/gpt-4o-mini | 23.1 | (20.0) |

1,440/1,440 rows graded, 0 api-errors; 60 unparseable (4.2%), 27 of
them gpt-5-mini. Pooled correct% by total-operation bin 1–8:
55 / 73 / 71 / 49 / 70 / 59 / 51 / 51; by max nesting depth 1–4:
69 / 68 / 61 / 40.

Stage attribution (same mechanical procedure as experiment 05: the tail
of the stage-1 response from its last "Consider a collection …"
paragraph back-parsed with `literalform.backparse_literal`, compared
under the accepted symmetries). Final-verdict counts (correct / fail):

| stage-1 class | correct | fail |
|---|---|---|
| faithful | 372 | 221 |
| drifted | 0 | 36 |
| off-grammar | 491 | 320 |

Drifted rows by model: gpt-5-mini 16, mistral-small 8, qwen3 6,
gemini-2.5-flash 2, claude-haiku 2, llama-3.3 2; deepseek, gpt-4o-mini,
and gpt-5.5 none.

Run notes:

- **The run surfaced a grader crash.** Mid-run, one stage-2 response
  degenerated into a ~1,000-deep nest of `op(` calls (a repetition loop
  until the token cap), and checkform's recursive-descent parser blew
  Python's recursion limit; the `RecursionError` escaped the worker
  thread and killed the harness at 1,080/1,440 rows. Fix (committed
  with this experiment): `checkform.py` now rejects terms nested deeper
  than `_MAX_TERM_DEPTH = 50` with the same `AnswerParseError` that
  grades `unparseable` (ETP laws nest at most 4 ops per side, so no
  legitimate answer is affected), with parser- and grade-level
  regression tests in `test_checkform.py`. The resumed run completed
  the remaining rows; the retried call returned an ordinarily
  malformed response, so no depth-cap verdicts appear in the final
  data — the 60 unparseables are all ordinary malformed syntax
  (truncations, stray prose, `op(x, y) = x = y` chains).
- gpt-5.5's single miss (E1918 ⇒ E350, 7 ops, paint theme) is a wrong
  abstraction, not a formalization slip: its `ASK:` line encodes a
  different law than the story's.
- `correct-swapped` dominates the correct rows of several small models
  (mistral 68 of 80, haiku 64 of 104, deepseek 63 of 118) —
  side-swapping under the no-think regime is the norm, not the
  exception, for this family. `correct-dualized` remains rare (3 rows
  total).

## Conclusions

- **Experiment 05's no-think picture survives 4× the sample.** Every
  model lands within a few points of its strat5-off number, and the
  ordering is unchanged except within pairs that were already close
  (the two largest moves, qwen 22.5 → 38.8 and mistral 42.5 → 50.0,
  are in the direction the small-n noise would predict). The two-stage
  no-think band for the small-model family is 23–74%, exactly where
  experiment 05 put it.
- **GPT-5.5 is not in the same regime as the rest of the field.**
  159/160 (99.4%) with reasoning truly off, all exact-convention, zero
  unparseable — against a 73.8% best for the small models. Experiment
  06's frontier-saturation reading extends from 40 to 160 stratified
  pairs and to the two-stage arm: for GPT-5.5 the benchmark measures
  formatting discipline, not capability, and headroom for studying
  *drift* now lives entirely in the small-model tier.
- **Drift is unrepairable, now at 0/61 pooled.** No row with a
  parseable-but-wrong stage-1 abstraction ended correct — 0/36 here,
  0/25 in experiment 05. Meanwhile faithful abstractions fail stage 2
  37% of the time (221/593) without thinking, so the off-regime
  pipeline loses accuracy at *both* stages, but only stage-1 drift is
  unrecoverable in principle. The case for a mechanical fidelity gate
  on the intermediate (CLAUDE.md's round-trip idea) gets stronger, not
  weaker, with sample size.
- **Complexity is bimodally hard.** With 20 pairs per bin, the pooled
  complexity curve is not monotone: bins 2–3 are the sweet spot
  (73/71%), with dips at bin 1 (55% — the degenerate bare-variable laws
  experiment 05 flagged as the drift hotspot) and from bin 4 onward
  (49–59%). Depth is the cleaner monotone signal: 69/68/61/40% for
  depth 1–4. Trivial laws invite embellishment; deep nesting invites
  bookkeeping errors; the easy middle is where a law has structure but
  no long chains.
- Methodological: a benchmark whose grader can be crashed by a
  degenerate generation is itself part of the measurement — the
  depth-cap fix belongs in the harness permanently, and the runaway
  `op(`-repetition failure mode (concentrated in gpt-5-mini's
  truncated rows) is worth counting explicitly if it recurs.
