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

Extended after the two-stage run: the same 160 pairs were also run
through both single-shot arms — story → grammar (experiment 02's task)
and literal → grammar (experiment 04's task) — so that experiment 05's
no-think ordering claim (literal ≥ two-stage > story) is tested at
scale on one byte-identical pair set instead of across experiments.

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
- **Forms** (three runs over the byte-identical pair set — sampling
  never consults the form):
  - `--form two-stage`, unchanged from experiment 05 — stage 1
    abstracts the themed story via `abstract_prompt.md`, stage 2 feeds
    the raw stage-1 response into `literal_prompt.md`, and the final
    answer is graded by checkform. Two API calls per row.
  - `--form story` — single-shot story formalization with
    `formalize_prompt.md` (experiment 02's task).
  - `--form literal` — single-shot literal formalization with
    `literal_prompt.md` (experiment 04's task).

  All calls at temperature 0.
- **Thinking mode**: `--reasoning off` only (experiment 05 established
  the on-regime ceiling; the no-think regime is where the two-stage
  effect lives). Native toggles as before: qwen3 gets `/no_think` on
  every call; gpt-5-mini floors at `effort: minimal`; gpt-5.5 supports
  a true `{"effort": "none", "exclude": true}` (the arm experiment 06
  validated); llama-3.3, mistral-small, and gpt-4o-mini get the prompt
  wrappers alone. Compliance clean: `max(reasoning_tokens,
  stage1_reasoning_tokens)` is 0 for every row except qwen3's familiar
  1-token think-block artifact.
- **Sampling** (seed 0, ETP `equations.txt` sha256 `e30e1a67…c8010`,
  unchanged): 160 pairs stratified 20 per total-operation bin 1–8,
  identical across the three runs (`run-strat20-s0-think-off`,
  `run-strat20-s0-story-think-off`, `run-strat20-s0-literal-think-off`;
  pair-id identity verified). Two consequences of `per_bin = 20`:
  - Bin 1 is *exhaustive*: the ETP list has exactly 2 zero-op and 5
    one-op laws, giving 20 ordered pairs total, and all 20 are in the
    sample (20/bin is the maximum perfectly-even design; 26/bin cannot
    be filled).
  - The pair set is **not** byte-identical to the strat5 sets of
    experiments 02–06 — a different per-bin draw count changes the RNG
    sequence after bin 1 — so comparisons to earlier experiments are
    aggregate-level, not per-pair.
- **Other**: `max_tokens` 4096 per call, temperature 0,
  `--concurrency 8`. 160 pairs × 9 models = 1,440 rows per arm
  (4,320 rows; 5,760 calls — two-stage makes two per row). Two-stage
  rows keep the standard schema plus `stage1_*` bookkeeping, as in
  experiment 05. OpenRouter-reported cost: two-stage $4.63 (gpt-5.5
  $3.25 of it), story $1.49, literal $1.63 — $7.75 total.

## Prompts

All templates byte-identical to their experiment 02/04/05 versions
(same sha256s at run time):

- Two-stage stage 1: `abstract_prompt.md` at sha256
  `1aa038f2d13dbddcc5c7f803d9166d9e8a82ce9b981a3463fc59a3d44cce8733`.
- Two-stage stage 2 and the literal arm: `literal_prompt.md` at sha256
  `089ffc52bb57c5aa7c2ead0e613ec7e73b11039aa79d729f8c80c63a0852f8b0`.
- Story arm: `formalize_prompt.md` at sha256
  `ad33f6de859156b81be0d889abd3c56e4d9275bd855eb6d804d4e8ebcfe4983c`.
- Regime wrappers (off regime; prefixes empty), from `run_meta.json`:
  - two-stage stage-1 suffix:
    > Respond with only the rewritten description, and no other text
    > before it.
  - two-stage stage-2 suffix, and the single-shot story and literal
    runs' suffix:
    > Respond with only the two required lines, and no other text
    > before them.

  plus a trailing `/no_think` line for qwen3 (every call).

## Reproduce

```sh
python3 benchmark.py --seed 0 --stratify-ops 20 --form two-stage --reasoning off \
    --concurrency 8 \
    --models deepseek/deepseek-chat-v3.1,qwen/qwen3-32b,meta-llama/llama-3.3-70b-instruct,mistralai/mistral-small-3.2-24b-instruct,openai/gpt-4o-mini,google/gemini-2.5-flash,anthropic/claude-haiku-4.5,openai/gpt-5-mini,openai/gpt-5.5 \
    --out-dir experiments/07-two-stage-scale/runs/run-strat20-s0-think-off

python3 benchmark.py --seed 0 --stratify-ops 20 --form story --reasoning off \
    --concurrency 8 \
    --models deepseek/deepseek-chat-v3.1,qwen/qwen3-32b,meta-llama/llama-3.3-70b-instruct,mistralai/mistral-small-3.2-24b-instruct,openai/gpt-4o-mini,google/gemini-2.5-flash,anthropic/claude-haiku-4.5,openai/gpt-5-mini,openai/gpt-5.5 \
    --out-dir experiments/07-two-stage-scale/runs/run-strat20-s0-story-think-off

python3 benchmark.py --seed 0 --stratify-ops 20 --form literal --reasoning off \
    --concurrency 8 \
    --models deepseek/deepseek-chat-v3.1,qwen/qwen3-32b,meta-llama/llama-3.3-70b-instruct,mistralai/mistral-small-3.2-24b-instruct,openai/gpt-4o-mini,google/gemini-2.5-flash,anthropic/claude-haiku-4.5,openai/gpt-5-mini,openai/gpt-5.5 \
    --out-dir experiments/07-two-stage-scale/runs/run-strat20-s0-literal-think-off

python3 charts.py experiments/07-two-stage-scale/runs/run-strat20-s0-think-off \
    --title "ETP formalization benchmark · two-stage at scale, no-think, + GPT-5.5" \
    --out experiments/07-two-stage-scale/report/benchmark-report.html --pdf

python3 charts.py experiments/07-two-stage-scale/runs/* \
    --title "ETP formalization benchmark · story vs literal vs two-stage at scale (no-think)" \
    --out experiments/07-two-stage-scale/report/comparison-report.html --pdf

# vacuous-law-excluded view (drops the 55 pairs containing E1 "x = x"
# or E2 "x = y"; 105 pairs and bins 2-8 remain):
python3 filter_vacuous.py experiments/07-two-stage-scale
python3 charts.py results/no-vacuous/07-two-stage-scale/runs/* \
    --title "ETP formalization benchmark · story vs literal vs two-stage at scale (no-think) · vacuous laws excluded" \
    --out experiments/07-two-stage-scale/report/comparison-report-no-vacuous.html --pdf
```

Requires checkform's term-depth cap (`_MAX_TERM_DEPTH`, added during
this experiment — see run notes).

## Results

Artifacts: the three run directories under `runs/` (each with
`summary.md`), `report/benchmark-report.html` / `.pdf` (the two-stage
run alone), `report/comparison-report.html` / `.pdf` (the headline
cross-arm figures: correct rate by model and form, and
accuracy-vs-complexity with one line per form, all on the identical
pair set), and `report/comparison-report-no-vacuous.html` / `.pdf`
(the same figures over the 105 pairs containing no vacuous law —
added after the bin-4 investigation below).

Correct% over the 160 stratified pairs, no-think, per arm (two-stage
column: experiment 05's strat5-off number in parentheses; not per-pair
comparable):

| model | story | literal | two-stage |
|---|---|---|---|
| openai/gpt-5.5 | 95.0 | **99.4** | **99.4** |
| deepseek/deepseek-chat-v3.1 | 79.4 | 80.0 | 73.8 (70.0) |
| google/gemini-2.5-flash | 80.0 | 87.5 | 73.8 (67.5) |
| anthropic/claude-haiku-4.5 | 54.4 | 86.9 | 65.0 (67.5) |
| meta-llama/llama-3.3-70b-instruct | 48.1 | 69.4 | 58.8 (60.0) |
| openai/gpt-5-mini | 53.8 | 65.0 | 56.9 (57.5) |
| mistralai/mistral-small-3.2-24b-instruct | 38.8 | 66.9 | 50.0 (42.5) |
| qwen/qwen3-32b | 37.5 | 63.7 | 38.8 (22.5) |
| openai/gpt-4o-mini | 31.9 | 51.9 | 23.1 (20.0) |

All three runs: 1,440/1,440 rows graded, 0 api-errors, regime
compliance clean (qwen3's 1-token artifact only). Unparseable: story
64, literal 52, two-stage 60 (27 of the two-stage ones gpt-5-mini).

Pooled correct% by total-operation bin 1–8 and by max nesting depth
1–4 (story / literal / two-stage):

| bin | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| story | 90 | 82 | 73 | 45 | 63 | 44 | 33 | 32 |
| literal | 100 | 99 | 96 | 71 | 71 | 62 | 52 | 46 |
| two-stage | 55 | 73 | 71 | 49 | 70 | 59 | 51 | 51 |

| depth | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| story | 89 | 69 | 49 | 29 |
| literal | 100 | 95 | 63 | 43 |
| two-stage | 69 | 68 | 61 | 40 |

**The bin-4 dip is a vacuous-law composition artifact, not a bug.**
Bin 4 is the only bin ≥ 4 whose splits can include a zero-op law (E1
`x = x` / E2 `x = y`; 0+5 needs a 5-op law, which doesn't exist), and
it drew 12/20 such pairs. Those pairs score 42.0% pooled vs 74.5% for
bin 4's other pairs — which sit exactly on the smooth curve — and bin
5 rebounds because vacuous pairs are structurally impossible there.
Law-level reading rates (a law graded correct up to its own side swap
/ dualization, pooled over arms and models) explain the mechanism:
0-op laws 79.7% (trivial is harder than simple — models embellish),
1-op 95.4%, 2-op 89.7%, 3-op 74.0%, 4-op 57.5% (depth-4 chains
~35–56%). A (0,4) split therefore multiplies a flat ~20% vacuous-side
tax by the worst per-law regime in the benchmark; below bin 4 the
partner stays on the easy side of the 3→4-op cliff, so bare pairs
underperform their bins (e.g. 67.4 vs 92.6 in bin 3) without making
the pooled curve non-monotone. The dry run grades this pair set 100%,
and hand-checked failures are genuine model errors (e.g. deepseek
shifting two pigments one nesting level in E1-E2254).

Correct% on the 105 pairs with no vacuous law
(`comparison-report-no-vacuous`):

| model | story | literal | two-stage |
|---|---|---|---|
| openai/gpt-5.5 | 92.4 | **99.0** | **99.0** |
| anthropic/claude-haiku-4.5 | 48.6 | 82.9 | 81.9 |
| google/gemini-2.5-flash | 75.2 | 84.8 | 74.3 |
| deepseek/deepseek-chat-v3.1 | 75.2 | 73.3 | 74.3 |
| meta-llama/llama-3.3-70b-instruct | 37.1 | 61.0 | 63.8 |
| mistralai/mistral-small-3.2-24b-instruct | 23.8 | 57.1 | 58.1 |
| openai/gpt-5-mini | 45.7 | 55.2 | 58.1 |
| qwen/qwen3-32b | 35.2 | 52.4 | 50.5 |
| openai/gpt-4o-mini | 16.2 | 36.2 | 29.5 |

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
- gpt-5.5's single two-stage miss (E1918 ⇒ E350, 7 ops, paint theme)
  is a wrong abstraction, not a formalization slip: its `ASK:` line
  encodes a different law than the story's. Its single literal miss is
  a different pair (E2878 ⇒ E1690), and its 8 story misses are 8
  further pairs — no pair is hard for it in more than one arm.
- The single-shot arms were run after (and because of) the two-stage
  results, on the identical pair set (`samples.jsonl` pair-id identity
  verified across the three runs before launching).
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
- **GPT-5.5 is not in the same regime as the rest of the field — except
  on the story.** 159/160 on both the literal and two-stage arms with
  reasoning truly off (different single misses, zero unparseable) —
  against an 87.5% best for the small models. But single-shot story
  formalization costs it 8 pairs (95.0%): the narrative disguise is the
  one part of the benchmark that still measures capability rather than
  formatting discipline at the frontier, and notably the two-stage
  rewrite recovers all of it (95.0 → 99.4 on the same pairs).
- **Drift is unrepairable, now at 0/61 pooled.** No row with a
  parseable-but-wrong stage-1 abstraction ended correct — 0/36 here,
  0/25 in experiment 05. Meanwhile faithful abstractions fail stage 2
  37% of the time (221/593) without thinking, so the off-regime
  pipeline loses accuracy at *both* stages, but only stage-1 drift is
  unrecoverable in principle. The case for a mechanical fidelity gate
  on the intermediate (CLAUDE.md's round-trip idea) gets stronger, not
  weaker, with sample size.
- **Experiment 05's ordering claim needs a revision at scale: literal
  dominates, but two-stage > story is model-dependent.** Literal ≥
  both other arms for all nine models, usually by a wide margin
  (haiku +21.9 over its next-best arm, qwen +24.9). But three models —
  deepseek (79.4 vs 73.8), gemini (80.0 vs 73.8), and gpt-4o-mini
  (31.9 vs 23.1) — formalize the *story* better single-shot than
  through their own two-stage rewrite; pooled, two-stage beats story by
  only 2.3 points (59.9 vs 57.6). Decomposition without thinking is
  not a general win over one-shot narrative reading; it's a win for
  roughly the models in the middle of the range (haiku +10.6,
  llama +10.7, mistral +11.2).
- **Where decomposition pays is complexity-dependent — and inverted.**
  The two-stage bin-1 dip is unique to that arm: on trivial laws the
  single-shot arms are at or near ceiling (story 90%, literal 100%)
  while two-stage sits at 55% — the stage-1 embellishment failure in
  its purest form. From bin 5 up the picture flips: two-stage tracks
  or beats story everywhere (bin 8: 51 vs 32) and even edges literal
  at bin 8 (51 vs 46). Same story by depth: two-stage has the flattest
  curve (69→40) against story's 89→29 and literal's 100→43. The
  rewrite is an externalized reasoning pass that costs a flat toll and
  pays off only once the story is deep enough that one-shot reading
  breaks down.
- **Excluding vacuous laws collapses most of the arm gaps at the low
  end — the "two-stage toll" is nearly all a vacuous-law toll.** On
  the 105 structured pairs, two-stage ≈ literal for seven of nine
  models (llama, mistral, and gpt-5-mini even edge literal), and every
  story-beats-two-stage inversion from the full sample disappears or
  flattens to a tie (gpt-4o-mini flips outright, 16.2 story vs 29.5
  two-stage). Stage-1 embellishment of trivial laws is where the
  decomposition loses; give every law real structure and the
  model-written rewrite is as good as the renderer's. The story arm,
  by contrast, stays far behind everywhere — the narrative cost is
  not a vacuous-law artifact.
- Methodological: a benchmark whose grader can be crashed by a
  degenerate generation is itself part of the measurement — the
  depth-cap fix belongs in the harness permanently, and the runaway
  `op(`-repetition failure mode (concentrated in gpt-5-mini's
  truncated rows) is worth counting explicitly if it recurs. Likewise,
  the total-operation bin axis conflates per-law difficulty profiles:
  depth (monotone here and in the filtered view) is the more
  trustworthy complexity axis for future experiments.
