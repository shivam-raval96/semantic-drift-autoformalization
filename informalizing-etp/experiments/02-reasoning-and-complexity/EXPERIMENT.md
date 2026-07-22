# 02 — Thinking mode and equation complexity

Run 2026-07-14/15.

## Question

How much of formalization accuracy is explained by thinking (reasoning
budget), and how does accuracy vary with equation complexity (total
operation count) once thinking is controlled?

## Setup

- **Models** (8, via OpenRouter — four open-weight, four lightweight
  closed):
  - `deepseek/deepseek-chat-v3.1`
  - `qwen/qwen3-32b`
  - `meta-llama/llama-3.3-70b-instruct`
  - `mistralai/mistral-small-3.2-24b-instruct`
  - `openai/gpt-4o-mini`
  - `google/gemini-2.5-flash`
  - `anthropic/claude-haiku-4.5`
  - `openai/gpt-5-mini`
- **Thinking mode**: standardized with `--reasoning on|off`, applied
  identically to every model — a uniform prompt wrapper (below) plus the
  native OpenRouter reasoning toggle where the model supports it. Known
  floors: qwen3 additionally needs `/no_think` in the off regime; gpt-5-mini
  cannot fully disable reasoning, so off floors at `effort: minimal`.
  llama-3.3, mistral-small, and gpt-4o-mini have no native toggle and get
  the wrapper alone. Each results row records `reasoning_tokens`, so regime
  compliance is auditable per response.
- **Sampling** (seed 0 throughout, ETP `equations.txt` sha256
  `e30e1a67…c8010`), two schemes × two regimes = 4 runs:
  - `run-s0-n30-think-{off,on}` — 30 uniform pairs (skew toward maximal
    4+4-operation pairs).
  - `run-strat5-s0-think-{off,on}` — 40 pairs stratified 5 per
    total-operation bin 1–8, for the accuracy-vs-complexity curve.
- **Other**: `max_tokens` 4096 (off), 16384/32768 (on, uniform/stratified);
  temperature 0.

## Prompts

- Template: `formalize_prompt.md` at sha256
  `ad33f6de859156b81be0d889abd3c56e4d9275bd855eb6d804d4e8ebcfe4983c`
  (unchanged from Experiment 01; the version in the repo reproduces every
  base prompt byte-exactly).
- Regime wrappers, applied uniformly to all models on top of the template
  output:
  - **on** — prefix:
    > Work through the story step by step first — write out what expression
    > each numbered intermediate stands for, one at a time — and only then
    > finish with the two required lines.
  - **off** — suffix:
    > Respond with only the two required lines, and no other text before
    > them.

    plus a trailing `/no_think` line for qwen3 models only.
- Base prompts are the `prompt` field in each run's `samples.jsonl`; the
  wrapper actually used is recorded verbatim in each `run_meta.json`
  (`regime_prefix`/`regime_suffix`, with `native_reasoning` giving the
  per-model API toggle), and each results row carries `sent_prompt_hash`.

## Reproduce

```sh
python3 benchmark.py --seed 0 --n 30 --reasoning off \
    --out-dir experiments/02-reasoning-and-complexity/runs/run-s0-n30-think-off
python3 benchmark.py --seed 0 --n 30 --reasoning on \
    --out-dir experiments/02-reasoning-and-complexity/runs/run-s0-n30-think-on
python3 benchmark.py --seed 0 --stratify-ops 5 --reasoning off \
    --out-dir experiments/02-reasoning-and-complexity/runs/run-strat5-s0-think-off
python3 benchmark.py --seed 0 --stratify-ops 5 --reasoning on --max-tokens 32768 \
    --out-dir experiments/02-reasoning-and-complexity/runs/run-strat5-s0-think-on

python3 charts.py experiments/02-reasoning-and-complexity/runs/* \
    --out experiments/02-reasoning-and-complexity/report/benchmark-report.html --pdf
```

(All eight default models; `--models` may be omitted.)

## Results

Artifacts: the four run directories under `runs/` (each with `summary.md`),
and `report/benchmark-report.html` / `.pdf` — regime dumbbells per model,
accuracy-vs-complexity lines, and verdict-composition bars.

Correct% by regime (uniform n=30 / stratified 40):

| model | off | on |
|---|---|---|
| deepseek/deepseek-chat-v3.1 | 56.7 / 75.0 | 100 / 100 |
| google/gemini-2.5-flash | 56.7 / 82.5 | 100 / 100 |
| anthropic/claude-haiku-4.5 | 16.7 / 57.5 | 100 / 97.5 |
| openai/gpt-5-mini | 16.7 / 47.5 | 100 / 100 |
| qwen/qwen3-32b | 3.3 / 27.5 | 100 / 87.5 |
| meta-llama/llama-3.3-70b-instruct | 16.7 / 60.0 | 96.7 / 95.0 |
| mistralai/mistral-small-3.2-24b-instruct | 6.7 / 37.5 | 93.3 / 92.5 |
| openai/gpt-4o-mini | 0.0 / 37.5 | 70.0 / 87.5 |

**Vacuous-law-excluded view** (added 2026-07-22). Experiment 07
identified pairs containing a vacuous law (E1 `x = x` / E2 `x = y`) as
a measurement hazard, and the convention is now to exclude them
(`experiments/README.md`). The uniform n=30 runs contain no such
pairs; the stratified runs keep 24 of 40.
`report/benchmark-report-no-vacuous.html` / `.pdf` re-renders the
charts over the filtered copies
(`python3 filter_vacuous.py experiments/02-reasoning-and-complexity`,
then the charts.py command over
`results/no-vacuous/02-reasoning-and-complexity/runs/*`). Correct% on
the 24 surviving stratified pairs (original 40-pair number in
parentheses):

| model | off | on |
|---|---|---|
| google/gemini-2.5-flash | 79.2 (82.5) | 100 (100) |
| deepseek/deepseek-chat-v3.1 | 58.3 (75.0) | 100 (100) |
| meta-llama/llama-3.3-70b-instruct | 50.0 (60.0) | 100 (95.0) |
| anthropic/claude-haiku-4.5 | 41.7 (57.5) | 100 (97.5) |
| mistralai/mistral-small-3.2-24b-instruct | 25.0 (37.5) | 95.8 (92.5) |
| openai/gpt-4o-mini | 25.0 (37.5) | 79.2 (87.5) |
| openai/gpt-5-mini | 25.0 (47.5) | 100 (100) |
| qwen/qwen3-32b | 12.5 (27.5) | 100 (87.5) |

Every model drops in the off regime — vacuous pairs were the
single-shot story arm's *easy* cases (experiment 07 measured 90%
pooled story accuracy on them) — so the original stratified-off
numbers flatter this arm; the on regime stays at or near ceiling.

## Conclusions

- Thinking dominates: with reasoning on, every model scores 87.5–100%
  except gpt-4o-mini (70–87.5%); with reasoning off, the same models drop
  to 0–82.5%. Experiment 01's model spread was largely a reasoning-budget
  effect.
- Complexity matters mostly when not thinking: stratified (easier-pair)
  accuracy is far above uniform (maximal-pair) accuracy in the off regime,
  while the on regime is near ceiling at all complexities for the strong
  models.
- Even the step-by-step prompt wrapper without a native toggle (llama,
  mistral, gpt-4o-mini) recovers most of the gap — much of "thinking" here
  is just being allowed to write the intermediates out.
- Follow-ups this raises: harder/fuzzier stories to pull the on-regime off
  ceiling, and a plain-language control to isolate the cost of the
  narrative disguise (see `experiments/README.md` backlog).
