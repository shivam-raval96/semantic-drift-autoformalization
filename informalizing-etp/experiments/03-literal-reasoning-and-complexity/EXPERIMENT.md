# 03 — Literal form: thinking mode and equation complexity

Run 2026-07-20.

## Question

Experiment 02 repeated for the literal arm: with the narrative disguise
removed — the same implications rendered as direct literal descriptions
(`--form literal`) instead of themed stories — how much of formalization
accuracy is explained by thinking, and how does accuracy vary with
equation complexity once thinking is controlled? Compared against
Experiment 02 (identical pair sets by construction), this isolates the
cost of the story rendering itself.

## Setup

Identical to Experiment 02 except `--form literal`.

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
- **Form**: `--form literal` — direct literal descriptions
  (`literalform.py`) with the `literal_prompt.md` formalization prompt.
  Sampling never consults the form, so seed 0 yields the byte-identical
  (E, F) pair sets as Experiment 02's runs.
- **Thinking mode**: standardized with `--reasoning on|off`, applied
  identically to every model — a uniform prompt wrapper (below; the "on"
  prefix is the literal-form variant) plus the native OpenRouter
  reasoning toggle where supported. Known floors as in Experiment 02:
  qwen3 additionally needs `/no_think` in the off regime; gpt-5-mini
  floors at `effort: minimal`. llama-3.3, mistral-small, and gpt-4o-mini
  have no native toggle and get the wrapper alone. Each results row
  records `reasoning_tokens`.
- **Sampling** (seed 0 throughout, ETP `equations.txt` sha256
  `e30e1a67…c8010` — unchanged from Experiment 02), two schemes × two
  regimes = 4 runs:
  - `run-s0-n30-think-{off,on}` — 30 uniform pairs (skew toward maximal
    4+4-operation pairs).
  - `run-strat5-s0-think-{off,on}` — 40 pairs stratified 5 per
    total-operation bin 1–8.
- **Other**: `max_tokens` 4096 (off), 16384/32768 (on,
  uniform/stratified); temperature 0.

## Prompts

- Template: `literal_prompt.md` at sha256
  `b5527a931ae68acc7ffe24e9be652ca3d04e3601b637946097e4df665d61acad`.
- Regime wrappers, applied uniformly to all models on top of the template
  output:
  - **on** — prefix (literal-form variant):
    > Work through the description step by step first — write out what
    > expression each application of the operation stands for, one at a
    > time — and only then finish with the two required lines.
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
python3 benchmark.py --seed 0 --n 30 --form literal --reasoning off \
    --out-dir experiments/03-literal-reasoning-and-complexity/runs/run-s0-n30-think-off
python3 benchmark.py --seed 0 --n 30 --form literal --reasoning on \
    --out-dir experiments/03-literal-reasoning-and-complexity/runs/run-s0-n30-think-on
python3 benchmark.py --seed 0 --stratify-ops 5 --form literal --reasoning off \
    --out-dir experiments/03-literal-reasoning-and-complexity/runs/run-strat5-s0-think-off
python3 benchmark.py --seed 0 --stratify-ops 5 --form literal --reasoning on --max-tokens 32768 \
    --out-dir experiments/03-literal-reasoning-and-complexity/runs/run-strat5-s0-think-on

python3 charts.py experiments/03-literal-reasoning-and-complexity/runs/* \
    --out experiments/03-literal-reasoning-and-complexity/report/benchmark-report.html --pdf
```

(All eight default models; `--models` may be omitted.)

## Results

Artifacts: the four run directories under `runs/` (each with `summary.md`),
and `report/benchmark-report.html` / `.pdf`.

Correct% by regime (uniform n=30 / stratified 40), experiment-02 row order:

| model | off | on |
|---|---|---|
| deepseek/deepseek-chat-v3.1 | 73.3 / 92.5 | 100 / 100 |
| google/gemini-2.5-flash | 100 / 97.5 | 86.7 / 95.0 |
| anthropic/claude-haiku-4.5 | 83.3 / 92.5 | 90.0 / 95.0 |
| openai/gpt-5-mini | 60.0 / 95.0 | 76.7 / 100 |
| qwen/qwen3-32b | 90.0 / 95.0 | 90.0 / 92.5 |
| meta-llama/llama-3.3-70b-instruct | 60.0 / 87.5 | 66.7 / 87.5 |
| mistralai/mistral-small-3.2-24b-instruct | 50.0 / 80.0 | 60.0 / 85.0 |
| openai/gpt-4o-mini | 66.7 / 97.5 | 30.0 / 75.0 |

Run notes:

- One gemini call (E4189-E3981, uniform/on) returned provider
  `finish_reason: error` on four consecutive attempts before succeeding on
  the fifth; the thinking-on `results.jsonl` files therefore contain
  superseded api-error rows from resumed retries (summaries use the latest
  row per pair × model). Final state: 0 api-errors across all 1,120 calls.
- Regime compliance as in experiment 02: qwen3 logs a median of 1 reasoning
  token on some off-regime rows (negligible); llama-3.3, mistral-small, and
  gpt-4o-mini have no native toggle (0 reasoning tokens, wrapper only).
- `correct-swapped` is common in the on regime (deepseek: 9–11 of its
  correct rows) — models often state the equation with sides swapped, which
  checkform accepts. No `correct-dualized` rows anywhere: with explicit
  first/second-input wording, no model dualized.

**Vacuous-law-excluded view** (added 2026-07-22). Experiment 07
identified pairs containing a vacuous law (E1 `x = x` / E2 `x = y`) as
a measurement hazard, and the convention is now to exclude them
(`experiments/README.md`). The uniform n=30 runs contain no such
pairs; the stratified runs keep 24 of 40.
`report/benchmark-report-no-vacuous.html` / `.pdf` re-renders the
charts over the filtered copies (`python3 filter_vacuous.py
experiments/03-literal-reasoning-and-complexity`, then the charts.py
command over the `results/no-vacuous/...` runs). Correct% on the 24
surviving stratified pairs (original 40-pair number in parentheses):

| model | off | on |
|---|---|---|
| openai/gpt-4o-mini | 95.8 (97.5) | 66.7 (75.0) |
| google/gemini-2.5-flash | 95.8 (97.5) | 91.7 (95.0) |
| qwen/qwen3-32b | 91.7 (95.0) | 87.5 (92.5) |
| openai/gpt-5-mini | 91.7 (95.0) | 100 (100) |
| deepseek/deepseek-chat-v3.1 | 87.5 (92.5) | 100 (100) |
| anthropic/claude-haiku-4.5 | 87.5 (92.5) | 91.7 (95.0) |
| meta-llama/llama-3.3-70b-instruct | 79.2 (87.5) | 83.3 (87.5) |
| mistralai/mistral-small-3.2-24b-instruct | 70.8 (80.0) | 79.2 (85.0) |

Uniform small drops (2–9 points) in both regimes: vacuous pairs were
easy for this arm too, but its margins were narrower to begin with.

## Conclusions

- The narrative disguise is most of the no-thinking difficulty. With
  reasoning off, the literal arm scores 50–100 (uniform) and 80–97.5
  (stratified) versus the story arm's 0–56.7 and 27.5–82.5 on the identical
  pairs. Story-form experiment 02's off-regime collapse largely measured
  story decoding, not formalization.
- Thinking barely helps the literal form, and sometimes hurts: gains are
  modest (deepseek +26.7 uniform, gpt-5-mini +16.7), qwen is flat, and
  gemini (100 → 86.7 uniform) and gpt-4o-mini (66.7 → 30.0 uniform,
  97.5 → 75.0 stratified) get *worse* — gpt-4o-mini's on-regime failures
  are wrong/unparseable answers after verbose step-by-step work.
- Striking reversal on complex pairs with thinking on: the story arm *beats*
  the literal arm for six of eight models on the uniform (maximal 4+4-op)
  sample — e.g. llama 96.7 story vs 66.7 literal, mistral 93.3 vs 60.0,
  gpt-5-mini 100 vs 76.7. A plausible reading: the story form's named
  intermediates ("Batch 1…") scaffold the reconstruction of deep terms,
  while the literal form's inline nested prose must be tracked unaided. The
  disguise costs when models can't think, but its step structure pays for
  itself once they can.
- Complexity still matters most for weaker models: llama's on-regime gap
  (87.5 stratified vs 66.7 uniform) persists where the strong models are
  near ceiling at all complexities.
- Follow-ups: test whether adding named intermediates to the literal form
  (a "structured literal" arm) closes its on-regime gap, which would pin
  the reversal on naming rather than narrative; and check whether
  gpt-4o-mini's thinking-on regression reproduces at higher max_tokens.
