# 04 — Structured literal: named intermediates in the literal form

Run 2026-07-21.

## Question

Experiment 03's headline reversal — with thinking on, the story arm *beat*
the literal arm on complex pairs for six of eight models — suggested the
story form's named intermediates ("Batch 1 ...") scaffold the
reconstruction of deep terms, while the old literal form's inline nested
prose had to be tracked unaided. This experiment tests that reading
directly: `literalform.py` now renders each application of the operation
as its own definition step with a named result ("apply the operation to x
as its first input and y as its second input, and call the result
Value 1"), mirroring the story arm's intermediates without any narrative
disguise. Repeating Experiment 03's setup on this structured literal
rendering: does naming the intermediates close the literal arm's
thinking-on gap on complex pairs (pinning the reversal on naming rather
than narrative), and does it change the no-thinking picture?

## Setup

Identical to Experiment 03 (same models, regimes, seeds, sampling — pair
sets are byte-identical by construction) except that `--form literal` now
renders definition steps with named intermediates, and `literal_prompt.md`
was revised to teach unfolding them.

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
- **Form**: `--form literal` — structured literal descriptions
  (`literalform.py` after the definition-step change) with the revised
  `literal_prompt.md`. Sampling never consults the form, so seed 0 yields
  the byte-identical (E, F) pair sets as Experiments 02 and 03.
- **Thinking mode**: standardized with `--reasoning on|off`, applied
  identically to every model — the same uniform prompt wrappers as
  Experiment 03 (byte-identical, deliberately: "each application of the
  operation" now points at exactly one definition step) plus the native
  OpenRouter reasoning toggle where supported. Known floors as before:
  qwen3 additionally needs `/no_think` in the off regime; gpt-5-mini
  floors at `effort: minimal`. llama-3.3, mistral-small, and gpt-4o-mini
  have no native toggle and get the wrapper alone. Each results row
  records `reasoning_tokens`.
- **Sampling** (seed 0 throughout, ETP `equations.txt` sha256
  `e30e1a67…c8010` — unchanged from Experiments 02–03), two schemes × two
  regimes = 4 runs:
  - `run-s0-n30-think-{off,on}` — 30 uniform pairs (skew toward maximal
    4+4-operation pairs).
  - `run-strat5-s0-think-{off,on}` — 40 pairs stratified 5 per
    total-operation bin 1–8.
- **Other**: `max_tokens` 4096 (off), 16384/32768 (on,
  uniform/stratified); temperature 0.

## Prompts

- Template: `literal_prompt.md` at sha256
  `089ffc52bb57c5aa7c2ead0e613ec7e73b11039aa79d729f8c80c63a0852f8b0`
  (revised for this experiment: the worked example and rules teach the
  definition-step grammar and the unfolding of Value names; the
  `op(first, second)` / `ASSUME:` / `ASK:` contract is unchanged).
- Regime wrappers, byte-identical to Experiment 03's:
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

Requires `literalform.py` and `literal_prompt.md` at this experiment's
commit — the pre-04 literal renderer produced inline nested prose instead.

```sh
python3 benchmark.py --seed 0 --n 30 --form literal --reasoning off \
    --out-dir experiments/04-structured-literal/runs/run-s0-n30-think-off
python3 benchmark.py --seed 0 --n 30 --form literal --reasoning on \
    --out-dir experiments/04-structured-literal/runs/run-s0-n30-think-on
python3 benchmark.py --seed 0 --stratify-ops 5 --form literal --reasoning off \
    --out-dir experiments/04-structured-literal/runs/run-strat5-s0-think-off
python3 benchmark.py --seed 0 --stratify-ops 5 --form literal --reasoning on --max-tokens 32768 \
    --out-dir experiments/04-structured-literal/runs/run-strat5-s0-think-on

python3 charts.py experiments/04-structured-literal/runs/* \
    --out experiments/04-structured-literal/report/benchmark-report.html --pdf
```

(All eight default models; `--models` may be omitted.)

## Results

Artifacts: the four run directories under `runs/` (each with `summary.md`),
and `report/benchmark-report.html` / `.pdf`.

Correct% by regime (uniform n=30 / stratified 40), experiment-02/03 row
order:

| model | off | on |
|---|---|---|
| deepseek/deepseek-chat-v3.1 | 60.0 / 90.0 | 100 / 100 |
| google/gemini-2.5-flash | 83.3 / 85.0 | 100 / 100 |
| anthropic/claude-haiku-4.5 | 53.3 / 82.5 | 100 / 100 |
| openai/gpt-5-mini | 26.7 / 70.0 | 100 / 100 |
| qwen/qwen3-32b | 20.0 / 52.5 | 100 / 97.5 |
| meta-llama/llama-3.3-70b-instruct | 30.0 / 80.0 | 93.3 / 100 |
| mistralai/mistral-small-3.2-24b-instruct | 23.3 / 60.0 | 100 / 95.0 |
| openai/gpt-4o-mini | 6.7 / 57.5 | 100 / 100 |

Run notes:

- Two provider errors survived their in-run retries — one deepseek call
  (E4189-E3981, uniform/on; the same pair that was flaky for gemini in
  experiment 03) and one gemini call (E157-E3846, stratified/on) — and
  were completed by resumed reruns, so the on-regime `results.jsonl`
  files contain superseded api-error rows (summaries use the latest row
  per pair × model). Final state: 0 api-errors across all 1,120 calls.
- Regime compliance as in experiments 02–03: qwen3 logs a median of
  1 reasoning token on some off-regime rows (negligible); llama-3.3,
  mistral-small, and gpt-4o-mini have no native toggle (0 reasoning
  tokens, wrapper only).
- `correct-swapped` is common in both regimes (off/stratified deepseek:
  18 of its 36 correct rows) — models often state an equation with sides
  swapped, which checkform accepts. No `correct-dualized` rows anywhere,
  as in experiment 03: with explicit first/second-input wording, no model
  dualized.
- Off-regime failures skew `wrong` rather than `unparseable`, except
  gpt-5-mini (uniform/off: 10 of 30 unparseable at `effort: minimal`).

**Vacuous-law-excluded view** (added 2026-07-22). Experiment 07
identified pairs containing a vacuous law (E1 `x = x` / E2 `x = y`) as
a measurement hazard, and the convention is now to exclude them
(`experiments/README.md`). The uniform n=30 runs contain no such
pairs; the stratified runs keep 24 of 40.
`report/benchmark-report-no-vacuous.html` / `.pdf` re-renders the
charts over the filtered copies (`python3 filter_vacuous.py
experiments/04-structured-literal`, then the charts.py command over
the `results/no-vacuous/...` runs). Correct% on the 24 surviving
stratified pairs (original 40-pair number in parentheses):

| model | off | on |
|---|---|---|
| deepseek/deepseek-chat-v3.1 | 87.5 (90.0) | 100 (100) |
| google/gemini-2.5-flash | 79.2 (85.0) | 100 (100) |
| meta-llama/llama-3.3-70b-instruct | 75.0 (80.0) | 100 (100) |
| anthropic/claude-haiku-4.5 | 75.0 (82.5) | 100 (100) |
| openai/gpt-5-mini | 66.7 (70.0) | 100 (100) |
| mistralai/mistral-small-3.2-24b-instruct | 41.7 (60.0) | 100 (95.0) |
| qwen/qwen3-32b | 37.5 (52.5) | 95.8 (97.5) |
| openai/gpt-4o-mini | 33.3 (57.5) | 100 (100) |

The off regime drops hardest at the weak end (gpt-4o-mini −24.2,
mistral −18.3, qwen −15.0): renderer-written literal text reads
vacuous laws at ~100% (experiment 07), so those pairs padded exactly
the models with the least headroom. The on regime stays at ceiling.

## Conclusions

- **Named intermediates close the thinking-on gap completely.** On the
  uniform (maximal 4+4-op) sample with thinking on, every model scores
  93.3–100 versus 30–100 for the inline literal form (experiment 03) —
  llama 66.7 → 93.3, mistral 60 → 100, gpt-5-mini 76.7 → 100, gpt-4o-mini
  30 → 100. Experiment 03's reversal (story beating literal on complex
  pairs when models can think) is thereby pinned on the named-intermediate
  step structure, not on anything narrative: the structured literal now
  matches or beats the story arm's on-regime numbers (experiment 02:
  llama 96.7, mistral 93.3, gpt-4o-mini 70). The complexity gradient
  vanishes too — llama goes 5/5 in every stratified ops bin 1–8 with
  thinking on (87.5% overall in experiment 03).
- **The same scaffold is a burden without thinking.** Off-regime accuracy
  drops sharply versus the inline literal on the identical pairs —
  uniform: gpt-4o-mini 66.7 → 6.7, qwen 90 → 20, haiku 83.3 → 53.3,
  gpt-5-mini 60 → 26.7. Unfolding definition chains is inherently
  multi-step composition; inline nested prose could be transcribed to
  `op(...)` almost token by token.
- **This refines experiment 03's headline.** Story-form's off-regime
  collapse was there attributed to the narrative disguise, because the
  inline literal scored so much higher. But the structured literal —
  no narrative, same step structure as the stories — lands much closer to
  the story arm off-regime (uniform: story 0–56.7, structured literal
  6.7–83.3, inline literal 50–100, per model in the same order). Most of
  the no-thinking difficulty is the named-intermediate indirection
  itself; the disguise adds a smaller increment on top.
- Net effect: rendering form and thinking regime interact strongly.
  Definition-step renderings (story or literal) reward thinking and
  scaffold deep terms to ceiling; inline renderings are the easiest
  no-thinking form but cap thinking-on accuracy on complex pairs.
- Follow-ups: a fuzzier-story arm (planned in experiments/README.md)
  can now be read against a controlled baseline — with structure held
  constant, any story-vs-structured-literal difference isolates narrative
  distance alone.
