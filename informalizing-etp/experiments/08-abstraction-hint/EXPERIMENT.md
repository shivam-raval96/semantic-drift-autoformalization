# 08 — Abstract-first hints in the story prompt: two arms, no-think

Run 2026-07-22.

## Question

Experiment 07 confirmed at scale that when models cannot think, routing
the themed story through an explicit abstraction call (two-stage) beats
formalizing it directly. This experiment asks whether any of that
benefit can be recovered **in a single call** by steering the strategy
inside the direct story prompt itself:

1. **Hint arm** — `formalize_prompt.md` plus one instruction at the
   top: first abstract away unnecessary details, then translate.
2. **Hint + example arm** — the same instruction, plus the worked
   example extended to show what an abstracted (LiteralNL) version of a
   story looks like before its translation.

Both arms are graded against experiment 07's no-vacuous story arm (the
same 105 pairs): does the hint move single-call accuracy toward the
two-stage arm, and does seeing an example of the abstraction move it
further than the instruction alone?

## Setup

Same pipeline, grading, and regime machinery as experiment 07; the
changed variable is the story-arm prompt template (plus the new
sampling-time vacuous exclusion).

- **Models** (9, via OpenRouter — the experiment 07 set):
  - `deepseek/deepseek-chat-v3.1`
  - `qwen/qwen3-32b`
  - `meta-llama/llama-3.3-70b-instruct`
  - `mistralai/mistral-small-3.2-24b-instruct`
  - `openai/gpt-4o-mini`
  - `google/gemini-2.5-flash`
  - `anthropic/claude-haiku-4.5`
  - `openai/gpt-5-mini`
  - `openai/gpt-5.5`
- **Forms**: both runs are `--form story` — single user message, no
  system prompt, identical call mechanics to experiment 07's story arm —
  with `--prompt-template` selecting the hint template (a benchmark.py
  flag added for this experiment; the template override is recorded in
  `run_meta.json`). All calls at temperature 0.
- **Thinking mode**: `--reasoning off` only, as in experiment 07 (the
  no-think regime is where the two-stage effect lives, and a visible
  abstraction step is impossible — the hint can only steer internal
  computation). Native toggles unchanged: qwen3 gets `/no_think`,
  gpt-5-mini floors at `effort: minimal`, gpt-5.5 gets
  `{"effort": "none"}`, llama-3.3 / mistral-small / gpt-4o-mini get the
  prompt wrapper alone.
- **Sampling** (seed 0, ETP `equations.txt` sha256 `e30e1a67…c8010`):
  the experiment 07 draw — 160 pairs stratified 20 per total-operation
  bin 1–8 — with the new `--exclude-vacuous` flag dropping the 55 pairs
  containing a zero-op law (E1 `x = x` / E2 `x = y`) after the draw,
  per the vacuous-law convention in `experiments/README.md`. The RNG
  stream is untouched, so the surviving 105 pairs (bins 2–8) are
  pair-id-identical to experiment 07's no-vacuous view — verified
  against `results/no-vacuous/07-two-stage-scale` before launching.
  Unlike experiment 07, the vacuous pairs are never sent to the API.
- **Other**: `max_tokens` 4096, `--concurrency 8`. 105 pairs × 9 models
  = 945 single-shot calls per arm, 1,890 total.

## Prompts

Both templates are copies of `formalize_prompt.md` (experiment 07's
story arm, sha256 `ad33f6de…4983c`) with insertions; nothing else
changes. Both share, verbatim, a new paragraph after the opening
paragraph:

> The approach you should take to finding a formalization is first
> abstracting away unnecessary details about the story, and only then
> translating into the output format.

- **Hint arm**: `formalize_hint_prompt.md` at sha256
  `09d184c18ad22fb5bd313d865d223c3655f8d99a4ea079e487441a416e039172`.
  The added paragraph is the arm's entire diff (verified byte-exact by
  `test_benchmark.py`).
- **Hint + example arm**: `formalize_hint_example_prompt.md` at sha256
  `ac1ba5033883989c8f03e9c7bcc443bddb16b43d90a495f22ae98a33317d2b4a`.
  Beyond the shared paragraph, the worked example demonstrates the
  strategy: after the librarian story (unchanged), a one-line marker
  ("The abstracted version of this story:") introduces its LiteralNL
  abstraction as a second blockquote — byte-consistent with
  `literalform.render_description`, the same text as
  `abstract_prompt.md`'s worked example — and the translation then
  proceeds from the lettered abstraction, so the final example lines
  become `ASSUME: op(x, op(x, y)) = y` / `ASK: op(x, y) = op(y, x)`
  (a variable renaming of the original; grading is unaffected).
  Rule 1 gains "…, or the letters you introduced when abstracting" so
  the lettered example does not contradict it.
- Regime wrapper (off regime; prefix empty), unchanged from the
  experiment 07 story arm:
  > Respond with only the two required lines, and no other text
  > before them.

  plus a trailing `/no_think` line for qwen3 (every call).

## Reproduce

```sh
python3 benchmark.py --seed 0 --stratify-ops 20 --exclude-vacuous \
    --form story --prompt-template formalize_hint_prompt.md --reasoning off \
    --concurrency 8 \
    --models deepseek/deepseek-chat-v3.1,qwen/qwen3-32b,meta-llama/llama-3.3-70b-instruct,mistralai/mistral-small-3.2-24b-instruct,openai/gpt-4o-mini,google/gemini-2.5-flash,anthropic/claude-haiku-4.5,openai/gpt-5-mini,openai/gpt-5.5 \
    --out-dir experiments/08-abstraction-hint/runs/run-strat20-s0-story-hint-think-off

python3 benchmark.py --seed 0 --stratify-ops 20 --exclude-vacuous \
    --form story --prompt-template formalize_hint_example_prompt.md --reasoning off \
    --concurrency 8 \
    --models deepseek/deepseek-chat-v3.1,qwen/qwen3-32b,meta-llama/llama-3.3-70b-instruct,mistralai/mistral-small-3.2-24b-instruct,openai/gpt-4o-mini,google/gemini-2.5-flash,anthropic/claude-haiku-4.5,openai/gpt-5-mini,openai/gpt-5.5 \
    --out-dir experiments/08-abstraction-hint/runs/run-strat20-s0-story-hint-example-think-off

# comparison against experiment 07's arms over the same 105 pairs:
python3 filter_vacuous.py experiments/07-two-stage-scale
python3 charts.py results/no-vacuous/07-two-stage-scale/runs/* \
    experiments/08-abstraction-hint/runs/* \
    --title "ETP formalization benchmark · abstract-first hints vs experiment 07 arms (no-think, vacuous laws excluded)" \
    --out experiments/08-abstraction-hint/report/comparison-report.html --pdf
```

## Results

All 1,890 calls succeeded (zero api-error rows; every model graded on
all 105 pairs). Regime compliance clean: no reasoning tokens anywhere
except qwen3's familiar 1-token empty-think-block artifact.
OpenRouter-reported cost: hint $1.05, hint+example $1.21 — $2.26 total.

Correct% (exact + correct-swapped + correct-dualized) per model, both
arms against experiment 07's no-vacuous arms over the identical 105
pairs:

| model | story (07) | hint | hint+example | two-stage (07) | literal (07) |
|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 75.2 | 76.2 | 66.7 | 74.3 | 73.3 |
| qwen/qwen3-32b | 35.2 | 28.6 | 23.8 | 50.5 | 52.4 |
| meta-llama/llama-3.3-70b-instruct | 37.1 | 40.0 | 38.1 | 63.8 | 61.0 |
| mistralai/mistral-small-3.2-24b-instruct | 23.8 | 24.8 | 20.0 | 58.1 | 57.1 |
| openai/gpt-4o-mini | 16.2 | 21.9 | 17.1 | 29.5 | 36.2 |
| google/gemini-2.5-flash | 75.2 | 73.3 | 68.6 | 74.3 | 84.8 |
| anthropic/claude-haiku-4.5 | 48.6 | 44.8 | 44.8 | 81.9 | 82.9 |
| openai/gpt-5-mini | 45.7 | 49.5 | 47.6 | 58.1 | 55.2 |
| openai/gpt-5.5 | 92.4 | 94.3 | 93.3 | 99.0 | 99.0 |
| **mean** | **49.9** | **50.4** | **46.7** | **65.5** | **66.9** |

Paired per-(pair, model) comparison against the story arm (n = 945
shared rows):

- **Hint**: 48 rows gained, 44 lost (92 discordant; two-sided sign
  test p = 0.76). A wash.
- **Hint + example**: 47 gained, 78 lost (125 discordant; p = 0.007).
  A real regression, spread across models (deepseek −8.5, qwen3 −11.4,
  gemini −6.6, mistral −3.8 points) and across complexity bins 4–7
  rather than concentrated anywhere; gpt-5.5 stays at its ceiling.

## Conclusions

- **Telling a no-think model how to solve the task does not help.**
  The one-line abstract-first instruction moves mean accuracy by +0.5
  points (48/44 paired flips — noise). Whatever the two-stage pipeline
  adds, it is not knowledge of the strategy.
- **Showing the abstraction makes things worse.** The example arm loses
  3.2 mean points (p = 0.007), with the losses spread across mid-to-high
  complexity bins and most non-ceiling models. Speculatively: the
  lettered example invites models to attempt an internal
  story→letters→notation translation they cannot carry out without
  visible intermediate tokens, where the plain prompt lets them
  transcribe story names directly.
- **The two-stage gap stands.** Two-stage (65.5) and literal (66.9)
  remain ~16 points above every single-call story variant on the same
  pairs. Combined with experiments 05/07, the evidence now says the
  benefit lives in actually *generating* the intermediate description —
  externalized computation — not in the strategy framing, so prompt
  steering alone cannot substitute for the second call.
