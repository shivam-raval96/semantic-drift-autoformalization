# 05 — Two-stage formalization: abstract first, then formalize

Run 2026-07-21.

## Question

Experiments 02 and 04 measured single-shot formalization of the story and
structured-literal arms. This experiment decomposes the story task into
the two steps those arms span: stage 1 asks the model to *abstract* the
themed story into a structured literal description (the `literalform.py`
definition-step grammar, taught one-shot by the new `abstract_prompt.md`);
stage 2 feeds stage 1's raw output — verbatim, no extraction or
validation — into the unchanged `literal_prompt.md` to produce the
`ASSUME:`/`ASK:` lines, graded as usual. Does explicit abstraction match
or beat single-shot story formalization (experiment 02), and when the
pipeline fails, where does the fidelity go — in stage 1's abstraction or
in stage 2's formalization? Stage 2's task is byte-comparable to
experiment 04's single-stage literal arm on the identical pair set, so
(two-stage) − (exp 04) isolates stage-1 drift and (two-stage) − (exp 02)
isolates the value of decomposition.

## Setup

Identical models, regimes, seeds, and sampling as Experiments 02–04 (pair
sets are byte-identical by construction); the only new variable is the
two-stage pipeline (`--form two-stage`), which makes two API calls per
row instead of one.

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
- **Form**: `--form two-stage` — the themed story (same renderer and
  hence byte-identical story text as `--form story`) prompted with
  `abstract_prompt.md` (stage 1), whose raw response fills
  `literal_prompt.md`'s `{story}` slot (stage 2). The final response is
  graded by checkform against the pair's canonical forms, exactly as in
  the single-stage arms. Both calls go to the same model at
  temperature 0.
- **Thinking mode**: standardized with `--reasoning on|off`, applied
  identically to every model and to both stages. Stage 2 uses the
  literal-form wrappers byte-identical to Experiments 03–04; stage 1 uses
  a new "abstract" wording (below) because its required output is a
  description, not the two lines. Native reasoning toggles and floors as
  before: qwen3 needs `/no_think` in the off regime (both stages),
  gpt-5-mini floors at `effort: minimal`; llama-3.3, mistral-small, and
  gpt-4o-mini have no native toggle and get the wrappers alone. Rows
  record `reasoning_tokens` (stage 2) and `stage1_reasoning_tokens`.
- **Sampling** (seed 0 throughout, ETP `equations.txt` sha256
  `e30e1a67…c8010` — unchanged from Experiments 02–04), two schemes × two
  regimes = 4 runs:
  - `run-s0-n30-think-{off,on}` — 30 uniform pairs (skew toward maximal
    4+4-operation pairs).
  - `run-strat5-s0-think-{off,on}` — 40 pairs stratified 5 per
    total-operation bin 1–8.
- **Other**: `max_tokens` applies per call — 4096 (off), 16384/32768
  (on, uniform/stratified); temperature 0. Results rows keep the
  standard schema with top-level call fields describing the graded
  stage-2 call, plus `stage1_*` fields (response, sent-prompt hash,
  usage, latency, finish reason, reasoning tokens, routing) for the
  abstraction call. A failure at either stage buckets as `api-error` and
  is retried whole on resume; an on-topic-but-wrong stage-1 output is
  *not* an error — it flows into stage 2, because end-to-end fidelity is
  the thing measured.

## Prompts

- Stage 1 template: `abstract_prompt.md` (new for this experiment) at
  sha256
  `1aa038f2d13dbddcc5c7f803d9166d9e8a82ce9b981a3463fc59a3d44cce8733`.
  It defines the three-paragraph literalform grammar, then gives a
  one-shot worked example reusing `formalize_prompt.md`'s librarian
  story (a theme deliberately absent from `themes/`) paired with the
  byte-exact `literalform.py` rendering of the same laws
  (`x ∘ (x ∘ y) = y` ⇒ `x ∘ y = y ∘ x`, the same worked example
  `literal_prompt.md` teaches from) — `test_benchmark.py` pins both
  blockquotes to their sources.
- Stage 2 template: `literal_prompt.md` at sha256
  `089ffc52bb57c5aa7c2ead0e613ec7e73b11039aa79d729f8c80c63a0852f8b0` —
  byte-identical to Experiment 04's, so stage 2 differs from that
  experiment only in receiving model-written rather than
  renderer-written descriptions.
- Regime wrappers, applied per stage:
  - **on** — stage-1 prefix (new "abstract" variant):
    > Work through the story step by step first — write out what
    > expression each numbered intermediate stands for, one at a time —
    > and only then finish with the complete rewritten description.

    stage-2 prefix: the literal-form wording, byte-identical to
    Experiments 03–04.
  - **off** — stage-1 suffix:
    > Respond with only the rewritten description, and no other text
    > before it.

    stage-2 suffix: the two-required-lines wording, byte-identical to
    Experiments 01–04; plus a trailing `/no_think` line for qwen3 models
    only (both stages).
- Stage-1 base prompts are the `prompt` field in each run's
  `samples.jsonl`; stage-2 prompts are built at run time from stage-1
  responses and are not stored, but each row carries `sent_prompt_hash`
  (stage 2) and `stage1_sent_prompt_hash`, and `run_meta.json` records
  both templates' sha256s and both wrapper pairs verbatim
  (`regime_prefix`/`regime_suffix` for stage 1,
  `stage2_regime_prefix`/`stage2_regime_suffix` for stage 2).

## Reproduce

Requires `abstract_prompt.md`, `literal_prompt.md`, and `benchmark.py`'s
two-stage arm at this experiment's commit.

```sh
python3 benchmark.py --seed 0 --n 30 --form two-stage --reasoning off \
    --out-dir experiments/05-two-stage/runs/run-s0-n30-think-off
python3 benchmark.py --seed 0 --n 30 --form two-stage --reasoning on \
    --out-dir experiments/05-two-stage/runs/run-s0-n30-think-on
python3 benchmark.py --seed 0 --stratify-ops 5 --form two-stage --reasoning off \
    --out-dir experiments/05-two-stage/runs/run-strat5-s0-think-off
python3 benchmark.py --seed 0 --stratify-ops 5 --form two-stage --reasoning on --max-tokens 32768 \
    --out-dir experiments/05-two-stage/runs/run-strat5-s0-think-on

python3 charts.py experiments/05-two-stage/runs/* \
    --out experiments/05-two-stage/report/benchmark-report.html --pdf

# cross-arm comparison against the committed exp-02 (story) and exp-04
# (literal) runs — identical pair sets by construction:
python3 charts.py \
    experiments/02-reasoning-and-complexity/runs/* \
    experiments/04-structured-literal/runs/* \
    experiments/05-two-stage/runs/* \
    --title "ETP formalization benchmark · story vs literal vs two-stage" \
    --out experiments/05-two-stage/report/comparison-report.html --pdf
```

(All eight default models; `--models` may be omitted.)

## Results

Artifacts: the four run directories under `runs/` (each with `summary.md`),
`report/benchmark-report.html` / `.pdf` (this experiment's runs, regime
comparison), and `report/comparison-report.html` / `.pdf` (the headline
figures: story vs literal vs two-stage per model and regime, on the
identical pair sets from experiments 02 and 04).

Correct% by regime (uniform n=30 / stratified 40), experiment-02/03/04 row
order:

| model | off | on |
|---|---|---|
| deepseek/deepseek-chat-v3.1 | 70.0 / 70.0 | 100 / 100 |
| google/gemini-2.5-flash | 56.7 / 67.5 | 96.7 / 100 |
| anthropic/claude-haiku-4.5 | 50.0 / 67.5 | 100 / 92.5 |
| openai/gpt-5-mini | 33.3 / 57.5 | 100 / 100 |
| qwen/qwen3-32b | 16.7 / 22.5 | 93.3 / 77.5 |
| meta-llama/llama-3.3-70b-instruct | 36.7 / 60.0 | 96.7 / 90.0 |
| mistralai/mistral-small-3.2-24b-instruct | 16.7 / 42.5 | 90.0 / 77.5 |
| openai/gpt-4o-mini | 6.7 / 20.0 | 86.7 / 55.0 |

Stage attribution: for every row, the tail of the stage-1 response (from
its last "Consider a collection …" paragraph) was mechanically
back-parsed with `literalform.backparse_literal` and compared against the
pair's canonical laws under the accepted symmetries. **faithful** = exact
literalform grammar encoding the right laws; **drifted** = parses but
encodes different laws; **off-grammar** = the tail does not back-parse
(this includes harmless phrasing deviations, which is why many
off-grammar rows still end correct). Final-verdict counts
(correct / fail):

| run | stage-1 faithful | drifted | off-grammar |
|---|---|---|---|
| n30 off | 69 / 91 | 0 / 8 | 17 / 55 |
| n30 on | 166 / 2 | 0 / 3 | 63 / 6 |
| strat5 off | 103 / 60 | 0 / 5 | 60 / 92 |
| strat5 on | 151 / 1 | 0 / 9 | 126 / 33 |

Run notes:

- 22 provider errors survived their in-run retries (7 gemini + 2 deepseek
  uniform/on; 6 gemini + 4 deepseek + 3 qwen stratified/on), all
  content-`null` responses with `finish_reason` `"error"`; resumed reruns
  completed them, so the on-regime `results.jsonl` files contain
  superseded api-error rows (summaries and charts use the latest row per
  pair × model). Two qwen rows (E2-E42, E29-E2, stratified/on) failed
  differently: qwen burned the entire 32,768-token budget on reasoning in
  stage 1 and returned no content, deterministically across three
  attempts, until a different provider route graded them (one exact, one
  wrong). Final state: 0 api-errors across all 1,120 tasks.
- Regime compliance clean in both off runs, now checked over
  `max(reasoning_tokens, stage1_reasoning_tokens)`: qwen3's familiar
  1-token think-block artifact only. The summary tables' reasoning-token
  columns describe the graded stage-2 call; stage-1 reasoning lives in
  `stage1_reasoning_tokens` (gemini medians ≈ 1.7k there).
- First `correct-dualized` rows in the literal-prompt family: 5
  (stratified qwen 3, deepseek 1, llama 1). Experiments 03–04 had none
  because the literal text fixes the input convention; here stage 1 reads
  the story, which deliberately does not, so a consistently mirrored
  abstraction grades correct-dualized — a faithful reading, as designed.
- `correct-swapped` is very common for deepseek with thinking on (22 of
  30 / 26 of 40 correct rows); a swap can be introduced by either stage.
- gpt-4o-mini's stratified/on collapse (55.0) is concentrated in the
  low-complexity bins: on laws with a bare-variable side it invents an
  operation application in stage 1 — e.g. rendering the questioned law
  `x = x` as "apply the operation to x and x … does Value 1 equal
  Value 1?" — the exact degenerate-side pitfall `CLAUDE.md` warns
  renderers about. All 18 of its stratified/on failures are off-grammar
  stage-1 outputs of this character.

## Conclusions

- **Decomposition helps exactly where models cannot think, and costs
  fidelity where they can.** Off-regime uniform, two-stage beats
  single-shot story formalization (experiment 02) for every model —
  haiku 16.7 → 50.0, llama 16.7 → 36.7, gpt-5-mini 16.7 → 33.3, deepseek
  56.7 → 70.0, gpt-4o-mini 0 → 6.7 — landing near experiment 04's
  single-stage literal arm on the same pairs. The stage-1 rewrite acts as
  an externalized reasoning pass: it converts the story problem into
  (approximately) the literal problem even without native thinking. With
  thinking on, the strong models match experiment 02's ceiling (100
  everywhere for deepseek/haiku/gpt-5-mini), but the weaker half pays for
  the second hop on the stratified sample — qwen 87.5 → 77.5, mistral
  92.5 → 77.5, gpt-4o-mini 87.5 → 55.0 — where single-stage had nothing
  to compound.
- **A ground-truth-blind stage 2 cannot repair drift.** Across all 1,120
  tasks, not one row with a drifted (parseable-but-wrong) stage-1
  abstraction ended correct (0/25). Conversely, with thinking on, stage 2
  almost never fumbles a faithful abstraction (3 failures in 320 such
  rows across both on-runs — consistent with experiment 04, where stage
  2's task on renderer-perfect input was at ceiling). The pipeline's
  binding constraint is abstraction fidelity, not formalization: if a
  two-stage autoformalization pipeline is to beat one-shot, the leverage
  is a fidelity check on the intermediate representation (e.g. the
  round-trip gate already sketched in CLAUDE.md's roadmap), not a better
  formalizer.
- **Degenerate laws are the drift hotspot.** Stage-1 failures concentrate
  where a law's side is a bare variable (complexity bins 1–3) — models
  invent structure the story never described. This inverts the usual
  complexity story: uniform n=30 (nearly all maximal 4+4-op pairs) is the
  *easy* sample for the two-stage arm (gpt-4o-mini: 86.7 on, better than
  its 70.0 single-shot), while the stratified sample's simple pairs are
  where it breaks. Abstraction of deep terms is mechanical; abstraction
  of trivial ones invites embellishment.
- Reading across experiments 02/04/05 with thinking on: story ≈ two-stage
  ≈ literal at the top of the model range — the rendering form barely
  matters once models can reason and intermediates are named. The
  differences live in the weaker models and the no-thinking regime, and
  there the ordering is literal ≥ two-stage > story: each hop away from
  the explicit rendering costs, and the model-written hop costs less than
  the narrative one.
- Follow-up raised: gate stage 2 on a mechanical `backparse_literal`
  round-trip of stage 1's output (retry or fall back to single-shot on
  failure) — the off-grammar-but-correct rows suggest a lenient
  normalizer would be needed, and the drifted rows put an upper bound of
  ~2% on what a perfect gate could recover here.
