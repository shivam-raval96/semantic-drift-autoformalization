# 13 — Implication truth: can models *decide* the implication, per form?

Run 2026-07-24/25.

## Question

Experiments 01–12 measure whether a model can *read* an implication
statement — reconstruct the pair of laws from a story or description.
This experiment asks the much harder question the corpus was built to
carry: can a model *decide* the implication? Given "law E implies law
F" rendered in one of three presentation formats, the model must answer
whether F is forced in every structure satisfying E — a theorem-or-
counterexample judgment, graded against the ETP's Lean-verified ground
truth. Does the presentation format (themed story vs literal description
vs the rigid grammar) change models' ability to *reason about*
the mathematics, over and above what it costs them to parse it?

The formalization experiments give a per-form reading baseline: whatever
accuracy a model loses moving from rigid grammar to literal to story on the
*truth* task, only the part exceeding its reading loss is a genuine
"reasoning tax" of the representation.

## Difficulty context

Chance is 50% by construction (bins are balanced 50/50 true/false), and
a constant answerer also scores 50% — so per-class recall
(`true acc%` / `false acc%`) and the answer-bias rate (`ans-true%`) in
each summary are the vitals to read before accuracy.

What is known about the intrinsic difficulty of these judgments:

- In the ETP itself, ~99.995% of the 22M implications were resolved
  automatically (Vampire: 63% refuted by finite-model building, 37%
  proved by superposition; arXiv:2512.07087) — a *random* implication
  is overwhelmingly ATP-easy, and the famous hard tail (677→255, the
  Asterix/Obelix pair 65↔1491, 854, 1485, 1323) needed human insight
  and infinite constructions. Uniformly sampled pairs measure the easy
  bulk, not that tail.
- False implications split by counterexample kind: most fall to a small
  finite magma; the hardest need infinite ones. The `status` field
  (explicit vs implicit proof) is recorded per pair as a weak
  difficulty tag; Vampire per-implication solve times
  (`2025-08-11-vampire.json.gz` upstream) and the late-open list
  (`unknowns_10_07.json`) are sharper signals left for a follow-up.
- From experiments 02/07/09, *reading* difficulty is monotone in
  operation count with a 3→4-op cliff in the no-think regime. Truth
  judgment adds proof search on top, so the per-bin curves here ride on
  a much lower ceiling.

## Setup

- **Task**: single call per (pair, model); the model reasons and must
  end with `ANSWER: True` or `ANSWER: False`. Graded by string
  comparison (`proveform.py`); the last conclusive ANSWER line wins,
  hedges are unparseable. A successful call that returns *no* content —
  in practice the model spending its whole token budget on reasoning
  (`finish_reason: length`) — is also graded unparseable: failing to
  answer is a model outcome, not an infrastructure error, and stays in
  the denominator. Only genuine transport errors are excluded (none
  remain in the final runs).
- **Ground truth**: upstream outcomes snapshot 2024-11-10
  (`truthdata.py`; zip sha256 recorded in each run_meta.json, matrix
  sha256 `d9b70013ea296dd8e3f848e3494ff70ea88cd2b3b9a9df4ae9942e2cdc19b861`).
  Only Lean-verified `*_proof_*` statuses are sampled; the snapshot's 4
  unknown + 186 conjecture entries are excluded.
- **Models** (4, via OpenRouter — three open-weight and one lightweight
  closed):
  - `deepseek/deepseek-chat-v3.1`
  - `qwen/qwen3-32b`
  - `meta-llama/llama-3.3-70b-instruct`
  - `openai/gpt-5-mini`
- **Forms** (three runs over the byte-identical pair set — sampling
  never consults the form): `--form story` (storyform narrative),
  `--form literal` (literalform description), `--form symbolic` (the rigid-grammar arm, new
  `symbolform.py`: the implication as the two `ASSUME:`/`ASK:` lines of
  checkform's own prefix grammar). Prompts `prove_story_prompt.md`,
  `prove_literal_prompt.md`, `prove_symbolic_prompt.md` — parallel
  skeletons, identical task semantics and worked examples (one True,
  one False), differing only in the input-format section. No prompt
  contains equation labels or any ETP reference (runner hard-fails on
  leakage).
- **Thinking mode**: `--reasoning on` only — truth judgment needs
  derivations/counterexamples and the off regime risks flooring every
  model at chance; an off-regime triple is the natural follow-up.
  Temperature 0, max_tokens 16384.
- **Sampling** (seed 0): 70 pairs via `--per-bin 10 --bins 2:8` — 10
  pairs per total-ops bin, each bin exactly 5 proof-true + 5
  proof-false (`truthdata.sample_truth_balanced`). Vacuous laws E1/E2
  and the diagonal are structurally excluded. Per-bin availability
  (true/false): bin 2 = 10/10 (the quota takes half the bin's entire
  20-pair population), bin 3 = 170/220, bin 8 = 6.8M/11.6M.

## Reproduce

```sh
set -a; source ../.env; set +a          # OPENROUTER_API_KEY
python3 truthdata.py build              # one-time ground-truth ingest
for form in story literal symbolic; do
  python3 experiments/13-implication-truth/run_experiment.py \
    --form $form --per-bin 10 --bins 2:8 --seed 0 --reasoning on \
    --out-dir experiments/13-implication-truth/runs/run-truth10-s0-$form-think-on
done
python3 charts.py experiments/13-implication-truth/runs/run-truth10-s0-*-think-on \
  --out experiments/13-implication-truth/report/report.html
```

Dry-run validation (offline, must grade 100% correct):
`--dry-run --out-dir results/truth-dry-$form`.

## Results

All 280 (pair, model) tasks per arm graded; unparseable (no conclusive
ANSWER line, or an empty budget-exhausted response) counts as a failure.
Chance = 50%.

**Accuracy by model and form**

| model | story | literal | rigid grammar |
|---|---|---|---|
| deepseek-chat-v3.1 | 88.6 | 90.0 | 91.4 |
| gpt-5-mini | 87.1 | 91.4 | 92.9 |
| qwen3-32b | 75.7 | **91.4** | **91.4** |
| llama-3.3-70b-instruct | 51.4 | 60.0 | 54.3 |
| *pooled* | *75.7* | *83.2* | *82.5* |

**Per-class recall and answer bias** (true acc / false acc · ans-true%)

| model | story | literal | rigid grammar |
|---|---|---|---|
| deepseek-chat-v3.1 | 82.9 / 94.3 · 47 | 88.6 / 91.4 · 49 | 91.4 / 91.4 · 50 |
| gpt-5-mini | 77.1 / 97.1 · 41 | 85.7 / 97.1 · 45 | 88.6 / 97.1 · 46 |
| qwen3-32b | 62.9 / 88.6 · 41 | 94.3 / 88.6 · 54 | 91.4 / 91.4 · 49 |
| llama-3.3-70b-instruct | 14.3 / 88.6 · 13 | 22.9 / 97.1 · 13 | 20.0 / 88.6 · 16 |

**Pooled accuracy by total-ops bin** (story / literal / rigid grammar):
bin 2: 90 / 95 / 92.5 · bin 3: 75 / 82.5 / 82.5 · bin 4: 82.5 / 87.5 /
87.5 · bin 5: 72.5 / 90 / 85 · bin 6: 77.5 / 75 / 75 · bin 7: 72.5 /
80 / 77.5 · bin 8: **60 / 72.5 / 77.5**.

**By proof kind** (explicit n=24 per arm, implicit n=256): story 83.3 /
75.0 · literal 91.7 / 82.4 · rigid grammar 100.0 / 80.9 — directly-proved
edges are easier than transitive-closure-only ones in every arm.

**No-answer rows** (graded unparseable): story 12 (qwen 7, deepseek 4,
gpt-5-mini 1), literal 7, rigid grammar 6. Roughly half are empty responses —
deepseek/qwen burning the full 16k-token budget on reasoning without
ever emitting the ANSWER line — and the rest responses with no
conclusive final line. llama, which barely reasons (0 median reasoning
tokens), never fails to answer; it just answers badly.

Report: `report/report.html`.

## Conclusions

1. **Random ETP implications are decidable by mid-tier models when
   thinking is on.** Three of four models sit at 87–93% in their best
   format — far above the 50% chance floor — confirming that the
   uniformly-sampled bulk of the implication graph (the part Vampire
   also found easy) is LLM-tractable.
2. **The story tax is a reasoning tax, and it lands on qwen.**
   qwen3-32b loses ~16 points moving literal → story (91.4 → 75.7)
   while deepseek and gpt-5-mini barely move; pooled accuracy drops
   from ~83% (literal/rigid grammar) to 75.7% (story). The loss concentrates
   on true implications (qwen true-recall 94.3 literal vs 62.9 story):
   the narrative disguise impairs *derivation* much more than
   *refutation*.
3. **Literal ≈ rigid grammar everywhere.** Once the disguise is gone, the
   choice of plain-English steps vs rigid `op(...)` notation is
   irrelevant — same conclusion the formalization experiments reached
   for reading, now for reasoning.
4. **Refuting is uniformly easier than proving.** False-recall is
   88–97% for every model in every arm; true-recall is what separates
   models and formats. Every model leans False (ans-true < 50% in 11 of
   12 cells), consistent with counterexample-hunting being the easier
   move on this distribution.
5. **llama-3.3-70b does not do this task — it votes.** Answering True
   only 13–16% of the time, its 51–60% accuracy is mostly the false
   half of the balanced deck. Accuracy alone would have read as "at
   chance"; the per-class recall and bias columns are what expose the
   degenerate strategy.
6. **Difficulty rises with complexity, fastest in the story arm** (60%
   pooled at bin 8 vs 77.5% rigid grammar), echoing the formalization
   experiments' complexity slope — on a much lower ceiling.
7. **A failure mode formalization never showed: reasoning-budget
   exhaustion.** deepseek and qwen sometimes spend all 16k tokens
   without committing to an answer. Truth judgment needs an answer
   budget policy (or higher caps) that formalization never did.

Follow-ups: the `--reasoning off` triple (does the story tax explode
without thinking?); frontier models on the hard tail (unknowns list,
Vampire-slow implications) instead of the easy bulk; per-pair
paired-arm analysis of which implications flip between story and
literal.
