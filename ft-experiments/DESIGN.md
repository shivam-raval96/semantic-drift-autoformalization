# FT v2 — task-pair fine-tuning with cross-grammar generalization

Status: DRAFT (implementation details pending pipeline audit). Locked sections
are marked LOCKED and change only with a written amendment here.

## Question

Does fine-tuning on story->grammar pairs teach translation (a semantic skill
that transfers to a never-trained output grammar) or a format (a story->string
mapping locked to the trained notation)? Mentor design: train on pairs in one
grammar, evaluate the same problems asked in a different rigid grammar.

## Grammars (LOCKED)

All three serialize the SAME equation ASTs; grading always parses back to the
AST and reuses the existing canonicalization + verdict machinery (R1 intact).

- **A (trained)** — existing RG: `ASSUME:` / `ASK:`, prefix `op(a, b)`,
  variables x,y,z,w,u,v by first appearance.
- **B-near (never trained)** — surface-vocabulary swap, structure identical:
  `GIVEN:` / `SHOW:`, prefix `f(a, b)`, same variable convention.
  Tests: can the model re-skin its learned format when told to in-context?
- **B-far (never trained)** — structural change: infix, `LAW:` / `DERIVE:`,
  fully parenthesized `(l ∘ r)` notation (the repo's existing infix surface, so
  its battle-tested parser `storyform.parse_equation` grades it — no new parser
  to trust), same variable convention. `∘` avoids the markdown/`*` stripping
  hazard in answer extraction. Tests: does the learned mapping live at the
  tree level or the token level?

## Training (LOCKED unless smoke test forces a change)

- Data: the train_v1 equation pool (pair-disjoint from all SAIR rows,
  law-disjoint from eval_v1 — audit re-run for v2), rendered to stories with
  the repo renderer, paired with the reference grammar-A serialization.
- **Theme holdout:** stories rendered in 3 of the 4 themes only; the held-out
  theme's ~25% slice of eval_v1 measures input-side generalization.
- Format: the exact frozen story-arm eval template as the prompt, reference RG
  as the completion. Loss on completion tokens only (prompt masked), EOS
  trained. No packing.
- Models: Llama-3.1-8B-Instruct (base at ~0% correct: injection test) and
  Qwen3-32B (base 13-61% by tier/arm: displacement test). Same LoRA recipe as
  v1 for comparability: r=16, alpha=16, all layers, q/k/v/o/gate/up/down.
  LR 2e-4 cosine, seed 0. Checkpoints at step 0 and every ~50 steps.
- Possible third model: only if the model-suite research finds a candidate that
  passes a limit-200 signal screen (base correct in ~10-60% band). Not blocking.

## Eval matrix (LOCKED)

Frozen eval_v1 (777 problems, 4 tiers). Greedy, no-think, frozen sha-pinned
templates. Three-way verdicts (correct / wrong-well-formed / unparseable) per
tier, never pooled as the headline.

| Cell | grammar A | B-near | B-far |
|---|---|---|---|
| base | exists (v1) | new | new |
| FT | new | new | new |

Arms: story->G and literal->G (literal = untrained input side, free input
generalization axis; two-stage dropped for v2 scope). Prompt templates for
B-near/B-far mirror the A templates exactly, with only the grammar
specification section replaced; frozen before any model sees them.

**Format-following control:** a small mechanical suite of unrelated
in-context output-format tasks (JSON / bracketed list / keyword formats on
trivial content), base vs FT. Separates grammar-specific lock-in from general
SFT narrowing if B degrades.

**Checkpoint curves:** correct% and unparseable% on grammar A and B-far
(limit-200 subsets, seeded, -limitN guard) at every saved checkpoint — the
skill-vs-lock-in dynamics plot. Final cells run on the full 777 at the
curve-chosen checkpoint.

**Representation arm:** capture + law-disjoint probe on FT models
(contrast_v1, existing instruments), reported separately from behavior (R7).

## Predictions and kill criteria (LOCKED, written before any run)

- **P1 skill transfer:** FT-A correct up AND FT-B-far correct up (unparseable
  not exploding) => translation learned; notation is a thin layer.
- **P2 format lock-in:** FT-A up, FT-B-far flat or down, format control
  intact => grammar-specific lock-in (v1's override result, now for
  task-training). If the format control also collapses => general SFT
  narrowing, weaker claim.
- **P3 null:** FT-A flat with training-side probes confirming learning
  happened (loss, holdout pair perplexity, sample generations) => task FT
  does not inject translation at this scale/recipe; write it up, stop.
- Noise bar: tiers are 180-200 items; ±3-4 pts is noise. Claims need
  consistent sign across tiers and pooled deltas well beyond it.
- B-near sits between: near-transfer only (B-near up, B-far flat) reads as
  token-level re-skinning without tree-level abstraction.

## Implementation notes (from pipeline audit, 2026-08-19)

- **Grading path for B grammars (transcode seam, zero grader refactor):**
  extract labeled lines with a B-specific extractor (checkform's regex is
  hardcoded to ASSUME|ASK) → parse with the B parser → AST → re-serialize via
  `ftlib.rg_text` → `checkform.grade` unchanged. B parse failures are our
  "unparseable". Round-trip identity enforced by `ftlib.verify_rg_round_trip`
  on every reference.
- **B-near parser:** small parametrized clone of checkform's `_PrefixParser`
  (op token `f`, labels GIVEN/SHOW); B-far parser: `storyform.parse_equation`
  as-is (labels LAW/DERIVE). Variable renaming per-equation by first
  appearance (x,y,z,w,u,v), matching `ftlib.rg_equation` convention exactly.
- **Answer extraction for B:** mirror checkform's `_LINE_RE` with the new
  labels; cleaner strips backticks/bold but NOT `*` or `.` (checkform's
  `_clean` eats those — hazard documented in audit).
- **Theme holdout:** `render_story(e, f, theme_key)` takes an explicit theme;
  train_v2 assigns deterministically (hash mod 3) over the 3 training themes.
  eval_v1 keeps its natural 4-theme assignment; the held-out theme's slice is
  the input-generalization probe.
- **Template pinning:** new B templates registered in ft-experiments
  `config.py EVAL["template_shas"]` (eval scripts assert digests pre-run).
- **Story-render failures hard-crash in the contrast builder;** train_v2
  generator must catch per-pair, log drop counts, top up quotas.

## Relation to Denis's mech-interp results (branch denislim/mech-interp)

Denis (Qwen3-4B, team-update-exp01-exp03.md, 2026-08-15): law identity is
decodable form-invariantly mid-stack (86% vs 1.9% chance), steering the
story->literal direction is a clean null, and accuracy is governed by
thinking-token budget (story needs ~2x literal's budget) — his read:
the bottleneck is serial compute, not representation style. No overlap with
v2 (his story->RG follow-up is representational, not training-based). Two
implications adopted here: (a) his steering null independently corroborates
our represented-but-not-read family; (b) under our no-think protocol, P1
success reads as "pair-training amortizes the serial computation into the
weights", and P3 reads as "that computation cannot be amortized at this
scale/recipe" — testable against his budget curve if we later add a
thinking-on arm.

## Standing rules

R1 mechanical ground truth; R5 no training against checker verdicts on model
outputs (labels here are reference serializations, fixed before any model
runs); R7 translation and implication numbers never share a table; eval_v1
frozen; one change at a time (v2 changes training data + objective vs v1 —
grammar B exists only on the eval side).
