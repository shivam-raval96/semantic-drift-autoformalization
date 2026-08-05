# PLAN — Probing & steering: is translation correctness linearly represented?

Proposal in the lab template: hypotheses carry refutation conditions; expected results are
written before any run; Actual results stay empty until runs exist.

## Question

1. **Probing.** When Llama-3.1-8B-Instruct reads (story prompt + candidate RG answer), does
   a linear probe on the residual stream separate correct from incorrect answers?
2. **Steering.** Does the correct−incorrect mean-difference direction, added during
   generation, causally move translation verdicts on problems the model currently fails?

## Models

- Subject (activations): `meta-llama/Llama-3.1-8B-Instruct` (bf16, A10G via Modal) — lab
  convention and the ft-experiments base model.
- Contrastive data: v1 wrong answers are mechanical perturbations (no LLM involved). A
  sibling set of frontier-model-generated natural errors is planned [meeting suggestion —
  emphasis to be discussed]; generator model unpinned.

## Dataset (contrast_v1)

- 1,000 problems (stratified over complexity, ETP + genform, vacuous excluded, seeded),
  rendered with the repo pipeline: story + reference RG.
- Per problem: a correct answer (reference RG, checkform-verified `correct`) and a wrong
  answer (single-node perturbation of the reference — argument swap / variable
  substitution / subtree edit — checkform-verified well-formed `wrong`). 2,000 labeled
  texts, exact 50/50 by construction.
- Labels are mechanical (checkform) — R1. Probes are monitors, never training signal for
  the model — R5 untouched.
- Provenance per row: source, law labels, ops/depth, pair hash, perturbation type, split
  group.

## Hypotheses

- **H1 (probing):** mid-layer probes reach AUROC well above every baseline under
  held-out-law splits. *Refuted if* probe ≈ bag-of-words baseline out-of-law →
  correctness is not linearly represented at the tested sites (or only lexically).
- **H2 (steering):** the verdict distribution moves monotonically with steering strength α
  while a norm-matched random direction stays flat. *Refuted if* there is no differential
  effect, or the direction only inflates the unparseable rate (coherence loss, not
  control).

## Sanity checks

- Every dataset row re-graded by checkform before freezing (correct→`correct`,
  wrong→`wrong`).
- Capture two sites (answer-last-token, answer-mean-pooled): single-site nulls are
  probe-site artifacts (lab precedent).
- Shuffled-label probe must land at chance (harness check).

## Baselines that must be beaten

- Bag-of-words / TF-IDF logistic regression on the same texts (the lexical floor).
- Layer-0 (embedding) probe.
- Majority class (50%).

## Expected results (written before running)

- Per-layer AUROC curve rising from ≈lexical floor at layer 0 to a mid-layer peak
  (precedent: related certificate-gold drift probe at 0.96 AUROC; layer-16 steering effect
  on 8B).
- Steering dose–response: correct% up / wrong% down along +α at moderate α; unparseable
  rising at large α in both arms (norm effect), verdict movement only in the steered arm.

## Actual results

**Probing (runs/probe-v1/probe_results.json, 2026-08-05).** Curves rise from 0.50 at
layer 0 to peaks of 0.599 (last, L32) / 0.623 (mean, L28); controls clean (shuffled
0.49–0.52, length 0.50, BoW floor 0.539). BUT under the law-disjoint split the probe
falls to 0.520–0.527 (fold range 0.47–0.58) ≈ the BoW floor — the pre-registered
refutation condition for H1. Signal is law-tied, concentrated on easy problems
(easy 0.695 vs medium/hard ~0.59 at best layer) and on size-changing edits
(prune 0.686 / grow 0.648 vs var_sub 0.589 / arg_swap 0.565).

**Verdict:** H1 refuted for base Llama-3.1-8B reading bare story+answer text at these
sites — no law-general linear correctness direction. Consistent with the model's
behavioral floor on this task (~0% correct translations in ft-experiments base evals).
H2 (steering) is moot from this direction as specified. Candidate follow-ups, not yet
decided: probe the grammar-FT'd checkpoint (does FT create the direction?), task-framed
verification prompts, perturbation-token-local sites, or a task-competent model.
