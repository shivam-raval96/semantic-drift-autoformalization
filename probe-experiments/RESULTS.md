# Probing: is translation correctness linearly represented? — results for review

## 1. What the experiment is

When Llama-3.1-8B-Instruct *reads* a themed story (a deterministically informalized ETP
implication) together with a candidate rigid-grammar formalization, is the correctness of
that candidate **linearly decodable** from the model's residual stream — and, if a robust
direction exists, does injecting it during generation steer translation behavior?
Contrastive-pair design from the mentor meeting; hypotheses and refutation conditions
pre-registered in `PLAN.md` before any run. Relevant background fact: this same base
model scores ~0–1% correct at *producing* these translations
(`ft-experiments/runs/base-v1`), so this probes whether the evaluation representation
exists even where generation fails.

## 2. Setup

**Dataset (`contrast_v1`, frozen, sha-pinned).** 1,000 problems: 334 easy (ops_total
2–4, ETP laws) / 333 medium (5–8, ETP) / 333 hard (10–12, genform synthetics); vacuous
laws excluded; seeded and byte-reproducible. Per problem: story (repo renderer,
back-parse verified), correct answer = reference RG (round-trip verified), wrong answer =
**one surgical AST edit** of the reference — arg_swap / var_sub / prune / grow (counts
226/274/247/253) — accepted only if checkform grades it well-formed `wrong`; 193
candidate edits landing inside the grader's symmetry orbit were rejected and resampled;
both wrong lines keep ≥1 `op(`. Labels are 100% mechanical (checkform; R1). 2,000 texts,
exact 50/50.

**Capture.** Bare text `story + "\n\n" + answer` (no chat template), reader mode (no
generation). Llama-3.1-8B-Instruct bf16 on A10G; residual stream at all 33 layers
(embeddings + 32 blocks), two sites per text: final answer token (`last`) and mean over
answer tokens (`mean`); float16; activations checksum-bound to the dataset.

**Probing.** Per layer × site: StandardScaler + logistic regression, 5-fold GroupKFold
grouped by problem (a problem's correct/wrong twins never straddle a split); all AUROCs
out-of-fold. Controls: char-TF-IDF (2–5-grams) on the answer text alone (lexical floor);
answer-length single-feature probe; layer-0 embeddings; shuffled-train-label probes
(3 seeds); law-disjoint robustness split (GroupKFold by law connected component — caveat:
the largest component holds 556/1000 problems).

**Steering.** Not run. Pre-registered H2 takes its direction from the best probe layer;
the probing outcome (below) removed its premise — a direction that does not generalize
across laws cannot produce a law-general behavioral effect. Open review point: whether to
run it anyway for completeness.

## 3. Expected outcome (written before running)

- **H1 (probing):** per-layer AUROC rising from ≈lexical floor at layer 0 to a strong
  mid-layer peak, well above every baseline **under held-out-law splits**. Refutation
  condition: probe ≈ bag-of-words baseline out-of-law. Prior in this lab: a
  certificate-gold drift probe reached AUROC 0.96 (different task — model-generated
  drift, task-competent model, verdict-side).
- **H2 (steering):** verdict distribution moves monotonically with steering strength
  while a norm-matched random direction stays flat.

## 4. What we got

| Measurement | AUROC |
|---|---|
| Best probe, `mean` site (layer 28) | **0.623** |
| Best probe, `last` site (layer 32) | 0.599 |
| Layer 0 (embeddings), either site | 0.498–0.503 |
| **Law-disjoint split, best layers** | **0.520 (mean) / 0.527 (last)**, fold range 0.47–0.58 |
| Lexical floor (char TF-IDF on answers) | 0.539 |
| Answer length | 0.505 |
| Shuffled-label control (3 seeds) | 0.491–0.515 |

- **Curve shape:** flat 0.50 at layer 0, monotone rise through depth (mean site: 0.614
  by layer 16, peak 0.623 at layer 28) — the in-distribution signal is computed by the
  network, not lexical.
- **Breakdowns at best layer (mean site):** by tier — easy 0.695, medium 0.584, hard
  0.589. By edit type — prune 0.686, grow 0.648, var_sub 0.589, **arg_swap 0.565**
  (the purely order-semantic edit is the least visible).
- **Verdict vs pre-registration:** under the law-disjoint split the probe sits at the
  lexical floor — the pre-registered refutation condition for H1. **H1 refuted** for
  this model, this reading format, and these sites: the in-distribution 0.62 is
  law-tied signal, not a law-general correctness direction. H2 not run (premise
  removed).
- **Scope limits a reviewer should weigh:** single subject model (base 8B, which is at
  ~0% behavioral competence on this task); passive bare-text reading (no
  verification-framed prompt); two pooled sites only (no perturbation-token-local
  reading); law-disjoint split partially degenerate (556-problem component); probes are
  plain logistic regression on raw activations.

## Artifacts

| Artifact | Path |
|---|---|
| Pre-registration | `probe-experiments/PLAN.md` |
| Dataset + manifest | `probe-experiments/contrast_v1/` |
| Dataset generator / verifier | `probe-experiments/data-gen/` (`VERIFY PASS` required) |
| Capture runner + run records | `probe-experiments/capture/`, `runs/capture-v1/` |
| Probe code + full results JSON | `probe-experiments/probing/`, `runs/probe-v1/probe_results.json` |

Repro: `data-gen/build_contrast.py` → `data-gen/verify_contrast.py` →
`modal run capture/modal_capture.py` → `.venv/bin/python probing/fit_probes.py`.
