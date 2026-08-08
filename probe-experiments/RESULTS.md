# Probing: is translation correctness linearly represented?

## Experiment

Llama-3.1-8B-Instruct reads a story (informalized ETP implication) plus a candidate
rigid-grammar answer. Question: can a linear probe on its hidden states separate correct
from wrong answers, and does that hold on unseen equations? Pre-registered in `PLAN.md`.
Background: this model scores ~0-1% when *producing* these translations
(`ft-experiments/runs/base-v1`), so this tests whether evaluation exists internally even
where generation fails.

## Data creation (contrast_v1)

```
sample 1,000 implication pairs      seed 0, vacuous excluded
  easy 334 (ops 2-4, ETP) | medium 333 (5-8, ETP) | hard 333 (10-12, genform)
        v
render story (repo renderer)        back-parse must recover the source laws
        v
correct answer = reference RG       must re-grade "correct", identity transform
        v
wrong answer = ONE AST edit         arg_swap 226 / var_sub 274 / prune 247 / grow 253
        v                           must re-grade well-formed "wrong"
        v                           193 edits inside grader symmetries: rejected, redone
freeze + verify                     manifest, sha256, 2,000 texts, exact 50/50
```

Labels are mechanical end to end (checkform only, R1).

## Setup

| | |
|---|---|
| Model | meta-llama/Llama-3.1-8B-Instruct, bf16, read-only (no generation) |
| Input | bare text: story + blank line + answer, no chat template |
| Captured | residual stream, all 33 layers, two sites (last answer token; mean over answer tokens) |
| Probe | StandardScaler + logistic regression, per layer per site |
| Splits | primary: 5-fold GroupKFold by problem. Strict: by law component (largest holds 556/1000) |
| Controls | char TF-IDF on answers, answer length, layer 0, shuffled labels x3 |
| Frozen | dataset sha-pinned, capture bound to that sha, all seeds and probe config fixed |
| Steering | not run: pre-registered direction comes from the best probe layer, which did not generalize (below) |

## Expected (written before running)

H1: probe well above all baselines and stable on held-out laws. Refuted if it matches
the bag-of-words floor out-of-law. Lab precedent on a related task with a competent
model: 0.96 AUROC. H2: steering shifts verdicts monotonically, random direction flat.

## Result

![AUROC by layer](runs/probe-v1/auroc_by_layer.png)

In-distribution the probe reaches 0.623 (mean site, layer 28) and 0.599 (last site,
layer 32), above the 0.539 lexical floor; layer 0 is 0.50 and controls are clean
(shuffled 0.49-0.52, length 0.505). On the law-disjoint split it drops to 0.52-0.53,
i.e. the lexical floor: the signal is tied to specific equations, not a general
correctness direction. At the best layer: easy 0.695 vs medium/hard ~0.59; size-changing
edits most visible (prune 0.686, grow 0.648), the order-semantic arg_swap least (0.565).
**H1 refuted under the pre-registered condition. H2 dropped (premise removed).**
Full numbers: `runs/probe-v1/probe_results.json`.

## Open questions

1. Does the grammar-FT checkpoint (Phase 5a adapter) develop the direction the base model lacks?
2. Does verification framing ("is this translation correct?") surface it vs passive reading?
3. Is the signal local to the edited tokens rather than the pooled sites we read?
4. Worth running steering at layer 28 anyway as a completeness check?

Repro: `data-gen/build_contrast.py` -> `data-gen/verify_contrast.py` ->
`modal run capture/modal_capture.py` -> `probing/fit_probes.py` -> `analysis/make_figure.py`.
