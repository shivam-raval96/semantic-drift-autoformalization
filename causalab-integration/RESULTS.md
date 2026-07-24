# Shakeout results — etp_implication on Qwen2.5-1.5B-Instruct

Run 2026-07-24 on a MacBook (MPS), causalab task `etp_implication`,
data v5 (sha 5e7f69b0), runners `etp_qwen15_manifold` /
`etp_qwen15_lawmanifold`. Artifacts under the causalab checkout at
`artifacts/etp_implication/qwen25_15b_instruct/`.

## Behavioral

- Baseline 59.7% (191/320) on the balanced, complexity-stratified
  True/False implication task. An open 1.5B is above chance; the
  per-level surface floors (0.50-0.69 BoW leave-pair-out) remain the
  bar for any probe claim.

## Causal localization

- Interchange interventions in a PCA-64 subspace at the premise span's
  final token transfer the counterfactual output distribution at ~0.99
  (KL metric) at layers 7, 14, and 21. The cell is causally
  load-bearing and low-dimensional. The metric is target-independent:
  this is not a decoding-accuracy claim.

## Fingerprint isometry (first measurement)

Thin-plate spline through 136 law centroids at L7/premise_last,
parameterized by the certified-fingerprint 3D chart (fp3; 60.7% of
implication-table variance):

- TRUE chart:  recon_mse 0.2563, residual 3.064
- 50 embedding-shuffle nulls: mse min 0.2658, mean 0.2919, max 0.3367
- Nulls beating the true chart: 0/50 -> empirical p = 0.0196

Reading: Qwen2.5-1.5B's premise-law representations are organized in
partial agreement with Lean's implication metric, beyond what spline
capacity explains. Effect is modest (true beats the best null by ~5%,
the mean by ~12%); p = 0.0196 with a 50-seed null (true chart beat every shuffle).

Scope caveats: one model (1.5B), one layer/cell, ~2.4 samples per
centroid, 3D chart carrying 61% of table variance, formal+instance
registers only. Strengthening moves, in order: more null seeds (cheap),
larger n_train (unique-prompt pool supports ~6x), remaining layers,
Llama-1B/8B for cross-model comparison, story register via Storyform.

## Ops lessons (encoded so they are not relearned)

1. Verify a run ran: causalab skips completed output dirs, and
   shuffled runs write to `_shufN`-suffixed dirs. Both masqueraded as
   results once. Check for a training log line, not just exit 0.
2. Base LMs ignore the answer format; use instruct models.
3. subspace/activation_manifold need explicit `layers` while `locate`
   stays deferred (certified-resampling constraint).
4. Interchange needs fixed-arity positions: use `premise_last` /
   `conclusion_last`, never the variable-length spans.
5. Verdict (1D) and law identity (3D) need separate runners:
   `intrinsic_dim` must match the embedding dimension.
