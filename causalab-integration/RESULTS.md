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

Layer profile (raw fits; per-layer nulls pending — cross-layer MSE
comparison is suggestive only): L7 0.2563, L14 0.2404, L21 0.2468 —
agreement with the certified chart peaks mid-network, consistent with
mid-depth abstraction.

Scope caveats: one model (1.5B), one layer/cell for the null, ~2.4 samples per
centroid, 3D chart carrying 61% of table variance, formal+instance
registers only. Strengthening moves, in order: more null seeds (cheap),
larger n_train (unique-prompt pool supports ~6x), remaining layers,
Llama-1B/8B for cross-model comparison, story register via Storyform.

## Cross-model comparison (Llama-3.2-1B, same protocol)

Llama-1B baseline: 51.2% - AT CHANCE on the judgment task (vs Qwen
59.7%). Yet its law-identity geometry shows the same isometry: TRUE
chart recon_mse 0.2522 vs 20-seed null min 0.2692 / mean 0.2882 -
0/20 nulls beat it, p = 0.048. Effect size matches Qwen almost exactly.

Two readings, both important:
1. Cross-architecture replication: two model families agree with the
   same frozen certified chart - the first evidence the geometry is
   world-shaped rather than representer-shaped.
2. Dissociation: Llama-1B REPRESENTS the certified ontology while
   unable to USE it for the task - the represented-but-unread pattern
   (ELK's central object) observed behaviorally+geometrically at 1B.

Protocol notes: L4/16 quarter-depth cell (analogous to Qwen L7/28);
the law-manifold stage requires a premise_law subspace artifact as
prerequisite (etp_1b_lawsubspace runner; scoring crash on string
labels after metadata write is tolerated).

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

## Dataset geometry (2026-07-25): the premise-strength shortcut

PCA of the certified implication table itself (scripts/dataset_pca.py,
scripts/full_etp_pca.py; figures in analysis/dataset_pca/):

v5 task data (247 laws, 51,381 certified pairs):
- Law fingerprints: PC1-3 = 60.7% var (recomputed PCs correlate 1.000
  with the stored fp3 chart). No complexity clusters (n_ops/depth/family
  silhouettes negative); complexity is a gradient. The dominant axis is
  implication strength: out-degree corr r = 0.48 with PC1.
- Pairs (concat fingerprints, balanced 1,598): truth separates, AUC
  0.945 from 3 PCs, kNN agreement 87% vs 50% null. BUT: premise
  out-degree ALONE predicts truth at AUC 0.9635 (conclusion in-degree
  0.667). A NEW KILL BASELINE: a model can look competent by encoding
  "how strong is the premise law" as one scalar. Future splits must
  stratify by premise strength.

FULL ETP table (4694 laws, 22,028,942 settled implications, from
teorth/equational_theories 2024-11-10 outcomes):
- 8,173,585 True off-diagonal (37% - our v5 subsample's 1.6% True rate
  is a law-selection artifact, worth remembering when balancing).
- PC1 IS out-degree: r = 0.999, 74.4% of variance. PC1-3 = 87.5%.
- Premise-outdeg-only AUC over all 22M pairs: 0.9835.
- Truth from 3 pair-PCs: AUC 0.992.
- Complexity (n_ops 0-4, depth): near-zero correlation with the leading
  PCs, negative silhouettes - at full scale too, logic-space geometry
  organizes by strength, not by syntactic complexity.

Interpretation for the isometry work: the fp3 chart the models align
with is largely a strength-plus-residual geometry; the causal protocol
(causal_patch.py) and any behavioral claims must therefore separate
"represents premise strength" from "represents the specific law" -
the rot/shuf controls and per-strength stratification do this.

## Grader validation (informalizing-etp/experiments/12)

Replay fidelity 10,320/10,320. Silent:loud >= 6:1 at ETP sizes under
EVERY grader variant (strict convention, no-dual, extraction variants);
ratio inverts past ETP depth (loud/grammar collapse takes over). Silent
failures: <1% rescueable by any convention leniency; 55% are
single-equation-local (premise drifts 1.4x more often than conclusion).

## Manifold checks (2026-07-25, scripts/manifold_checks.py, Qwen L14)

C1 dimensionality: participation ratio 2.46 (spectrum 62.9/6.5/4.9%) -
the 247 law centroids genuinely occupy ~2.5 effective dims; 3D chart is
the right order.
C2 held-out generalization (the REAL isometry test): leave-one-law-out
ridge from the certified chart predicts unseen-law centroids at LOO R^2
0.341 while ALL 50 permuted charts score negative (null max -0.003),
p = 0.0196. kNN neighbor preservation 0.151 vs 0.040 null (p = 0.002).
The spline was not memorizing anchors.
C3 linearity: RBF gains only +0.028 LOO R^2 over ridge - the structure
is essentially a LINEAR 3D embedding; write "chart/subspace", soften
"manifold" (the Othello lesson, applied to ourselves).
C4 strength confound REJECTED (closes gap G2 for Qwen): outdeg alone
predicts centroids at LOO R^2 -0.044 (nothing); chart residualized on
outdeg keeps R^2 0.337, p = 0.0196. The alignment tracks law identity,
not the one-scalar dataset shortcut (which would have allowed AUC 0.96).

## Causal patching, screened (analysis/causal/qwen_L14/results.json)

Screening (240 candidates -> 60 most premise-sensitive) works: median
directed donor gap 0.66 logits (was 0.145 unscreened), clean acc 58%.
Verdict: NO patch condition mediates it - full -0.05, chart -0.08 of
the donor gap, rot -0.11, shuf -0.003, flip rate 0 everywhere. Even the
full centroid patch fails while the text-level premise flip succeeds:
the premise information driving the verdict does not flow through the
single (L14, premise_last) cell as a centroid difference.
Combined statement: the certified geometry is PRESENT, GENERALIZES to
held-out laws, and is NOT CAUSALLY READ at this cell at 1.5B. Also
retro-caveats MI-2: KL-transfer 0.99 was measured where base~donor
distributions were close; rerun on the screened subset (gap G4).
Follow-ups: multi-position/multi-layer patching (G1b); Llama-1B
screened run in flight; 8B kit carries --screen 240.
