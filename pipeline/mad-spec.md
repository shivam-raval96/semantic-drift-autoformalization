# MAD-ETP benchmark spec v0 (FROZEN before first detector run; changes = new version)

## Definitions
- **Anomaly** (extensional): a translation whose output is certified-non-equivalent
  to the intended law (VBU). L-code failures are a third class, EXCLUDED from v0.
- **Trusted set**: certified-faithful items — provably clean, not assumed clean.
- **MAD-legality**: a detector may use (a) the trusted set with its activations,
  (b) the eval input+activations. It may NEVER see anomaly labels or certificates
  at fit time. Certificates score, never train.
- **Skyline** (illegal, reported for context): supervised certificate-trained probe.

## Data (v0)
acts-geo.npz + results-geo.jsonl (Llama-3.1-8B, dual sites, layers step-2).
Big-E items only for R1/R2/R4 (ladder items enter in R3). Old-distractor-register
items are excluded everywhere (handcheck-2026-07-05: mislabeled gold).

## Regimes
- **R1 clean**: trusted = faithful E-items in law-hash train partition; eval =
  held-out partition (natural VBU mixture).
- **R2 contaminated**: R1 with 10% certified-VBU injected into trusted (labels
  still hidden); measures robustness of "trusted" assumptions.
- **R3 shifted**: trusted = faithful ladder A0/A1 items; eval = held-out E items.
  Mechanism detection under representation shift.
- **R4 rare**: R1 eval subsampled to 5% anomaly base rate (operational regime).

## Metrics
AUROC; **recall@1%FPR** (headline for R4); risk-coverage curve (selective
prediction: sweep score threshold, plot certified-error-rate vs coverage) — the
abstention/'honest silence' axis, grumeter-compliant. Cluster bootstrap by item
for CIs. v1 will add explicit out-of-scope marking distinct from the anomaly
score; v0 abstention = score-threshold silence.

## v0 detector roster
MAD-legal: Mahalanobis (per layer/site, shrinkage-regularized), trusted-PCA
residual energy, kNN distance (k=5). Floors: random; surface-only baseline
(BoW distance to trusted surfaces — input-difficulty control, the gauntlet
lesson). Skyline: supervised probe. Behavioral k-sample floor deferred to v0.1
(needs multi-sample acts on big-E).

## Grumeter card required per detector
scope / fit data / in-scope scores / boundary rule / out-of-scope behavior.
