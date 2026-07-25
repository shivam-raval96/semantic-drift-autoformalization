# Handoffs — four scoped experiments (pre-sprint package)

One brief per person. Each is self-contained: question, why it matters now,
what already exists to build on, design with controls and kill baselines,
and the deliverable. The common method norms apply to all four:

- **Certify or exclude**: every implication label comes from a certificate
  (countermodel or construction); uncertified pairs never enter a split.
- **Balanced splits, stratified per complexity level** (and now also per
  premise strength — see the new confound below).
- **Leave-pair-out / leave-law-out** when anything is fit; pooled numbers
  are reported only next to their stratified versions.
- **Run dirs are immutable**; new prompt = new experiment directory.

## A dataset fact everyone needs (new, 2026-07-25)

PCA of the certified implication table shows pair truth is almost entirely
predictable from **premise strength alone**: out-degree of the premise law
gives AUC 0.9635 on our v5 task data and 0.9835 on the full 22M-pair ETP
table (`causalab-integration/analysis/dataset_pca/`). Consequence: any
model score on a True/False implication task must be compared against the
premise-outdeg-only baseline, and future splits should stratify by premise
strength so the shortcut cannot carry the task.

## The briefs

- `denis-elicitation.md` — elicitation specification curve for judgment evals
- `harsh-sql-register.md` — the SQL register: executable semantics for laws
- `ke-failure-anatomy.md` — the valid × faithful 2×2, per model and per hop
- `oren-two-hop-attribution.md` — gold-conditioning ablation for two-stage
