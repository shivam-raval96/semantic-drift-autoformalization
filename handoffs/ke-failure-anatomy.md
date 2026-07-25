# Ke — Failure anatomy: the valid × faithful 2×2 across models and scale

## Question

When a model's formalization fails, *where* does it fail — and how does
the failure anatomy move with model scale, form (story/literal/two-stage),
and complexity? You own the definitive version of the decomposition chart.

## Why this matters now

Experiment 12 (new, `informalizing-etp/experiments/12-grader-validation/`)
validated the grader and found structure nobody has charted yet: of 3,364
silent failures, **55% are single-equation-local** — the model translated
one law faithfully and drifted on the other (1,091 premise-drifted vs 756
conclusion-drifted — premise drifts 1.4× more often!). 0 direction
confusions, 24 one-sided dualizations. The 2×2 we've been reporting
(parseable × correct) hides all of this. Your experiment 07 scale sweep
(9 models, byte-identical pairs, three arms) is exactly the dataset to
chart it on — no new API calls needed for phase 1.

## What exists to build on

- `experiments/12-grader-validation/validate_grading.py` — offline
  re-grader with the component probes (e_wrong_only / f_wrong_only /
  onesided_dual / structural). Extend it; don't rewrite it.
- Your exp 07 runs (4,320 rows, 9 models) + exp 09 (6,000 rows, deep
  synthetic laws) — all replayable offline, replay fidelity is 100%.
- charts.py axes-composition machinery for the stacked-bar rendering.

## Design

Phase 1 (offline, free): the full anatomy table
  rows: model × form × ops bin
  columns: correct / premise-drifted / conclusion-drifted / both-drifted
           / one-sided-dual / unparseable
Questions with committed data:
1. Does premise-vs-conclusion drift asymmetry (1,091 vs 756) hold per
   model, or is it driven by a few? (ASSUME is rendered *first* — is this
   a serial-position effect or a complexity effect? ops_e vs ops_f
   regression separates them: premise laws aren't systematically harder.)
2. Does the anatomy migrate with scale the way the silent/loud split does
   (silent:loud ≥6:1 at ETP sizes, inverts at depth — exp 12)?
3. Two-stage vs literal: does decomposition change *which* equation
   drifts, or just how much?

Phase 2 (cheap): the 2×2 relabel on new balanced runs at fixed
elicitation (coordinate with Denis so the grids compose rather than
collide — same item set if possible).

**Nulls**: shuffle equation-pair assignment within (model, bin) cells —
the asymmetry should vanish; if it doesn't, it's a grader artifact (tell
me immediately). Bootstrap CIs per cell; cells under 30 rows get pooled,
never silently dropped.

## Kill baselines

- Length/position confound: regress drift side on (ops_e − ops_f) before
  claiming a serial-position effect.
- Any accuracy claim still clears the BoW per-level floors.

## Deliverable

One anatomy chart (stacked bars, model × form, per ops bin) + the
asymmetry regression + a half-page: "what actually breaks when
translation breaks." This slots directly into the NeurIPS paper's
behavioral section as its most novel figure.
