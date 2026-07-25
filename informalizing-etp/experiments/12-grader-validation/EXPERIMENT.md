# Experiment 12 — Grader validation: V-A specification curve, V-B rescue grading

## Question

The silent-vs-loud decomposition (wrong-but-parseable vs unparseable) is the
project's behavioral headline. Both halves depend on our own grader. Does the
headline survive the grader's degrees of freedom (V-A), and are the "silent"
failures genuine semantic drift rather than near-misses a slightly more
lenient grader would rescue (V-B)?

## Setup

No new model calls. All 10,320 committed responses from experiments 07
(two-stage-scale, 4,320 rows, ETP-sampled laws) and 09 (synthetic-complexity,
6,000 rows, synthetic laws to depth 8) are re-graded offline by
`validate_grading.py` (deterministic, stdlib + repo modules only).

- **V-A variants**: `replay` (exact checkform), `strict_convention` (no
  accepted symmetries), `no_dual`, `first_line` extraction,
  `plain_lines` extraction.
- **V-B probes** on default-wrong rows (deliberately illegal rescues):
  direction swap, one-sided dualization, per-equation component check
  (ASSUME-only-wrong / ASK-only-wrong), remainder = structural.

Reproduce: `python3 experiments/12-grader-validation/validate_grading.py`

## Results

Full tables in `summary.md` / `report.json`. Key findings:

1. **Replay fidelity 10,320/10,320** — stored verdicts reproduce exactly;
   the grading pipeline is deterministic end to end.
2. **The silent≫loud claim survives every variant, with its scope made
   precise.** At ETP-realistic sizes (ops_total ≤ 9, bins 0–3) silent:loud
   is 6.6–22:1 under replay and stays ≥5:1 under every grader variant.
   Pooled over ALL rows including deep synthetic laws it is only 2.1:1,
   because loud (grammar collapse) takes over at depth — bin 7 is 639/893,
   below 1:1. The correct claim is therefore conditional: *silent failure
   dominates at ETP complexity; loud failure dominates past it.* State it
   with the bin scope, never pooled.
3. **Extraction rules are immaterial** (first-vs-last line, markdown
   tolerance: ranking rho = 1.000, counts move <2%). The one consequential
   choice is `strict_convention` — refusing the accepted symmetries
   reclassifies 1,502 correct→wrong (mostly side swaps) and perturbs model
   ranking to rho 0.867. Those symmetries are argued from the story
   semantics (checkform docstring); this quantifies what the argument is
   worth.
4. **V-B: silent failures are real drift, not grader strictness.** Of 3,364
   wrong rows: 0 are direction confusions, 24 are one-sided dualizations —
   together <1%. The rest: 1,091 premise-drifted (ASSUME wrong, ASK right),
   756 conclusion-drifted, 1,493 structural (both wrong). ~55% of silent
   failures localize to exactly one equation — drift is component-local,
   not global scrambling. Structural share is stable across forms
   (literal 44%, story 48%, two-stage 40%).

## Conclusions

The 8:1 headline needed a scope qualifier, and now has one backed by a
specification curve: **silent:loud ≥ 6:1 at ETP sizes under every grader
specification tried; the ratio inverts at synthetic depth as failures go
loud.** The silent failures themselves are overwhelmingly genuine semantic
drift (99% not rescueable by any convention leniency), and more than half
are single-equation-local — a new, more precise anatomy: models usually
translate one law faithfully and drift on the other, rather than garbling
both.
