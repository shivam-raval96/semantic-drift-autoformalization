# 05 — Does Literal NL performance predict Story performance?

## Question

For the same ETP equation pair and model, does success on Structured
Literal NL → Rigid Grammar predict success on Story → Rigid Grammar?  This
tests whether failures already occur in structural encoding, or arise when
the model must first abstract away the narrative.

## Setup

- **Inputs**: committed results from Experiment 02 (Story) and Experiment 04
  (Structured Literal), paired by `pair_id × model`.
- **Samples**: the same 30 complex pairs and 40 operation-count-stratified
  pairs in both experiments.
- **Models**: the same eight models used in Experiments 02 and 04.
- **Regimes**: reasoning off and reasoning on, analyzed separately.
- **Metrics**:
  - model-level Pearson and Spearman correlation between Literal and Story
    accuracy across the eight models;
  - paired outcome counts and phi correlation across pair-model observations;
  - `P(Story correct | Literal correct)` and
    `P(Story correct | Literal wrong)`.
- **Compute**: no model calls; this is a deterministic analysis of existing
  artifacts.

Structured Literal is the primary comparison because it retains the Story
arm's named-intermediate scaffold while removing the narrative setting.

## Reproduce

```sh
python3 paired_analysis.py
```

The command writes Markdown, CSV audit tables, and a self-contained HTML
report to `experiments/05-story-literal-correlation/report/`.

## Results

| Sample | Reasoning | N | P(Story ✓ \| Literal ✓) | P(Story ✓ \| Literal ✗) | Pair-model phi | Model Pearson r | Model Spearman ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Complex 30 | off | 240 | 41.8% | 9.4% | 0.381 | 0.903 | 0.970 |
| Stratified 40 | off | 320 | 64.9% | 22.5% | 0.381 | 0.945 | 0.946 |
| Complex 30 | on | 239 | 94.9% | 100.0% | -0.021 | -0.065 | 0.283 |
| Stratified 40 | on | 320 | 95.3% | 66.7% | 0.126 | 0.449 | 0.555 |

One Experiment 04 complex/reasoning-on API error remains ungraded, so that
condition contains 239 rather than 240 paired observations.

**Vacuous-law-excluded view** (added 2026-07-22). Experiment 07
identified pairs containing a vacuous law (E1 `x = x` / E2 `x = y`) as
a measurement hazard, and the convention is now to exclude them
(`experiments/README.md`). `report/no-vacuous/` holds the same
analysis over filtered copies of the experiment 02 and 04 runs
(`python3 filter_vacuous.py experiments/02-reasoning-and-complexity
experiments/04-structured-literal`, then `python3 paired_analysis.py
--story results/no-vacuous/02-reasoning-and-complexity --literal
results/no-vacuous/04-structured-literal --out
experiments/05-story-literal-correlation/report/no-vacuous`). The
uniform ("Complex 30") rows are unchanged; the stratified rows drop to
N=192 (24 pairs): off-regime association softens but stands (phi
0.381 → 0.327, model Pearson r 0.945 → 0.805), while the stratified/on
association disappears outright (phi 0.126 → −0.013, r 0.449 →
−0.173) — that cell's apparent correlation was carried by the vacuous
pairs, and matches the uniform/on rows' null result.

## Conclusions

- **Reasoning off**: Literal performance strongly predicts Story performance
  across models (`r = 0.903–0.945`) and moderately predicts success at the
  pair-model level (`phi ≈ 0.381`). A Literal failure makes Story success much
  less likely.
- **Bottleneck reading**: Literal-correct/Story-wrong cases identify a likely
  narrative-abstraction bottleneck. Failure on both forms shows that the model
  already struggles with structural encoding without the story.
- **Reasoning on**: both forms are near ceiling. The few Literal failures make
  conditional rates and correlations unstable, so this regime cannot support
  a strong correlation conclusion.
- **Scope**: the model-level result uses only eight models, and pair-model rows
  share models and equations. A confirmatory paper analysis should use a
  mixed-effects model with model and equation effects.
