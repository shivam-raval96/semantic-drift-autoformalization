# Story–Literal Paired Correlation

This analysis pairs Experiment 02 (Story → RG) with Experiment 04 (Structured Literal NL → RG) by `pair_id × model`. No new model calls were made.

## Results

| Sample | Reasoning | N | Story → RG | Structured Literal NL → RG | P(Story → RG ✓ given Literal → RG ✓) | P(Story → RG ✓ given Literal → RG ✗) | Difference | Pair-model phi | Model Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Complex 30 | off | 240 | 21.7% | 37.9% | 41.8% | 9.4% | 32.4% | 0.381 | 0.903 |
| Complex 30 | on | 239 | 95.0% | 99.2% | 94.9% | 100.0% | -5.1% | -0.021 | -0.065 |
| Stratified 40 | off | 320 | 53.1% | 72.2% | 64.9% | 22.5% | 42.5% | 0.381 | 0.945 |
| Stratified 40 | on | 320 | 95.0% | 99.1% | 95.3% | 66.7% | 28.6% | 0.126 | 0.449 |

## Interpretation

- With reasoning off, Literal success is predictive: Story accuracy rises from 9.4% to 41.8% on the complex set, and from 22.5% to 64.9% on the stratified set.
- The corresponding phi coefficients are 0.381 and 0.381, a moderate positive item-model association.
- Across the eight models, Literal and Story accuracy correlate strongly when reasoning is off: Pearson r is 0.903 on the complex set and 0.945 on the stratified set (Spearman 0.970 and 0.946).
- With reasoning on, both tasks are near ceiling. Conditional rates based on the few Literal failures are unstable and should not be interpreted as evidence of no association.
- Literal-correct/Story-wrong cases isolate a likely narrative-abstraction bottleneck; failures on both forms indicate that structural encoding is already difficult without the story.

## Limitations

The model-level correlations use only eight models. The pair-model associations are descriptive: rows from the same model and equation are not statistically independent; a confirmatory analysis should use a mixed-effects model with model and equation effects. The comparison uses Structured Literal because it retains the Story arm's named-intermediate scaffold.

## Data notes

- uniform/on: 1 ungraded literal evaluations excluded
