# Story–Literal Paired Correlation

This analysis pairs Experiment 02 (Story → RG) with Experiment 04 (Structured Literal NL → RG) by `pair_id × model`. No new model calls were made.

## Results

| Sample | Reasoning | N | Story → RG | Structured Literal NL → RG | P(Story → RG ✓ given Literal → RG ✓) | P(Story → RG ✓ given Literal → RG ✗) | Difference | Pair-model phi | Model Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Complex 30 | off | 240 | 21.7% | 37.9% | 41.8% | 9.4% | 32.4% | 0.381 | 0.903 |
| Complex 30 | on | 239 | 95.0% | 99.2% | 94.9% | 100.0% | -5.1% | -0.021 | -0.065 |
| Stratified 40 | off | 192 | 39.6% | 62.0% | 52.1% | 19.2% | 32.9% | 0.327 | 0.805 |
| Stratified 40 | on | 192 | 96.9% | 99.5% | 96.9% | 100.0% | -3.1% | -0.013 | -0.173 |

## Interpretation

- With reasoning off, Literal success is predictive: Story accuracy rises from 9.4% to 41.8% on the complex set, and from 19.2% to 52.1% on the stratified set.
- The corresponding phi coefficients are 0.381 and 0.327, a moderate positive item-model association.
- Across the eight models, Literal and Story accuracy correlate strongly when reasoning is off: Pearson r is 0.903 on the complex set and 0.805 on the stratified set (Spearman 0.970 and 0.896).
- With reasoning on, both tasks are near ceiling. Conditional rates based on the few Literal failures are unstable and should not be interpreted as evidence of no association.
- Literal-correct/Story-wrong cases isolate a likely narrative-abstraction bottleneck; failures on both forms indicate that structural encoding is already difficult without the story.

## Limitations

The model-level correlations use only eight models. The pair-model associations are descriptive: rows from the same model and equation are not statistically independent; a confirmatory analysis should use a mixed-effects model with model and equation effects. The comparison uses Structured Literal because it retains the Story arm's named-intermediate scaffold.

## Data notes

- uniform/on: 1 ungraded literal evaluations excluded
