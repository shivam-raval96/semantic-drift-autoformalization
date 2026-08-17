# Overnight dashboard — 2026-08-18/19

Live status. Updated as things change. Detail lives in `RESEARCH_LOG.md`.

## HEADLINE (what changed tonight)

1. **Reader mode: correctness never reaches the verbal channel, at any depth.**
   Model's own yes/no readout on passive-reading activations is flat 0.50-0.55
   across all 65 layers, while a supervised probe on the SAME activations reads
   0.70. `runs/lens-v1/margin_lens.json`
2. **Asked mode: the question performs a late routing.** Same readout rises from
   0.50 to **0.686 at layer 53**, then DECLINES to 0.659 at the output layer.
   `runs/lens-v1/margin_lens_asked.json`
3. **The model knows more than it says.** Supervised probe on asked-mode
   activations: 0.72-0.74. Model's own expressed readout: 0.659. Gap up to 0.21
   mid-stack. `runs/lens-v1/knows_vs_says.json`
4. **Qwen3.6-27B is a much better verifier: margin AUROC 0.826** (vs Qwen3-32B
   0.669) AND has free pre-fitted J-lens + R-lens with identical geometry
   (65x5120). It is now the flagship model for the workspace experiment.

## RUNNING

| Lane | Job | Purpose |
|---|---|---|
| A100 | Qwen3.6-27B reader capture (2000 texts) | flagship probe substrate |
| A100 | Qwen3.6-27B asked capture | recruitment curve on a capable verifier |
| A100 | Qwen3-32B asked capture, FULL 2000 texts | fixes underpowered 300-text version |
| A100 | gemma-3-27b margin gate | roster/co-emergence point |
| A10G | qwen3.5-4b margin gate | roster/co-emergence point |

## AGENTS

- model-roster scout: verifying which models have free pre-fitted lenses + what
  recent interpretability papers use as baselines.

## COMPLETED TONIGHT

- margin lens, reader mode, Qwen3-32B (all 65 layers) - flat
- margin lens, asked mode, Qwen3-32B - peak L53, decline to output
- knows-vs-says probe/margin gap per layer
- Qwen3.6-27B margin gate 0.826
- head-row extraction (norm + yes/no lm_head rows) for lens work

## FAILED / FIXED

- head extraction crashed on bf16 via numpy -> switched safetensors framework
  to torch, then `.float()`. Fixed, rerun clean.
- asked-mode capture inherited the 300-text gate sample (underpowered for
  law-disjoint splits) -> now full 2000 by default, `--gate-sample` to reproduce
  the old subset.

## OPEN QUESTIONS (priority order)

1. Is our correctness direction outside J-space (the model's verbalization
   workspace)? That would explain the steering null mechanically.
2. Why does the model's own readout PEAK at L53 and decline by L64? Is
   information actively discarded before the output?
3. Does capability<->representation co-emergence hold across model families?

## NEXT

1. Qwen3.6-27B probes (reader + asked) once captures land.
2. Download pre-fitted J-lens + R-lens for Qwen3.6-27B; compute the J-lens
   yes/no direction; measure cosine against our probe direction; steer both.
3. Steering on Qwen3.6-27B (a capable verifier - our 32B steering null may have
   been confounded by weak capability).
