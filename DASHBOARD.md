# Overnight dashboard — 2026-08-18/19

Live status. Detail and reasoning in `RESEARCH_LOG.md`. Every number below
traces to a committed file under `probe-experiments/runs/`.

## HEADLINE (corrected, validated)

1. **Reader mode: correctness never reaches the verbal channel at any depth.**
   Model's own yes/no readout on passive-reading activations is flat
   (max 0.545 at L59) while a supervised probe on the SAME activations reads
   0.705. `runs/lens-v1/margin_lens.json`
2. **Asked mode: the question recruits it, late.** Same readout climbs through
   the last third: 0.574 (L48), peak **0.6855 (L53)**, 0.6701 at the output.
   `runs/lens-v1/margin_lens_asked.json`
3. **METHOD VALIDATED TWO WAYS.** (a) A reproduce-the-true-logits gate caught a
   real double-normalization bug (HF pre-norms the last hidden state); fixed,
   all analyses recomputed. (b) After the fix the final-layer lens reads 0.6701
   vs the independently measured behavioral gate 0.6694 — agreement to 0.0007
   between two entirely separate measurement paths.
   `runs/lens-v1/validate_lens_qwen3-32b.json`
4. **The model knows more than it says**: probe 0.7247 vs expressed 0.6701 at
   the output; max gap 0.128 at L46. `runs/lens-v1/knows_vs_says.json`
5. **Qwen3-32B — our own model — has a FREE pre-fitted J-lens** (Neuronpedia,
   n=1000). The flagship workspace experiment can run on the exact model all
   our results live on. No cross-model caveat, no expensive lens fit.

## DEMOTED / WEAKENED CLAIMS (honesty ledger)

- "Readout peaks at L53 then declines 0.027 to the output" — after the bug fix
  the decline is 0.015 on 300 texts. NOT currently defensible; pending the
  full 2,000-text rerun + significance test.

## RUNNING

| Lane | Job | Purpose |
|---|---|---|
| Modal CPU | Qwen3-32B J-lens direction extraction | flagship: causal yes/no direction |
| A100 | Qwen3-32B asked capture, full 2,000 | powers the law-disjoint asked analysis |
| A100 | Qwen3.6-27B asked capture, full 2,000 | recruitment curve on the best verifier |
| local | Qwen3.6-27B reader probes | co-emergence point at 0.826 capability |
| agent | skeptical reviewer | trying to falsify the recruitment claim |
| watchdog | 5-min heartbeat over all lanes | early failure detection |

## COMPLETED TONIGHT

- margin lens reader + asked (Qwen3-32B), bug found, fixed, revalidated
- knows-vs-says per-layer probe/readout gap
- lens-formula validation harness (now a permanent gate)
- behavioral margin gates: qwen3.5-4b 0.626, gemma-3-27b 0.602,
  qwen3.6-27b 0.826
- literature recon: J-lens/J-space, R-lens, logit/tuned lens, verified lens
  inventory + architecture warnings

## FAILED / FIXED

- head-row extraction: numpy cannot read bf16 -> read via torch. Fixed.
- asked capture inherited the 300-text gate sample -> now full 2,000 by default.
- lens image build: `debian_slim` has no git for a pip-from-github -> apt git.
- 3.6-27B lens extraction: multimodal wrapper nests weights under
  `model.language_model.*` -> key autodetection. (32B, being dense, is clean.)
- **margin lens double-normalized the final layer** -> caught by the validation
  gate, diagnosed with a discriminating test, fixed, all numbers recomputed.

## FINDINGS + CONFIDENCE

| Finding | Confidence |
|---|---|
| Correctness readable by probe, absent from verbal channel in reader mode | HIGH (2000 texts, validated method, probe cross-check) |
| Question recruits the signal into the channel in the last third | HIGH (same instrument reads flat vs 0.686; final layer matches behavior) |
| Model expresses less than it represents | MODERATE (300 texts; full rerun in flight) |
| Peak-then-decline before output | LOW (0.015 on 300 texts — demoted) |

## OPEN QUESTIONS

1. Is our probe direction orthogonal to the J-lens (verbalization) direction?
   That would mechanically explain the steering null.
2. **Does ANY direction move the margin?** We never ran a positive control for
   the steering harness. The J-lens direction is one by construction — if it
   also fails to move anything, the original steering null is uninformative.
3. Does capability<->representation co-emergence hold across families?

## NEXT

1. Cosine(probe direction, J-lens yes/no direction) per layer — flagship.
2. Steering with the J-lens direction as a POSITIVE CONTROL, alongside our
   probe direction and random.
3. Qwen3.6-27B probes + recruitment curve; co-emergence scatter across roster.
