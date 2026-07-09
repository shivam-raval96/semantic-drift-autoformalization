# Steering (activation addition) — grumeter card
(steer.py; steer-summary.json + steer-random-summary.json)

**Measures:** the causal effect on certified translation outcomes of adding
±α · (mean drifted − mean faithful activation direction) at one layer, all
positions, with certified endpoints and an in-run α=0 baseline reproduction.

**Valid within:** Llama-3.1-8B, layer 16, the mean-difference direction, α in
units of 5% typical hidden norm, |α| ≤ 6. One direction, one layer, one model,
no bootstrap yet — causal phrasing is licensed only at exactly this strength.

**Positive means:** VBU count moves monotonically along the direction (34→101
over the moderate range) while a norm-matched RANDOM direction is flat with
intact validity — the effect is direction-specific, and "fluent wrongness" is
steerable (+α: fewest invalid, most drift; −α converts silent drift to
faithfulness or loud failure). / **Does NOT mean:** the direction IS "the
drift feature," that it is unique, minimal, or transfers across layers/models.

**Null means:** the mean-difference direction carries no drift-relevant causal
content at this layer/norm — the monitor could still read a correlate.

**Confounds & controls:** generic-perturbation degradation (random-direction
control PASSED); norm effects (α normed to hidden scale, extremes flagged);
hook/CUDA instrumentation errors (α=0-first sanity gate reproduces the
unsteered run before any sweep is trusted).

**Licenses:** "the mean-difference direction carries drift-relevant content
whose addition/removal changes semantic faithfulness" — with the standing
caveats verbatim. **Never licenses:** "we found the drift feature," multi-layer
or cross-model causal claims, or safety conclusions about steering as a fix.
