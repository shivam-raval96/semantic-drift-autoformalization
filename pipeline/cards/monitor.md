# Drift monitor (linear probe) — grumeter card
(probes.py monitor target; geometry2.py §1; gauntlet.py; overopt_eval.py)

**Measures:** linear decodability of *upcoming certified drift* from the
residual stream at a fixed site/layer, before any output token, for ONE model,
on items from the training distribution's family (law-hash or template split).

**Valid within:** the probed model (Llama-3.1-8B for all headline numbers), the
probed site+layer, single-binary-op equational translation items, selection
pressure up to best-of-64. Exploratory numbers: 0.96 big-E (n=533), 0.914
within-register vs BoW 0.540.

**Positive means:** a drift-predictive signal exists in the residual stream and
is linearly accessible at that site; it beats input-only baselines, so it is
not (only) a register/difficulty detector. / **Does NOT mean:** the model
"knows" it is wrong (self-report foil dissociates exactly where competence
ends); transfer to other models, domains, or generation-time optimization
pressure; a causal role (that's the steering card).

**Null means:** no linear read-out at the probed sites — and per Rule 1 a
linear null alone is NEVER "not represented" (probe-site artifact precedent:
the intent null flipped at the mean-pooled site). / **Does NOT mean:** the
signal is absent elsewhere in the network.

**Confounds & controls:** input difficulty (~0.7-0.75 of raw signal — gauntlet
baselines + within-register analysis control it); register leakage (template
split + label-shuffle tripwire); LIBRARY-trained runs (0.804/0.84) carry
mislabeled distractor items in TRAIN (handcheck-2026-07-05.md) — big-E and
fresh-ETP runs are clean.

**Licenses:** "drift is linearly readable pre-generation for this model on this
distribution, beyond surface features" (pending H1 CI). **Never licenses:**
"LLMs know when they're wrong," cross-model monitors, robustness to RL-lite
optimization against the monitor (untested; overopt card).

**Amendment 2026-07-07 (contamination + boundary):** library-trained monitor
variants (0.804 gauntlet / 0.84 first-probe) had mislabeled v1-distractor items
in TRAIN (handcheck-2026-07-05) — treat as exploratory; big-E monitor (0.96)
unaffected. Boundary status: NON-COMPLIANT (emits scores everywhere); fix path =
MAD trusted-support gate on its own inputs -> 'decline to score' (queued v0.1).

**Boundary-gate test 2026-07-07 (mad-v01-report.md): in-domain NULL.** kNN-OOD
gating does not improve accepted-set AUROC on in-distribution items (0.973 ->
0.953 at 50% coverage). Status remains NON-COMPLIANT; the required evidence is
an out-of-scope probe (different fragment/domain through the same gate).
