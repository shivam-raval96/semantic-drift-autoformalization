# Intent probe (law identity) — grumeter card
(probes.py intent target; geometry2.py §2, ladder protocol)

**Measures:** decodability of WHICH law a surface expresses, from residual-stream
activations, across a register/rung shift the probe never saw (train three
rungs, test the held-out rung; 60 classes, chance 0.017).

**Valid within:** the probed model + sites; ladder rungs A0-A3 as rendered
(A2 admission gated on renderer rework); classes present in training (a
held-out LAW is an unseen class — 0% by construction, which is why intent uses
register/rung splits, never law splits).

**Positive means:** law identity is represented and linearly accessible at that
site, robust to that rung shift (mean-pooled: 0.983/0.783/0.317 across
A1/A2/A3). / **Does NOT mean:** the model will translate faithfully (decoding
and behavior can dissociate), or that the representation is "used."

**Null means:** not decodable at that site — the final-token null that flipped
at the mean-pooled site is the standing warning: single-site nulls can be
probe-site artifacts. / **Does NOT mean:** meaning is absent from the model.

**Confounds & controls:** rung-correlated lexical features (the held-out-rung
protocol is the control: the probe must generalize across a surface change it
never saw); layer selection by train-rung CV only (verified no test peeking).

**Licenses:** "meaning-geometry exists, is linear, and degrades with ambiguity
rung" (pending H3 CI + permutation). **Never licenses:** claims about causal
use, other domains, or rungs A4/A5 (templates not yet signed off).
