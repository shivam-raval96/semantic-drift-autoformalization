# Telephone (iterated translation) — grumeter card
(telephone.py)

**Measures:** persistence of certified meaning under iterated law -> NL -> law
round-trips: survival vs origin, per-hop kernel (equivalent/weaker/stronger/
incomparable/dead), terminal states (attractors). Every hop certificate-checked
both against origin and previous state.

**Valid within:** the 24 library origins, the two fixed prompts (name-free
informalization + the frozen translation prompt), the run's models.
DETERMINISTIC variant (temp 0, exploratory): one chain per (origin, model) —
percentages are over origins, with zero sampling variance; treat as point
observations. SAMPLED variant (H5, temp 0.7, 5 chains/origin): supports CIs
over chains; this is the only variant the prereg quotes.

**Positive (plateau) means:** the hop kernel has absorbing states — drift
freezes, chains reach fixed points. / **Does NOT mean:** meaning is "safe"
after hop 3 in general; attractors are model-specific basins, not a property
of the laws.

**Null (no plateau) means:** iterated translation keeps eroding meaning at
these depths — H5a dies, H5b may still stand.

**Confounds & controls:** the informalization prompt bans standard names —
survival partly measures compliance with that ban; equivalence at ≤4 bound
means "alive" inherits the equivalent* asymmetry (a chain could drift
undetectably past the bound); hop-1 conflates one-shot translation error with
iterated dynamics (compare hop-1 to single-shot rates before attributing to
iteration).

**Licenses:** "meaning has a measurable, model-specific half-life under
iterated translation; the drift kernel has absorbing states" (pending H5a/b).
**Never licenses:** claims about NL-only chains, other prompt framings, or
attractor identity generalizing across model families.
