# Overoptimization selector — grumeter card
(overopt.py sampling; overopt_eval.py best-of-n; overopt_pressure.py n=64 —
all mentors' tree, synced on request)

**Measures:** what happens when the learned monitor is used as a SELECTOR over
k sampled candidates: certified faithfulness of monitor-picked vs random vs
oracle picks, as selection pressure n grows.

**Valid within:** Llama-3.1-8B candidates (temp 0.8), selection up to n=64,
candidate-level monitor (mean output-token acts, L16), 34 test items at n=64
(plateau shape robust, exact levels provisional).

**Positive means:** monitor selection lifts certified faithfulness (+15pts at
n=12) — the monitor has usable signal under mild pressure. The widening
monitor-oracle gap (0.06→0.15) means the monitor SATURATES: it separates
clearly-bad from plausible but cannot rank the top tail. / **Does NOT mean:**
Goodhart-proof; "exhausted, not Goodharted" is the exact exploratory phrasing —
no inversion observed AT THESE PRESSURES.

**Null/inversion would mean:** the monitor's signal is an optimization target
that collapses under pressure — the classic proxy failure.

**Confounds & controls:** selection (best-of-n) is the WEAKEST pressure form —
optimizing generation against the monitor (RL-lite) is untested and explicitly
outside this card's bounds; small test-item count at n=64.

**Licenses:** "the monitor survives selection pressure to n=64, saturating
rather than inverting." **Never licenses:** robustness to generation-time
optimization, deployment claims, or any statement about pressures beyond n=64.
