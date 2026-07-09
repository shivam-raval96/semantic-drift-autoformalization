# Certificate checker — grumeter card
(magma.py: satisfies / find_countermodel / implication_status / classify_relation)

**Measures:** the directional semantic relation between two equational laws over
a single binary operation (equivalent / weaker / stronger / incomparable), via
finite-magma countermodel search.

**Valid within:** magmas of size ≤ 3 exhaustively, size 4 sampled (20k tables);
implications only via two certified routes (construction-known; substitution
instance, σ stored). Outside these bounds the instrument says NOTHING: the label
is `not refuted (<=4)` / `equivalent*`, which is bounded silence, not a verdict.

**Positive (non-implication) means:** a concrete table separates the laws — the
certificate is inline and re-verifiable by satisfies(). It can never be wrong,
only re-checked. / **Does NOT mean:** anything about how "far apart" the laws
are semantically; distance is not measured here.

**`equivalent*` means:** no countermodel found within bound. / **Does NOT
mean:** proven equivalence. Consequence every table inherits: *faithful rates
are upper bounds; certified-drift rates are lower bounds.* (Z3 sweep found 5
size-5 countermodels among pairs unresolved at ≤4 — the bound bites.)

**Confounds:** none for certificates themselves; the risk lives in what is
COMPARED (a mis-parsed model output, a wrong gold — see the 2026-07-05
distractor-gold finding, which the checker faithfully scored as drift because
the gold was wrong, not the checker).

**Licenses:** "certified drift occurred," with direction; drift lower bounds.
**Never licenses:** equivalence claims, implication claims outside the two
routes, anything past size 4.
