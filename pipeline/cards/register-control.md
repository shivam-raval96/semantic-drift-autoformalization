# Register control probe — grumeter card
(probes.py register target)

**Measures:** probe POWER on this n and this activation set: can any signal at
all be decoded under the same splits? Register (surface form) is trivially
present in the input, so decoding it calibrates the instrument, not the model.

**Valid within:** the same rows/splits as the probes it calibrates. Nothing
else.

**Positive (100% from layer ~4-10 at every scale) means:** the methodology can
find signals of this strength at this n — so a same-n null elsewhere (intent at
final token) is informative. / **Does NOT mean:** anything about meaning,
faithfulness, or model quality. It is a power statement.

**Null means:** the probe pipeline is broken or n is hopeless; nothing else
from that run is interpretable.

**Confounds:** none material — that's the point of a control task.

**Licenses:** "the same-n null on target X is not a power artifact."
**Never licenses:** any substantive claim about the model. Never quote its
accuracy as a finding.
