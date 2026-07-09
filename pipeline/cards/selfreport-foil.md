# Self-report foil — grumeter card
(selfreport.py; Rule 2: this channel NEVER feeds a monitor)

**Measures:** how well a model's elicited confidence about its own translation
predicts its certified faithfulness (AUROC of self-report vs certificate
verdict). It exists as a FOIL: the thing latent-space monitors must beat and
must not secretly be.

**Valid within:** the one fixed elicitation template; models that answer in
prose degrade to default confidence 5 (recorded raw). Positive classes were
small on the E-arm (14/9/2 VBU) — exact AUROCs provisional, the collapse
robust.

**Positive means:** the model's elicited channel tracks its errors on that
distribution (library items: 0.69-0.81). / **Does NOT mean:** introspection —
it may be difficulty heuristics.

**Below-chance (sonnet 0.46 on unnamed-ETP while 85% faithful) means:**
elicited report and competence dissociate exactly where familiarity ends. /
**Does NOT mean:** the model represents nothing about its errors (the monitor
reads a signal the elicited channel does not carry — that contrast is the
finding).

**Confounds:** parse failures -> default confidence (audited in raw field);
one template only.

**Licenses:** "monitors must not rely on elicitation; the elicited channel
collapses off-distribution." **Never licenses:** claims about model
consciousness/introspection, or that self-report is useless on-distribution.
