# Template banks v1/v2 — grumeter card
(laws.py NL_TEMPLATES = v1; templates_v2.py = v2; handcheck-2026-07-05.md = authority)

**Measures:** nothing themselves — they are the STIMULI. Cards exist because
stimulus validity bounds every downstream claim.

**Valid within:** v1 = exploratory tier only (frozen, hash e41a158babcfec38);
known defects kept for reproducibility: distractor register UNWINNABLE by
construction (gold bug, caught in hand-check), invented canonical names
(unipot/labsorb/rabsorb collisions). v2 = confirmatory tier (hash
048d8ec6e4a35108): per-register status; dropped/obscure-name/rewritten fields;
distractor well-posed (gold = intended, neighbor vocabulary).

**Positive/null means:** n/a — but any rate computed on v1 distractor items or
v1 obscure-name canonicals is measuring the stimulus bug, not the model.

**Confounds & controls:** obscure-name stratum NEVER pooled into faithful rates
(v2 run: 0/4 faithful across all models — quarantine validated); v1->v2 changes
are conscious acts pinned by separate freeze hashes.

**A2 renderer gate (2026-07-07): PASS.** Short lowercase entity names ("ana",
"bo", "cy", "dee") drop A2 L-fail 53% -> 13.3% (60-call pilot, llama+4o-mini,
gate <20%). Renderer swap in ladder_items.py is a stimulus change = conscious
act: apply at next ladder regeneration, new data files, note in prereg.
