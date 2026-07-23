# ETP implication task

Judgment task over certified magma-law implications, built for the MARS V
semantic-drift project. A premise law and a conclusion law are rendered in a
register and the model answers True/False; ground truth is a certificate
lookup, never a judge.

## Variables

- `premise_law`, `conclusion_law` — law ids from the vendored library
  (21 non-degenerate laws; degenerate/vacuous laws excluded per the
  project's convention).
- `template` — the register: `formal` (equation string), `instance`
  (fully explicit prose, mechanical or hand-written), `paraphrase` and
  `named` (hand-written subset only; `named` names the law and carries a
  memorization confound — excluded from default sampling).
- `implication` — certified truth of "premise implies conclusion".

## Counterfactual generators

- `random_counterfactual` — balanced independent resampling (baseline, locate).
- `flip_premise` / `flip_conclusion` — swap one law so the certified label
  flips, holding everything else; the causal test that a located
  representation carries law identity into the implication computation.
- `cross_register` — same law pair, different register; interchange on the
  premise/conclusion spans tests representation invariance across surface
  forms.

## Data

`data/etp_pairs.json` — 369 certified ordered pairs (7 implies, 362
non-implications; ~51 uncertified pairs excluded, never guessed). The True
side is thin by construction: certified implications among non-degenerate
distinct laws are rare (substitution instances and construction-known
routes). Balance is enforced at sampling time. Provenance and regeneration:
see the JSON's `provenance` field and `config.py`'s docstring; source of
truth is the semantic-drift-autoformalization repo, branch
`certificate-pipeline` (`pipeline/laws.py`, `pipeline/magma.py`).

## Integration constraint: certified sampling only

`task.resample_variable` must remain `"all"`. The framework's
single-variable resampling (used by `locate` pairwise mode) bypasses the
task's generators and samples law values freely, which can produce
uncertified pairs; the `implication` mechanism raises `KeyError` on those
by design — certify or exclude, fail loud, never guess. To integrate
`locate`, either (a) add a runner hook that consumes the task's
`flip_premise` generator (it already yields pairs differing only in
`premise_law`, all certified), or (b) write a certified-aware resampler
that restricts the resampling domain per conclusion. Option (a) is the
intended path.

## Research mapping (MARS V)

- `locate` + `flip_premise`: where does law identity causally live?
- `cross_register` interchange: is the representation register-invariant
  (the story/literal/formal question in miniature)?
- `subspace`/`activation_manifold` on `premise_law`: geometry of the law
  ontology against the certified metric.
