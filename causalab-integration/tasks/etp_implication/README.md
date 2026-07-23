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

`data/etp_pairs.json` (v4) — 247 laws (21 base + 32 synthetic laws per
op-count bin 1-8, genform-style + 104 single-variable-identification
instances + 90 compound-term substitution instances, the latter filling
the odd/high op levels the identification route cannot reach) and
51,381 certified ordered pairs (799 implies, 50,582 non-implications;
9,381 uncertified pairs excluded, never guessed). Every law carries
`n_ops` and `depth`; pairs stratify into EIGHT width-2 complexity
levels by total op count (2-3 ... 16+, experiment-07 style), every
level holding >= 15 True pairs (capacity: 41/143/201/193/108/61/37/15),
so 20 items per level with exact label balance is supported everywhere
(generate_dataset n=320 yields exactly 20 True + 20 False per level,
all prompts unique).

**Balance and stratification:** `generate_dataset` samples without
replacement over unique prompts, exactly label-balanced within every
complexity bin (per-cell k = min(n / (2 x bins), capacity)) — the set
that survives causalab's dedup stays balanced and stratified. Capacity
caps requested n; balance wins over size. Caveat:
causalab draws train (seed) and test (seed+1) independently from the
same pool, so prompt overlap across train/test is possible — grouped
(leave-pair-out) evaluation splits are mandatory for any probe claim,
which also covers this.

**Known surface floor (measure before trusting any probe):** because
derived children textually resemble their parents, lexical overlap
between premise and conclusion partially predicts the label — on v4 a
bag-of-words classifier reaches leave-pair-out AUROC ~0.71 pooled, and
per level 0.55/0.62/0.67/0.69/0.65/0.52/0.59/0.50 (pooled is inflated
by level composition). This is the NLI lexical-overlap confound in
miniature. Any claim that a probe reads *implication* must beat the
per-level surface floor under a leave-pair-out split, within
complexity strata — never the pooled 0.5. Provenance and regeneration:
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
