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

`data/etp_pairs.json` (v3) — 157 laws (21 base + 32 synthetic laws
generated per op-count bin 1-8, genform-style + 104 derived instance
laws, each a single-variable identification whose parent -> child
implication certifies via the substitution route) and 20,563 certified
ordered pairs (398 implies, 20,165 non-implications; 3,929 uncertified
pairs excluded, never guessed). Every law carries `n_ops` and `depth`;
pairs stratify into four complexity bins by total op count (OPS_BINS:
0-3 / 4-7 / 8-11 / 12+), each bin containing both labels
(True capacity per bin: 31 / 189 / 117 / 61 pairs).

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
between premise and conclusion partially predicts the label — on v3 a
bag-of-words classifier reaches leave-pair-out AUROC ~0.70 overall,
but per complexity bin 0.53 / 0.66 / 0.66 / 0.55 (the overall number is
inflated by bin composition: True concentrates at low complexity and
bin is visible via prompt length). This is the NLI lexical-overlap
confound in miniature. Any claim that a probe reads *implication* must
beat the per-bin surface floor under a leave-pair-out split, within
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
