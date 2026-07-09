# The Ambiguity Ladder — design note (2026-07-05, Luiza + Claude)

Goal: an ORDERED scale of representations of the same formal intent, from
unambiguous to highly ambiguous — the project's analogue of Goodfire's days-of-week
manifold (Shivam's advice: find inherent orderings; Ke's S2/S3 slides). Feeds the
benchmark one-pager (Luiza+Shivam action item) and the geometry experiments.

## Definition that makes ambiguity measurable here
A surface is ambiguous to the extent that its VALID formalizations are
NON-EQUIVALENT. This is AmbiEnt's operationalization (ambiguity = multiple
plausible labels under different readings) transplanted into a domain where
"different reading" is decidable: our checker certifies whether two formalizations
mean the same thing. No NLI model, no LLM judge — the team's judge-skepticism
(June 21 sync) is satisfied by construction.

## Primary metric: certified semantic entropy (CSE)
Semantic entropy (Kuhn/Gal/Farquhar) samples k generations, clusters them by
meaning, and takes entropy over clusters — but their clustering uses NLI, which is
noisy. Ours uses the certificate checker: sample k formalizations of a surface
from an ENSEMBLE of translator models (not the model under test), partition by
certified equivalence, compute entropy over the partition.
- CSE(surface) = H(certified-meaning distribution). 0 = everyone converges on one
  meaning; high = the surface underdetermines the law.
- Ensemble-based CSE is an INPUT property (aleatoric, in the input-clarification
  literature's terms), separated from any single model's incapacity (epistemic).
  This is the decomposition the uncertainty literature struggles to make crisp —
  decidable equivalence makes it crisp.
Secondary metrics: disagreement rate (1 - modal share; simpler), number of distinct
certified meanings (support size), parse-validity rate per rung (L-floor),
cross-model certified agreement, and a small AmbiEnt-style human pass on the
hand-check sample (annotators list readings; feeds the gold-standard goal).

## The ladder (each rung = a renderer over the same ETP equation)
| rung | representation | ambiguity type introduced | example for (x*y)*y = x |
|---|---|---|---|
| A0 | formal string | none (transcription control) | `(x * y) * y = x` |
| A1 | parenthesis-spelled instance | none (explicit grouping) | "(a combined with b) combined with b equals a" |
| A2 | fuzzified entities (Luiza's June-21 proposal) | lexical noise, structure still explicit | "Ana's item merged with Bo's, then merged with Bo's again, gives back Ana's" |
| A3 | grouping-implicit instance | SYNTACTIC/scope: grouping unstated | "a combined with b combined with b equals a" |
| A4 | structural paraphrase | vagueness of prose | "combining with the same thing twice returns the original" |
| A5 | intensional description | pragmatic/definite-description | "the law where repeating a combination undoes it" |

Notes:
- A3 is the sharpest rung: ambiguity is introduced by DELETING grouping cues, so
  each A3 item has an enumerable set of readings (all bracketings) and its CSE has
  a combinatorial prediction — a rare case where measured ambiguity can be checked
  against ground-truth reading counts. (Sonnet's re-nesting failures on A1-style
  items suggest models resolve implicit grouping with a systematic bias — the
  ladder turns that anecdote into a measurement.)
- A0-A2 should have CSE ~= 0; the ladder is VALIDATED by checking CSE is
  monotone in rung index before any downstream use (if not, reorder or repair
  rungs — the scale must earn its ordering empirically, not by declaration).
- Rung renderers are mechanical extensions of etp_items.py (A1 exists; A3 is A1
  minus parentheses; A2 is A1 plus an entity lexicon; A4/A5 need template banks —
  the only rungs needing human template sign-off).

## What the ladder unlocks
1. **Drift vs ambiguity curve**: certified drift rate as a function of CSE —
   separates "model fails on ambiguous input" (defensible behavior) from "model
   fails on unambiguous input" (true unfaithfulness). Blame assignment becomes
   quantitative: aleatoric vs epistemic drift.
2. **Monitor stratification**: does the 0.84 drift monitor track input ambiguity
   (then it's a CSE estimator — still useful, less interesting) or target-specific
   failure at fixed CSE (the gauntlet's within-register logic, now graded)?
3. **Geometry (the days-of-week analogue)**: same equation rendered at A0..A5 —
   does the rung index trace an ordered 1-D trajectory in activation space (PCA,
   per team norms; no t-SNE/UMAP)? If yes: an "ambiguity direction" exists; steer
   along it and watch certified drift respond. This is the S2/S3 experiment shape
   with our variables.
4. **Abstention evaluation**: at high CSE the RIGHT behavior is a clarifying
   question or hedged output, not a confident formalization. The ladder gives the
   first benchmark where over-confident disambiguation is provable (AmbiEnt found
   GPT-4 disambiguates correctly only ~32% of the time — this failure mode is
   frontier-relevant).

## Fit with the June syncs
- "Benchmark the problem first" (team decision): the ladder IS the benchmark axis —
  drift rate x ambiguity level x domain.
- Round-trip critique (Oren: round-trip conflates formalization and
  informalization skill): CSE avoids round-trips entirely; our telephone
  experiment measures the round-trip dynamics separately.
- Gold-standard-with-ambiguous-inputs (June 21 decision): rungs A3/A5 items with
  enumerated readings + certified labels per reading are exactly that dataset.
- Statement-only principle (June 30 decision): every rung formalizes statements;
  no proofs anywhere.

## Sources
- [Semantic Uncertainty (Kuhn, Gal, Farquhar 2023)](https://arxiv.org/abs/2302.09664);
  [Nature 2024 hallucination-detection follow-up](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html);
  [Semantic Entropy Probes](https://arxiv.org/pdf/2406.15927) (cheap SE from hidden
  states — directly relevant to our monitor: an SE-probe is a natural baseline).
- [AmbiEnt / "We're Afraid LMs Aren't Modeling Ambiguity" (Liu et al 2023)](https://arxiv.org/abs/2304.14399);
  [A Taxonomy of Ambiguity Types for NLP](https://arxiv.org/html/2403.14072v1)
  (rung-type vocabulary: lexical/syntactic/scopal/pragmatic).
- [GYAFC formality corpus](https://arxiv.org/abs/1803.06535) (graded human scale
  methodology; formality is a sibling ordered axis, not ambiguity itself).
- [Input Clarification Ensembling (aleatoric vs epistemic for LLMs)](https://arxiv.org/abs/2311.08718);
  [The Anatomy of Uncertainty in LLMs](https://arxiv.org/abs/2603.24967) (three-way
  split: input ambiguity / knowledge gaps / decoding randomness — our CSE isolates
  the first, temp-0 kills the third, certificates expose the second).
