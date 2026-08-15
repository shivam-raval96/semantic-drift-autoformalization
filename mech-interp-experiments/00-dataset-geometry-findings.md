# Experiment 0 — Dataset geometry (model-free): findings

Notebook: [`00-dataset-geometry.ipynb`](00-dataset-geometry.ipynb) ·
Artifacts: `exp00-outputs/baselines.json`, `exp00-outputs/similarity-matrices.npz`, `exp00-figures/`

**Status:** run, complete. No model involved; CPU only.

**Question.** How much of the structure experiment 3 looks for in activation space is already
recoverable from the raw text, and does the label geometry support the claims we want to make?

## Headline

Two numbers to carry forward. **The model-free floor for law retrieval is 24.5%, not the 4.8%
TF-IDF number** — text length alone is a near-fingerprint of the law, because the renderers are
templates. And **the 52 laws are mutually near-orthogonal points** (mean subtree cosine 0.06, two
similarity metrics agreeing at rho = 0.08), so "activations cluster by law" is the only
law-related question this sample can answer; anything about a manifold *among* laws is untestable
here regardless of what the model does.

## Setup

52 implication pairs (stratified over `ops_total` 1–8, vacuous laws dropped, seed 0) × 6
deterministic surface forms = 312 texts. Forms: four themed stories (`graft`, `paint`, `signal`,
`tea`, near-disjoint vocabulary), one literal-NL description, one Rigid Grammar rendering.
Featurizations: word TF-IDF, char TF-IDF (3–5 grams), word count alone, and a symbolic
bag-of-alpha-renamed-subtrees (IDF-weighted) that serves as the ceiling check.

`texts SHA d536d50e5c5c` — must match experiment 3's, or the floors don't transfer.

## Results

### A/C — Surface floors for 1-NN law retrieval (chance 1.9%)

| representation | all surfaces | story × theme |
|---|---|---|
| word TF-IDF | 3.5% | 4.8% |
| char TF-IDF | 6.7% | 9.1% |
| **length only** | **15.4%** | **24.5%** |
| symbolic (ceiling check) | 100% | 100% |

Length is the strongest surface retriever on both metrics, so it — not TF-IDF — sets the floor
experiment 3 has to clear. After residualizing word count out of the representation, word TF-IDF
rises to 5.8% / 12.0% and char TF-IDF to 6.4% / 12.5%, which are the numbers to compare a
length-residualized activation result against.

Lexical leakage is confined to the story–story block and is tiny even there (same-law minus
different-law mean similarity: 0.015 word / 0.013 char). Cross-form leakage is effectively zero
(story–literal 0.0004, story–RG 0.000, literal–RG 0.000), so cross-form co-clustering cannot be
shared wording.

### B — Complexity is almost entirely text length

Correlation of word count with `ops_total`, computed within each form:

| form | r(ops_total) | r(depth) |
|---|---|---|
| story | 0.928 | 0.571 |
| literal | 0.9995 | 0.696 |
| rg | 1.000 | 0.701 |

This is a dataset-level confound, not a plotting problem. Any "complexity gradient" reported in
activation space is a length gradient until it survives residualization; the clean fix is
length-matched instances upstream.

### A — The formality ladder is not present in surface space

Ladder statistic (literal centroid projected onto the story→RG axis; betweenness means t ≈ 0.5
with a small off-axis residual):

| representation | t | off-axis residual |
|---|---|---|
| word TF-IDF | +0.17 | 0.88 |
| char TF-IDF | +0.22 | 0.95 |
| symbolic (check) | +1.00 | 0.00 |

Literal-NL sits nowhere near between story and RG lexically, and is mostly off-axis. That large
residual is the null an activation-space ladder claim would have to beat — which makes experiment
7's ordered-formality hypothesis a real, non-trivial test rather than a restatement of the text.

### D — Target geometry among the laws

- Subtree cosine between distinct laws: mean 0.062, max 0.962.
- Normalized term edit distance: mean 0.422, min 0.038.
- Agreement between the two notions of law similarity: **rho = 0.080**.
- 52/52 laws have a distinct symbolic vector.

The two metrics essentially disagree, so there is no metric-independent notion of "similar law" in
this sample. Combined with the low mean cosine, the laws behave as unrelated points.

## Implications for later experiments

- Report experiment 3's retrieval against **24.5% (story × theme)**, not TF-IDF's 4.8%. The
  currently plotted TF-IDF line understates the model-free floor by 5×.
- Any complexity-geometry claim needs the length-residualized version.
- Do not frame experiment 7 as recovering a manifold *among* laws on this dataset; it can only
  test clustering by law and the story/literal/RG ladder.

## Open items

- `similarity-matrices.npz` was exported for RSA against activation similarity matrices; that
  analysis has not been run yet.
- A sentence-encoder floor (`SENTENCE_ENCODER`) is wired up but was left off.
