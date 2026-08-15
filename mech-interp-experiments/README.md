# mech-interp-experiments

Mechanistic interpretability experiments on how a language model formalizes implication problems.

## Layout

New experiments are **plain Python scripts** run on a rented GPU box, one dated folder per day
and one file per experiment, writing to a run directory that is never edited afterwards. See
[`CLAUDE.md`](../CLAUDE.md) for the full convention.

```
shared/          machinery every experiment imports: dataset, model, generation,
                 hooks, grading, statistics, run directories
tests/           unit tests for the above, plus the length-balance acceptance check
2026-08-15/      one day's experiments, each a single .py file with its runs/ and figures/
0*.ipynb         earlier Colab work, left as it is
```

Everything the model is asked to do — loading it, generating under a thinking-token budget,
capturing activations, steering, grading — lives in `shared/` and exists exactly once. That
rule is not stylistic: the budget sweep was previously written twice and the two copies
disagree by 5 to 7 accuracy points at every budget, which would have been read as a result.
`shared/generation.py` documents how they differed and which behaviour was kept.

```sh
cd mech-interp-experiments
python3 -m unittest discover -s tests -t .   # no GPU needed
python3 tests/check_length_balance.py        # the depth grid's acceptance check
python3 2026-08-15/depth-at-fixed-length.py --quick
```

## Experiments

| started | experiment | question | status |
|---|---|---|---|
| 2026-08-15 | [`depth-at-fixed-length`](2026-08-15/depth-at-fixed-length.md) | does accuracy fall with a problem's serial depth once text length is held constant? | written, not run |

## Earlier Colab notebooks

These predate the convention above and are left alone. They follow the ideas in
[`MARS - mech-interp-experiments.md`](../MARS%20-%20mech-interp-experiments.md):

0. Dataset geometry (model-free) — [`00-dataset-geometry.ipynb`](00-dataset-geometry.ipynb). Not one of
   the numbered ideas: it measures what structure the dataset itself contains and how much of it a bag
   of words or a word count already recovers, so the activation results in 3 and 7 can be reported
   against a floor. Run it first; it needs no GPU and takes about a minute.
1. Contrastive steering vector: story → abstracted NL — [`01-contrastive-steering-story-to-literal.ipynb`](01-contrastive-steering-story-to-literal.ipynb)
2. Probing and steering with contrastive datasets
3. PCA / activation-structure visualization — [`03-pca-activation-structure.ipynb`](03-pca-activation-structure.ipynb)
4. Attention-pattern analysis: story vs abstracted version
5. SAE feature identification and boosting
6. Quick ambiguity-grading pipeline
7. Semantic manifold (Goodfire-style manifold steering)
8. Bottleneck localization: is the law represented, and does the output get it right? — [`08-bottleneck-localization.ipynb`](08-bottleneck-localization.ipynb)
9. Law-forced manifolds: does a stated law induce a geometry, and does narrative deform it? —
   proposal in [`09-law-forced-manifold-proposal.md`](09-law-forced-manifold-proposal.md), competence
   gate in [`09-law-forced-manifold-pilot.ipynb`](09-law-forced-manifold-pilot.ipynb). A rework of
   idea 7: the formality ladder has no behavior manifold to fit, so this substitutes a concept domain
   whose metric a law forces, and varies surface form. Run the pilot before anything else — it decides
   whether the full design is viable.
10. J-space abstraction vs translation — [`10-jspace-abstraction-vs-translation.ipynb`](10-jspace-abstraction-vs-translation.ipynb).
    Uses the Jacobian lens / J-space toolkit: silent-intermediate readouts, workspace ablation
    (story vs literal dissociation), and a law-identity swap pilot.

## Findings

Each notebook that has been run has a companion `*-findings.md` holding its results, caveats, and
next steps. Update it after every run rather than relying on the executed notebook, which lives in
Colab.

| experiment | findings | status |
|---|---|---|
| 0 — dataset geometry | [`00-dataset-geometry-findings.md`](00-dataset-geometry-findings.md) | complete |
| 1 — contrastive steering | [`01-contrastive-steering-story-to-literal-findings.md`](01-contrastive-steering-story-to-literal-findings.md) | complete (steering null, budget sweep positive) |
| 3 — PCA / activation structure | [`03-pca-activation-structure-findings.md`](03-pca-activation-structure-findings.md) | A–E and F2 done; F3/G/H OOM'd |
| 8 — bottleneck localization | [`08-bottleneck-localization-findings.md`](08-bottleneck-localization-findings.md) | not run |
| 9 — law-forced manifolds (pilot) | — | not run |
| 10 — J-space abstraction vs translation | — | not run |

The current picture across experiments: law identity is strongly decodable mid-stack (86%
cross-theme retrieval at layers 18–24), steering the story→literal direction closes 0% of the
no-think gap, and accuracy is instead a smooth function of thinking-token budget, with the story
framing costing roughly a 2× budget multiplier over literal-NL. Whether the decodable law
representation is causally read during formalization is still open.

## Running the older notebooks in Colab

Each notebook is self-contained: it installs its own dependencies in the first cell and can be opened directly in Colab via
`File → Open notebook → GitHub` (or by uploading the `.ipynb`). Prefer a GPU runtime (`Runtime → Change runtime type → GPU`) for anything that loads a model.

New work does not go in a notebook. Notebooks are for looking at results, never for producing them.
