# mech-interp-experiments

Mechanistic interpretability experiments, one Jupyter notebook per experiment, designed to run in Google Colab.

The experiments follow the ideas in [`MARS - mech-interp-experiments.md`](../MARS%20-%20mech-interp-experiments.md):

1. Contrastive steering vector: story → abstracted NL — [`01-contrastive-steering-story-to-literal.ipynb`](01-contrastive-steering-story-to-literal.ipynb)
2. Probing and steering with contrastive datasets
3. PCA / activation-structure visualization
4. Attention-pattern analysis: story vs abstracted version
5. SAE feature identification and boosting
6. Quick ambiguity-grading pipeline
7. Semantic manifold (Goodfire-style manifold steering)

## Running in Colab

Each notebook is self-contained: it installs its own dependencies in the first cell and can be opened directly in Colab via
`File → Open notebook → GitHub` (or by uploading the `.ipynb`). Prefer a GPU runtime (`Runtime → Change runtime type → GPU`) for anything that loads a model.
