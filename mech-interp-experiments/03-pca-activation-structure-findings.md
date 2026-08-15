# Experiment 3 — PCA / activation structure: findings

Notebook: [`03-pca-activation-structure.ipynb`](03-pca-activation-structure.ipynb) ·
Artifacts (Colab/Drive): `activations-*.pt`, `budget-sweep-*.jsonl`, `records-*.jsonl`

**Status:** sections A–E and F2 complete. **F3, G, and H did not run** — the latest run raised
`OutOfMemoryError` in `read_positions` during the full prompt-activation pass.

Model `Qwen/Qwen3-4B`, seed 0, 52 pairs × 6 surface forms = 312 texts (same `texts SHA` as
experiment 0, so its floors apply).

**Question.** Does activation space contain visible structure that tracks the ground-truth tags —
does the same law cluster across surface forms (the semantic core), and is that structure causally
used?

## Headline

**Law identity is strongly decodable from the mid-stack residual stream: 86% cross-theme 1-NN
retrieval at layers 18–24 (last token), against 1.9% chance, 1.9% at the embedding layer, and a
24.5% strongest model-free floor from experiment 0.** It survives both the lexical control
(themes share almost no vocabulary) and the structural-shape control (92.8% vs a 13.5%
shape-only baseline, 6.9×). Whether the model *reads* from that representation is still unknown —
that is exactly what the two sections that OOM'd were built to test.

## Results

### A — Representation type

Surface form dominates the top principal components at every layer; 2D PCA captures 72–100% of
variance, almost all of it separating Rigid Grammar from natural language. The four story themes
stay tightly co-located and literal-NL sits apart from both. No clean story → literal → RG ladder
is visible in the top-2 projection.

### B — Complexity

An ordered gradient rather than clusters, clearest along the RG arm. Caveat from experiment 0:
word count correlates with `ops_total` at r = 0.93–1.00 within every form, so this is a length
gradient until it is shown to survive residualization. That check has not been run.

### C/D — Law identity across surface forms (the semantic core)

1-NN law retrieval, chance = 1/52 = 1.9%:

| layer | pooled, all | pooled, story × theme | last, all | last, story × theme |
|---|---|---|---|---|
| 0 (embeddings) | 8.7% | 12.0% | 1.9% | 1.9% |
| 6 | 18.3% | 26.4% | 30.8% | 44.2% |
| 12 | 12.2% | 16.8% | 43.6% | 59.1% |
| 18 | 17.3% | 24.5% | 64.1% | **85.6%** |
| 24 | 29.2% | 42.3% | 62.5% | **86.1%** |
| 30 | 13.8% | 19.7% | 39.7% | 56.2% |

TF-IDF baseline: 3.5% all, 4.8% story × theme. The inverted U peaking at layers 18–24 is the
standard abstraction signature, and since the four themes share almost no vocabulary by
construction, the cross-theme column cannot be lexical overlap.

**Two corrections to how the notebook reports this.** First, the last-token position is far better
than mean-pooling here (86% vs 42% at layer 24), which is the opposite of the notebook's
`PRIMARY_POSITION = "pooled"` assumption — so the "best layer" selection and both section-C
scatters are running on the weaker position. Second, the plotted TF-IDF reference line understates
the model-free floor; experiment 0 shows length alone retrieves at 24.5% story × theme. The result
clears that comfortably, but the right line to draw is 24.5%.

### E — Structural-shape control

Retrieval restricted to same-shape candidates, so shape carries no information inside the pool:

| shape definition | shape-only baseline | strictest cell (last token, story × theme) | lift |
|---|---|---|---|
| `ops_total` | 13.5% | 92.8% (layer 18) | 6.9× |
| `ops_e`/`ops_f`/`depth` | 50.0% | 96.2% (layer 24) | 1.9× |

The fine definition is underpowered (median 1 competing law in the pool), so the coarse row is the
one to cite. Conclusion: the model represents *which* law it is, not merely how big the law is.

### F2 — Thinking-budget calibration

52 stories, one per law, wrapped in the formalization prompt:

| budget | correct | wrong | unparseable | cut off |
|---|---|---|---|---|
| 0 | 15.4% | 84.6% | 0.0% | 0% |
| 64 | 11.5% | 76.9% | 11.5% | 100% |
| 128 | 17.3% | 63.5% | 19.2% | 100% |
| 256 | 25.0% | 59.6% | 15.4% | 100% |
| 512 | 44.2% | 48.1% | 7.7% | 100% |

Recommendation taken: `THINK_BUDGET = 256`. This agrees with experiment 1's independent sweep
(story arm: 12% at 256, 38% at 512), including the dip below the no-think floor at small budgets.

By complexity, at budget 512: 100% on ops 1–2, 55% on ops 3–4, 40% on ops 5–8.

**Why this matters beyond calibration.** The story-span activations are provably
budget-independent — the thinking block is appended after the user turn, so under causal masking
it cannot reach the story tokens. Accuracy moves from 15% to 44% while that representation is
bit-identical. So the information needed was already present while the model was failing 85% of
the time: thinking adds serial room to unpack the representation, not comprehension.

### F3 / G / H — did not run

`read_positions` OOM'd on the full prompt-activation pass: `output_hidden_states=True` over 208
full-length formalization prompts at batch size 8 keeps all 37 layers' hidden states resident.
Consequently there are no results for the graded full pass, the faithfulness monitor, or the
law-transplant causal test in this run.

Fix before rerunning: reduce `GEN_BATCH_SIZE`, and slice to `LAYERS` inside the loop (or use a
hook that captures only the requested layers) instead of materializing every hidden state.

Carried over from earlier runs, and worth re-confirming rather than citing as settled: the first
run generated with no thinking at all, scored 12%, and starved both G and H; G's 0.71 AUC came
from `prompt_end`, where cross-theme retrieval is 19% against 77–87% at `story_end`.

## What this establishes

Combined with experiment 1: the semantic core is present and decodable by layer 18, steering along
the form axis is causally inert, and what separates 10% from 90% is serial token budget. The open
question — whether the decodable law representation is actually read during formalization — is
precisely what G and H would have answered, and it is also the subject of experiment 8.

## Next steps

1. Fix the OOM and rerun F3/G/H at `THINK_BUDGET = 256`, reading at `story_end`.
2. Switch `PRIMARY_POSITION` to `"last"` (or headline the last-token numbers) so section C's best
   layer and scatters reflect the stronger position.
3. Redraw section D's floor line at experiment 0's 24.5%, and add the length-residualized
   complexity check for section B.
4. If H is null at budget 256, rerun the transplant at the smallest budget that clears the floor —
   a long thinking trace gives the model room to attend back to the unpatched story text.
