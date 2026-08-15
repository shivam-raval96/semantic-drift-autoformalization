# Experiment 8 — Bottleneck localization: findings

Notebook: [`08-bottleneck-localization.ipynb`](08-bottleneck-localization.ipynb) ·
Artifacts: none yet (`exp08-outputs/` will hold `activations-*.pt` and graded rows)

**Status: not run.** The notebook is written and has no executed output.

**Question.** Experiment 3 showed law identity is recoverable from mid-stack activations. That is
correlational. This notebook crosses two axes on the *same* examples — is the law represented
(cross-theme 1-NN retrieval at the prompt's story span), and did the model formalize it correctly
(graded generation) — to localize where the no-think failure lives:

| | correct output | wrong output |
|---|---|---|
| **law represented** | pipeline works | **formalization bottleneck** |
| **law not represented** | got lucky / shallow route | **abstraction bottleneck** |

## What we already expect going in

From experiments 1 and 3, the prior is fairly strong that the dominant cell will be
*law represented + wrong output* — the formalization bottleneck:

- Experiment 3: 86% cross-theme retrieval at layers 18–24 while the same prompts are graded
  correct only ~15% of the time in no-think mode.
- Experiment 3's F2: the story-span representation is provably budget-independent, yet accuracy
  moves from 15% to 44% as the thinking budget grows. The representation was already carrying what
  was needed while the model was failing.
- Experiment 1: steering the story→literal direction closes 0% of the gap, and the budget sweep
  closes most of it.

If the run instead finds the abstraction cell dominating, that contradicts the above and is the
more interesting outcome — it would mean the retrieval result does not transfer from bare texts to
prompted inputs.

## Things to fix or watch before running

- **Read position.** The notebook's stated axis-1 read point is "the final prompt token". That is
  the position experiment 3 measured at 19% cross-theme retrieval, against 77–87% at the story
  span, because the formalization prompt ends in boilerplate identical across examples. Read at
  the story span, or the represented/not-represented axis will be mostly noise.
- **Regime.** It generates in no-think mode, where accuracy is ~10–15%, which leaves the
  "correct output" column nearly empty and the cross-tab underpowered. Consider running at
  experiment 3's calibrated `THINK_BUDGET = 256` (25% correct) or at 512 (44%), noting that the
  story-span activations are unchanged by the budget so axis 1 is unaffected.
- **OOM.** Experiment 3 died extracting hidden states over ~200 full-length prompts with
  `output_hidden_states=True` at batch size 8. This notebook does the same thing over ~208
  prompts; lower the batch size or capture only `LAYERS` before running.
- **Complexity confound.** Already handled in the design (numbers reported within complexity
  bins), which matters because experiment 0 found word count correlates with `ops_total` at
  r = 0.93–1.00 within form.

## Results

_To be filled in after the first run._
