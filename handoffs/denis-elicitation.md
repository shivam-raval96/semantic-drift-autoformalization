# Denis — Elicitation sensitivity: how much of a "judgment" is the asking?

## Question

How much does a model's measured competence on certified implication
judgment depend on *how the verdict is elicited*, and does elicitation
change the **bias** (True-rate) independently of the **accuracy**?

## Why this matters now

We piloted this by accident. In the behavioral check
(`causalab-integration/scripts/behavioral_check.py`), moving from a
4-token answer budget to reason-then-verdict with 3000 tokens flipped the
answers-True rate from 0.28 to 0.65 on the same balanced items — a bigger
swing than most model-to-model differences. If verdict bias is an artifact
of elicitation, every behavioral number in the project (and in most
published "LLMs can/can't reason" claims) carries an unmeasured
specification error. This is the behavioral twin of experiment 12's
grader-variant curve: there we varied the grader, here you vary the asker.

## What exists to build on

- `behavioral_check.py`: OpenRouter harness, balanced per-(level,label)
  sampling, .env key handling, last-line verdict parsing, follow-up nudge.
- v5 task data with certified labels + complexity strata
  (`causalab-integration/tasks/etp_implication/data/etp_pairs.json`).
- The two accidental data points above (screenshots in the chat log;
  rerun to reproduce cleanly).

## Design

Fix one balanced item set (160 items, 20 per ops level, True/False
balanced within level — the harness already does this). Cross it with an
elicitation grid, at temperature 0:

| axis | variants |
|---|---|
| budget | 1 token / 16 / 256 / 3000 |
| order | verdict-first / reason-then-verdict |
| format | bare True/False; JSON field; "final line" convention |
| framing | neutral; "be skeptical"; "answer True only if certain" |

Not the full product — pick ~10 cells spanning the axes. Three models
(one open small, one open large, one frontier), same items everywhere.

**Metrics per cell**: accuracy, True-rate (bias), unparsed rate, and
accuracy *conditional on* the premise-strength kill baseline (does any
cell beat premise-outdeg-only AUC 0.9635? see handoffs/README.md).

**Nulls / controls**:
- Label-shuffle null for accuracy (should be ~0.5 everywhere — balanced).
- The specification curve itself is the result: plot accuracy and bias
  across cells; a flat curve kills the concern, a wide one quantifies it.
- Answer-order control: half the items ask "False or True" to catch
  option-order bias.

## Kill baselines

- Premise-outdeg-only AUC (0.9635): if the best elicitation cell doesn't
  beat it, the model is not doing relational judgment under ANY asking.
- BoW leave-pair-out per-level floors (0.50–0.69) still apply.

## Deliverable

One run dir per cell + a single specification-curve figure (accuracy and
True-rate per cell, models overlaid) + a 1-page writeup: how wide is the
elicitation band, and does any conclusion about model competence flip
across it? Target: ~$20 of API, 2–3 days.
