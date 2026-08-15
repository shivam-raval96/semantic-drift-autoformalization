# CLAUDE.md

Mechanistic-interpretability experiments on how a language model formalizes
implication problems — "if a world always obeys rule E, must it also obey rule
F?" — presented either as a themed story or as plain mathematical English.

`informalizing-etp/` is a vendored checkout providing the deterministic story
and description renderers (`storyform.py`, `literalform.py`), the syntactic
grader (`checkform.py`), and the sampling utilities (`benchmark.py`).

**Treat it as read-only. Never add, move, or modify files inside it.** Import
from it by putting it on `sys.path`; everything written for this project lives
in `mech-interp-experiments/`, including any shared module.

## New experiments: one dated folder, one Python file

For any new experiment, **first create a folder under
`mech-interp-experiments/` named after the date** the experiment is started, in
`YYYY-MM-DD` form. Inside it, **the experiment is a single Python file**, named
after what it measures.

```
mech-interp-experiments/
  2026-08-15/
    steering-at-budget-512.py     the experiment: design, generation, analysis
    runs/
      steering-at-budget-512/     everything the script wrote
        run_meta.json             config and provenance, written once
        records/<condition>.jsonl one row per graded example
        summary.json              the headline numbers
    figures/
```

Several related experiments started on the same day share the dated folder,
one file each. A run directory is never edited after the fact.

Experiments are plain Python scripts run on a rented GPU box. They are not
notebooks — the numbered notebooks already in `mech-interp-experiments/` are
earlier Colab work and are left alone. Notebooks are for looking at results,
never for producing them.

## The experiment file

- Holds the design, the generation, and the analysis. Support `--analyze-only`
  so tables and figures can be redone from the run directory with no model and
  no GPU; iterating on a plot must never require re-running the model.
- Support `--quick` for a tiny smoke run that checks the plumbing end to end
  before committing to a real one.
- Take `--out-dir`, `--seed`, `--model`, and `--force` as arguments rather than
  requiring constants in the file to be edited.
- Write the question, what each possible outcome would mean, and the controls
  into a comment block at the top **before running anything**.
- Declare the sweep as data — a list of named conditions — not as nested loops,
  so each cell records its settings and a rerun resumes.

## Non-negotiables

- **Never reimplement shared machinery inside an experiment file.** Model
  loading, generation, thinking-budget forcing, steering hooks, activation
  capture, and grading belong in a shared module every experiment imports,
  living in `mech-interp-experiments/` alongside the dated folders. Two
  copies turn a difference between the copies into an apparent effect. That has
  already happened here twice: the thinking-budget sweep exists in two
  notebooks whose numbers disagree by 5–7 points at every budget, and the
  steered and unsteered budget generators are a near-verbatim copy-paste that
  are then compared against each other.
- **Resume safely.** Write a condition's records under a temporary name and
  publish them only once that condition finishes, so a crash cannot leave a
  short file that looks complete.
- **Record provenance** in every run directory: model, seed, all parameters,
  and the commit of both this repository and `informalizing-etp`.
- Fixed seed 0 unless the experiment is about sampling variance.
- One variable per experiment where possible; hold model, seed, and prompt
  template fixed across the conditions being compared.
- Exclude vacuous laws (`x = x`, `x = y`) unless the experiment is about them.
- Compare conditions **paired**, on the same problems, with McNemar's test. At
  ~52 examples that is the difference between resolving a ten-point shift and
  not.
- Report every rate with its denominator, a confidence interval, and the
  baseline it is measured against.
- A steering intervention needs two controls: a random direction of the same
  length, and the negated vector. A symmetric response to `+v` and `-v` means
  disturbance size is being measured, not direction.

## Writing for humans

Reports, figure captions, and commit messages are read by people who know the
field but not this project. Define every term of art in place on first use,
never use project-internal shorthand (`exp03`, `L18-a4`) as if it were common
knowledge, and always give a number together with what it is measured against.
