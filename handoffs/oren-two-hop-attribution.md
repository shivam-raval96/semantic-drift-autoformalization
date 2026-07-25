# Oren — Two-hop attribution: where does two-stage translation lose it?

## Question

The two-stage arm (story → abstract description → formal) helps no-think
models. When it fails, which hop failed — the abstraction (stage 1) or
the formalization (stage 2)? And does stage 2 fail *because* stage 1
drifted, or does it add its own drift on top?

## Why this matters now

Two-stage is our main decomposition intervention and the closest thing
the project has to a scalable-oversight protocol (each hop is simpler to
check than the whole). Attribution decides what to fix: better
abstraction prompts, or better formalization training. Experiment 12's
component anatomy (55% of failures are single-equation-local) makes hop
attribution sharper: a stage-1 drift on the premise should surface as a
premise-drifted final answer.

## What exists to build on

- `benchmark.py --form two-stage`: stage-1 rows already carry `stage1_*`
  bookkeeping; in dry runs stage 1 answers with the deterministic
  literalform rendering — that machinery IS the gold-conditioning arm.
- Exp 07/09 two-stage runs: stage-1 raw responses are stored, so hop-1
  quality can be graded offline against the deterministic literalform
  (backparse gives you the gold abstract form for every pair).
- Experiment 12's probes for classifying final-answer drift by component.

## Design — the gold-conditioning ablation

Three arms on one balanced pair set (reuse exp 07's set for
comparability), per model:

| arm | stage 1 | stage 2 | isolates |
|---|---|---|---|
| A model-model | model | model | end-to-end (exists: exp 07) |
| B gold-model | deterministic literalform | model | stage-2-only failure |
| C model-graded | model | — (grade stage 1 itself) | stage-1-only failure |

Arm C grading: backparse the stage-1 output as a literalform description
(exact round-trip machinery exists); statuses correct / drifted /
unparseable per equation.

**The attribution identity to test**: err(A) ≈ err(C) + err(B|C-correct).
Deviations are the interesting part — error *cancellation* (stage 2
recovers from stage-1 drift: how often, and is recovery itself faithful
or lucky?) vs error *amplification* (stage 2 drifts more given drifted
input than given gold input — measure err(B) vs err(A on C-correct
subset)).

**Controls**: same elicitation everywhere (coordinate with Denis); arm B
must use stage-1 text formatted indistinguishably from model output (no
"this is gold" tells — pass it through the same wrapper verbatim);
per-ops-bin reporting; component-level (premise/conclusion) attribution
via the exp-12 probes.

## Kill baselines

- If err(B) ≈ err(A) for a model, stage 1 contributes nothing to its
  failures — the decomposition story for that model is dead; say so.
- Format-only null: arm B with a *shuffled* gold description (right
  format, wrong content) should crater accuracy; if it doesn't, stage 2
  ignores its input and the two-hop framing is theater. This is the
  single most important control in the design.

## Deliverable

The attribution table (per model × ops bin: stage-1 share, stage-2
share, cancellation rate, amplification factor) + the format-only null
+ a half-page on what it means for decomposition-as-oversight. Phase 1
(arm C, offline) costs nothing; arms A/B reuse + one new sweep, ~$30.
