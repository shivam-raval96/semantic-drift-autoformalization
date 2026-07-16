# Experiment proposal template

Copy this to `notes/proposals/<area>-<slug>.md` and open a PR containing just
the proposal BEFORE spending API credits or GPU hours — review happens on the
design, not after the tokens are gone. A paper usually needs only 2–3 major
experiments; each can bundle several results (sanity checks, baselines, main
result). Proposing a fourth? Say which one it replaces.

Leave **Actual results** empty until the runs exist. When you fill it in,
every number links to its results file and to the instrument card
(`pipeline/cards/`) that licenses it.

---

## Experiment <n>: <short name>

**Models:** exact ids, temperature, seeds. (Deterministic-only models follow
the 3-seed convention in `pilot.py`.)

**Datasets:** which families/banks and sizes; freeze hash if the run is
instrument-bearing.

**Hypotheses:** falsifiable, one per line — and for each, what result would
*refute* it. "X improves Y" without a kill condition is a hope, not a
hypothesis.

**Sanity checks:** what must hold before the main number means anything —
e.g. perfect-model invariant, freeze tests green, no-leakage test, recovering
a known result on stored data.

**Baselines:** what must be *beaten* for the result to matter — random /
majority class, surface features, the gauntlet contenders.

**Expected results:** direction and rough magnitude, written down before
running.

**Expected plots:** axes and one sentence each. If you can't sketch the plot,
the experiment isn't designed yet.

---

**Actual results:** numbers with CIs; link results file + licensing card;
each deviation from expected gets one honest sentence.

---
---

# Worked example (prospective — Ke's starter task)

## Experiment 1: Does repair move the VBU cell?

**Models:** GPT-5.4 (the existing LADR pilot outputs; no new generation —
this experiment relabels stored attempts).

**Datasets:** `LADR_pilot_27` (27 theorems × {statement_only,
statement_plus_proof} × {one-shot, post-repair}); back-translation cards for
the faithful axis.

**Hypotheses:**
- H1: compiler-feedback repair raises the *valid* rate without raising the
  *faithful* rate — i.e. repaired items land disproportionately in VBU.
  Refuted if the VBU share of compiled items is flat or falls after repair.
- H2: the informal proof shifts failures from unfaithful toward invalid
  (visible errors) rather than reducing total error.
  Refuted if VBU share is equal or higher in `statement_plus_proof`.

**Sanity checks:** compile counts reproduce the pilot log (10/27 and 9/27
one-shot; 23/27 and 20/27 post-repair); every attempt gets exactly one
quadrant; faithful-axis labels are made blind to compile status.

**Baselines:** the one-shot 2×2 is the baseline the post-repair 2×2 is read
against; a random-relabel control on the faithful axis bounds annotation
noise.

**Expected results:** repair moves most fixed items into faithful·valid, but
some previously *invalid-and-unfaithful* items compile without becoming
faithful — VBU count rises from one-shot to post-repair.

**Expected plots:** two 2×2 heatmaps (one-shot vs post-repair, counts per
quadrant) per condition; one slope chart of the VBU count across the repair
boundary.

---

**Actual results:** (empty until run)
