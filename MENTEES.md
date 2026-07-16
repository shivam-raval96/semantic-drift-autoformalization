# Mentee guide — pick up here

This branch adds the working pipeline: the certificate-level checker, the
instruments, offline test suites, and enough stored data to reproduce a
result on your laptop before you ever touch an API key or a GPU.

## First 20 minutes (everyone)

```bash
cd pipeline
pip install -r requirements.txt   # numpy + scikit-learn; the pipeline itself is stdlib-only
python test_pilot.py       # 26 tests, offline (~1-2 min: countermodel searches)
python test_defense.py     # 9 tests: freeze hashes, leakage guards, stats
python pilot.py demo       # mock end-to-end -> results + report
```

Read [README.md](README.md) ground rules and
[pipeline/grumeter-cards.md](pipeline/grumeter-cards.md) — every instrument
has a scope card in [pipeline/cards/](pipeline/cards/); read the card before
trusting any number the instrument produces (including your own).

## First tasks (no API costs, no GPU)

**Oren — Storyform as an instrument.** Your renderer needs a scope card
before it becomes a stimulus register: copy the shape of
[pipeline/cards/cse.md](pipeline/cards/cse.md). Then look at
`pipeline/cse.py` — certified semantic entropy is how we'll *measure* the
claim that your stories are unambiguous. Offline replay of the stored
ensembles: `python cse.py t0` (analyze mode; samples are in
`cse-samples-v01-*.jsonl`).

**Ke — the 2×2 on your LADR pilot.** `benchmark/make_benchmark.py` labels
translation attempts into the faithful × valid quadrant. Adapt it to your
27-theorem pilot: compile = the *valid* axis; your back-translation cards
= the *faithful* axis. Do the pre-repair outputs first, then the
post-repair ones. The interesting number is how the valid-but-unfaithful
cell moves under repair.

**Denis — reproduce a probe result from stored activations.**
```bash
cd pipeline
python probes.py acts-hf-Qwen-Qwen2.5-0.5B-Instruct.npz results-hf-Qwen-Qwen2.5-0.5B-Instruct.jsonl
```
182 items, mean-pooled residual-stream activations. Read
[pipeline/cards/intent-probe.md](pipeline/cards/intent-probe.md) first —
note what a probe hit does and does not license. GPU capture for larger
models comes later; the runbook is [pipeline/LAMBDA.md](pipeline/LAMBDA.md).

**Harsh — run the gauntlet, then explain one kill.** The gauntlet pits
the probe against input-only contenders (register one-hots, surface
features, another model's encoding of the same prompt). It needs the
Llama-3.1-8B capture from the mentors' tree — ask and it'll be synced
(the small in-repo Qwen capture can't support it: 19 valid items, one
register; `gauntlet.py` will tell you the same). Then:
```bash
cd pipeline
python gauntlet.py    # defaults to the 8B capture once synced
```
Before the sync lands, do the part that's actually the deliverable: read
`gauntlet.py` (135 lines), pick one kill-baseline, and write a paragraph
on what it would mean if it *matched* the probe. Ladder stimuli for the
register work are in `pipeline/data-ladder.jsonl`; detector baselines
live in `detectors/`.

## From starter task to your own experiment

The starter tasks are entry points, not assignments — where they lead is
yours to decide. The interface between your direction and mentor review is
the experiment proposal: copy
[notes/experiment-template.md](notes/experiment-template.md) to
`notes/proposals/<area>-<slug>.md`, fill it in, and open a PR with just the
proposal *before* spending API credits or GPU hours. Review happens on the
design (hypotheses with kill conditions, sanity checks, baselines, expected
plots), which is cheap; re-running a badly designed experiment is not.

Calibration: a paper usually needs only 2–3 major experiments, each bundling
several results (sanity checks, baselines, main result). One well-designed
experiment that survives its own kill conditions beats five exploratory
sweeps. The template has a worked example.

## Rules that protect your own results

- **Freeze tests are load-bearing.** If `test_prompt_freeze` or
  `test_renderer_freeze` fails, you changed the instrument. Sometimes
  that's right — but it's a conscious act: update the hash in the same
  commit, say so in the PR title (`[instrument]`), and expect review.
  Results produced under different hashes are different experiments.
- **Certify or exclude.** 'Not refuted' is never treated as gold, anywhere.
- **Gold never enters a prompt.** There's a test for it; keep it green.
- **Mock numbers are fictions.** They validate pipelines, never findings.
- Work lands by PR from a branch named `<area>/<slug>` (e.g.
  `probes/narrative-register`). Small PRs, description says what + how
  verified.

## What's deliberately not here

Some lab instruments, larger activation dumps (the 8B capture is ~37MB,
the geometry set ~180MB), and full API-run outputs stay in the mentors'
private tree — ask and they'll be synced over when your experiment needs
them. If a file this guide references is missing, that's a bug in the
guide: open an issue.
