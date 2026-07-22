# Experiments

Each subdirectory is one experiment: a question about how well models
formalize Storyform stories, the benchmark runs that answer it, and the
generated report. Everything under `experiments/` is committed — this is the
project's lab notebook.

## Layout

```
experiments/
  NN-short-slug/
    EXPERIMENT.md   what was asked, how it was run, what was found
    runs/           one benchmark.py run directory per condition
      <run-name>/
        run_meta.json    seed, models, reasoning regime, prompt wrappers
        samples.jsonl    one row per (E, F) pair: story + exact prompt sent
        results.jsonl    one row per (pair, model): response + verdict
        summary.json     per-model bucket counts
        summary.md       the same summary as a Markdown table
    report/         charts.py output (self-contained HTML, optional PDF)
```

Experiments are numbered in the order they were started (`01-`, `02-`, ...).
A run directory is never edited after the fact; rerunning `benchmark.py`
with the same `--out-dir` only retries api-error rows.

## EXPERIMENT.md template

Every experiment file has these sections, in this order:

1. **Question** — the single question the experiment aims to answer.
2. **Setup** — models tested, reasoning/thinking regime, sampling scheme
   (seed, `--n` or `--stratify-ops`), and anything non-default.
3. **Prompts** — how the prompt each model saw was constructed: the prompt
   template (with its sha256 at run time), any regime wrapper text verbatim,
   and any per-model additions. The byte-exact prompt for every sample is in
   `runs/*/samples.jsonl` (`prompt` field); the wrapper actually applied is
   recorded in `runs/*/run_meta.json` (`regime_prefix` / `regime_suffix`)
   and `results.jsonl` rows carry `sent_prompt_hash`.
4. **Reproduce** — the exact commands, one per run directory, plus the
   `charts.py` command that produced the report.
5. **Results** — where the artifacts live and the headline numbers.
6. **Conclusions** — what the experiment answered, and what it raised.

## Running a new experiment (for humans and subagents)

1. Pick the next number and a short slug; create
   `experiments/NN-slug/` and write the **Question** and **Setup** sections
   of `EXPERIMENT.md` *before* running anything.
2. Run each condition with an explicit out-dir inside the experiment:

   ```sh
   python3 benchmark.py --seed 0 --n 30 --reasoning on \
       --out-dir experiments/NN-slug/runs/run-s0-n30-think-on
   ```

   Requires `OPENROUTER_API_KEY`. Sanity-check the pipeline first with
   `--dry-run` (no network; must grade 100% exact) into a throwaway
   directory under `results/` (gitignored scratch space).
3. Generate the report over all of the experiment's runs:

   ```sh
   python3 charts.py experiments/NN-slug/runs/* \
       --out experiments/NN-slug/report/report.html --pdf
   ```

4. Fill in **Prompts**, **Reproduce**, **Results**, and **Conclusions** from
   the artifacts (`summary.md` per run has the per-model table; copy the
   sha256 of `formalize_prompt.md` and the wrapper strings from
   `run_meta.json`, not from memory).
5. Commit the whole experiment directory in one commit.

Conventions that keep experiments comparable:

- Fixed seed (0 unless the experiment is about sampling variance) so runs
  within an experiment share the same pair set and `charts.py` can group
  and compare them.
- One variable per experiment where possible; hold models, seed, and prompt
  template fixed across the conditions being compared.
- Never edit `formalize_prompt.md` mid-experiment; if the prompt changes,
  that is a new experiment.
- **Exclude vacuous laws** — E1 `x = x` and E2 `x = y`, the only zero-op
  laws — from every sample, unless the experiment is specifically about
  them. Experiment 07 (`07-two-stage-scale/EXPERIMENT.md`) showed pairs
  containing one are a measurement hazard rather than a complexity
  gradation: models misread the vacuous law itself in ~20% of readings
  by inventing an operation the text never states (42% in the two-stage
  arm, whose stage-1 procedure grammar invites a "missing" definition
  step), and under `--stratify-ops` these pairs concentrate all of a
  bin's operations in the partner law, producing a spurious dip at
  ops-bin 4 that a total-operations axis misreads as non-monotone
  difficulty. Practicalities: neither sampler has an exclusion flag
  yet, so verify `samples.jsonl` before spending API budget; for
  existing runs, `filter_vacuous.py` copies run directories with the
  vacuous pairs removed (every affected experiment's `-no-vacuous`
  report is generated from such copies). Note also that
  ops-bin 1 consists *entirely* of vacuous-law pairs, so stratified
  sampling without them means bins 2–8. Uniform `--n` sampling is
  effectively unaffected (it almost never draws them). When comparing
  against pre-07 runs, prefer the vacuous-excluded view
  (`07-two-stage-scale/report/comparison-report-no-vacuous.*`).

## Planned experiments

- Compare models' ability to formalize the fuzzified stories against their
  ability to formalize a direct, plainly-worded translation of the same ETP
  implication into natural language (isolates the cost of the narrative
  disguise).
- Increase the fuzzification of the story and measure how accuracy degrades
  with narrative distance.
