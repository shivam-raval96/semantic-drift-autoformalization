# Semantic drift in autoformalization

Autoformalization turns natural-language mathematics into machine-checkable formal
statements. The usual ways of evaluating it are weak: check whether the output
compiles, or ask an LLM whether it looks right. Neither catches **semantic drift**:
a formalization that is well-formed, even provable, but means something different
from the source.

This repository measures that failure mode deterministically. The pipeline runs
*backwards* from normal practice:

1. Start from a formal object whose meaning is known exactly: an implication
   "law E implies law F" between two magma equations from the
   [Equational Theories Project](https://github.com/teorth/equational_theories)
   (4,694 laws, Lean-verified implication graph).
2. Render it into natural language with a **deterministic, invertible renderer**,
   either as a themed story with all notation stripped, or as a plain "literal"
   description.
3. Ask a model to recover the formal statement in a small rigid grammar taught
   inside the prompt.
4. Grade the answer with a **parser, not a judge**. Because the text was generated
   from a known formal object, the reference answer exists before any model runs.

Every verdict is one of **correct / wrong-but-well-formed / unparseable**. The
wrong-vs-unparseable split is the point: "wrong" is exactly the failure that
would sail through a proof checker.

The paper draft is `paper/manuscript.tex` (*Deterministically Measuring
Autoformalization Faithfulness*). A plain-language walkthrough of the pipeline is
in `benchmark-pipeline.md`; the formal experimental-setup writeup is
`exp-setup-overview.md`.

## A worked example

Input: `x ∘ y = (y ∘ y) ∘ x` implies `x ∘ y = y ∘ x` (ETP E387 ⇒ E43), paint theme.

> In a certain paint workshop, the colorist follows one unbreakable habit. Take
> any two pigments at all — call the first crimson and the second ochre. She runs
> two procedures side by side. In the first, she pours crimson into ochre and sets
> the result aside as Batch 1. In the second, she pours ochre into ochre and calls
> the result Batch 2; then she pours Batch 2 into crimson and calls that Batch 3.
> However she chooses her two starting pigments, Batch 1 and Batch 3 always come
> out the exact same color. [...] One morning her apprentice wonders [...] must
> Batch 1 and Batch 2 always come out the same color?

The model must end its response with:

```
ASSUME: op(x, y) = op(op(y, y), x)
ASK: op(x, y) = op(y, x)
```

The grader canonicalizes both lines and compares against the stored reference,
accepting variable renaming, per-equation side swap, and uniform dualization
(mirroring every op's argument order across *both* equations). It never accepts a
reversed implication or a dualization applied to only one equation.

## Repository map

```
.
├── informalizing-etp/       the benchmark: renderers, grader, harness, all 14 experiments
├── main-results/            figures and the four headline results for the paper
├── paper/                   NeurIPS-style manuscript, figures, bib
├── ft-experiments/          fine-tuning thread: does FT teach translation or a notation?
├── probe-experiments/       interpretability thread: is correctness linearly represented?
├── benchmark-pipeline.md    plain-English, step-by-step description of the pipeline
├── exp-setup-overview.md    experimental setup writeup (substrate, rendering, grading, harness)
├── exp_setup_outline.md     the same in outline form
├── CLAUDE.md                project handoff: rules, data, infra, results of all three threads
├── RESEARCH_LOG.md          dated decisions, dead ends, and falsified claims
├── MORNING_REPORT.md, DASHBOARD.md, ONEPAGER.md, REFACTOR_SPEC.md   status docs from the FT/probe work
└── index.html               earlier project page
```

### `informalizing-etp/` — the benchmark

This is the core. Python 3, standard library only for rendering and grading.

| file | role |
|---|---|
| `storyform.py` | equation string → AST → SSA-style steps → themed story. Pure function; theme chosen by hashing the canonicalized pair. Also defines the AST (`Var`, `Op`), `canonical()`, `dual()`. |
| `literalform.py` | the contrasting arm: same implication as a direct description with lettered variables and "Value 1" intermediates. |
| `symbolform.py` | rigid-grammar rendering of a pair (used by the truth-judgment task). |
| `backparse.py` | story text → both term trees. Proves the rendering is injective; run over every story in tests. |
| `checkform.py` | **the grader**. Extracts the last `ASSUME:`/`ASK:` lines, parses, canonicalizes under the symmetry group, returns correct/wrong/unparseable. |
| `benchmark.py` | eval harness. Samples pairs, renders in the chosen arm, queries models through OpenRouter, grades, writes an immutable resumable run directory. |
| `charts.py` | run directories → self-contained HTML (optionally PDF) report. |
| `genform.py` | seeded synthesis of laws with more than 4 operations (the ETP list stops at 4). |
| `truthdata.py`, `proveform.py` | implication-truth companion task: sample truth-balanced pairs from the Lean-verified outcomes and grade True/False answers. |
| `voteform.py` | reduce K temperature-sampled runs to majority-vote run directories. |
| `paired_analysis.py` | per-pair story-vs-literal comparison across runs. |
| `filter_vacuous.py` | copy run dirs with the zero-op laws `x = x` / `x = y` removed. |
| `themes/` | four story themes as JSON: `paint`, `tea`, `graft`, `signal`. Schema in `themes/README.md`. |
| `prompts/` | every prompt template, one file each (`formalize_prompt.md`, `literal_prompt.md`, `abstract_prompt.md`, near/far-grammar variants, prove/invent prompts). |
| `data/equations.txt` | the 4,694 ETP laws, hash-pinned. |
| `tests/` | round-trip, no-leakage, determinism, coverage, and grader tests. |
| `experiments/` | the committed lab notebook, one directory per experiment. |

Three presentation arms are compared throughout:

- **story**: themed narrative, no notation (`--form story`)
- **literal**: plain description, open variables and "the operation" (`--form literal`)
- **two-stage**: story → model rewrites it as a literal description → second call
  formalizes that verbatim (`--form two-stage`)

Sampling never looks at the arm, so the same seed gives the byte-identical pair
set in every arm.

### `informalizing-etp/experiments/` — the lab notebook

Each `NN-slug/` has `EXPERIMENT.md` (question, setup, prompts with hashes,
reproduce commands, results, conclusions), `runs/` (raw `benchmark.py` output:
`run_meta.json`, `samples.jsonl`, `results.jsonl`, `summary.json`, `summary.md`),
and `report/`. Run directories are never edited after the fact. Full conventions in
`experiments/README.md`.

| # | question |
|---|---|
| 01 | pilot: does the end-to-end loop work, can models formalize stories at all |
| 02 | thinking on/off × equation complexity, story arm |
| 03 | the same sweep, literal arm |
| 04 | structured literal (named intermediates in the literal arm) |
| 05 | does literal success predict story success on the same pair |
| 05-two-stage | abstract first, then formalize |
| 06 | frontier saturation check (GPT-5.5) |
| 07 | two-stage at scale, stratified complexity, 9 models |
| 08 | abstraction hints inside the story prompt; thinking-budget sweep |
| 09 | synthetic laws from 1 to 10 operations (past the ETP cap) |
| 10 | let the model invent its own rigid grammar |
| 11, 12 | majority voting, open-weight then frontier |
| 13 | implication-truth task (True/False, graded against Lean outcomes) |
| 14 | complexity sweep with native auto reasoning: token spend vs difficulty |

### `main-results/` — what goes in the paper

`README.md` there maps each headline result to its figure and source experiment:

1. Abstraction helps: two-stage beats single-shot story (exp 07).
2. Reasoning on beats reasoning off in all three arms (exp 05, 09).
3. Reasoning models spend more thinking tokens on story than literal at matched
   problems.
4. Token spend rises with complexity, and auto reasoning removes the complexity
   cliff seen with thinking off (exp 14).

`plot_results1.py`, `make_plots.py`, and `style.py` regenerate `figures/` from the
committed run directories (needs `matplotlib`). Subfolders hold copies of the
relevant `EXPERIMENT.md`, summaries, and reports per result.

### `ft-experiments/` — fine-tuning thread

Does fine-tuning teach translation or just a notation? Two controlled experiments
on Llama-3.1-8B, Ministral-3-14B, Qwen3-32B with identical LoRA config:

- **v1, grammar-only**: train on bare `ASSUME:/ASK:` text. Syntax became perfect,
  correctness did not move, and the 32B collapsed (34% → 4%).
- **v2, task pairs**: train on story → answer pairs in one grammar, evaluate in two
  grammars the model never saw. Near-full transfer to a re-skinned grammar, partial
  transfer to a structurally different one that scales with model size, and an
  input-side overfit on the held-out `tea` theme.

Start with `ft-experiments/README.md`, then `DESIGN.md` (pre-registered) and
`RESULTS.md`. Notebooks are the readable path. Training and eval run on Modal;
checkpoints are on HuggingFace (`SoHarshh/mars-v-ft-checkpoints`). Frozen data
lives in `eval_v1/`, `train_v1/`, `train_v2/`.

### `probe-experiments/` — interpretability thread

Is the correctness of a translation linearly readable in the residual stream, and
does that direction do anything causally? Stages: `data-gen/` (contrast_v1: correct
vs one-edit-wrong pairs) → `capture/` → `probing/` → `steering/` → `analysis/`.
Results in `RESULTS.md` / `REPORT.html`. Headline: correctness is linearly decodable
but the direction is orthogonal to the model's own yes/no axis and causally inert;
asking the model a verification question routes it onto that axis.

### `paper/`

`manuscript.tex` plus `checklist.tex`, `references.bib`, the NeurIPS style files,
and figures (`figures/*.svg` from `main-results/`, plus the raster copies the
manuscript includes).

## Quickstart

Rendering and grading need only Python 3.

```sh
cd informalizing-etp

# render a story (theme picked by hash), or force one
python3 storyform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x"
python3 storyform.py "x = x ◇ x" "x ◇ y = y ◇ x" --theme tea

# the literal arm of the same pair
python3 literalform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x"

# write a corpus record, print the filled prompt, grade a raw response
python3 storyform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x" \
    --e-label E387 --f-label E43 --out-dir corpus
python3 checkform.py prompt corpus/E387-E43-paint.json
python3 checkform.py grade  corpus/E387-E43-paint.json response.txt   # exit 0/1/2

# tests
python3 -m unittest discover -s tests
```

Running a benchmark needs `OPENROUTER_API_KEY`. Always dry-run first: it runs the
whole pipeline offline with the reference answer filled in and must grade 100%.

```sh
python3 benchmark.py --dry-run --seed 0 --stratify-ops 5 --out-dir results/dry

python3 benchmark.py --seed 0 --stratify-ops 20 --form story   --reasoning off \
    --out-dir experiments/NN-slug/runs/run-strat20-s0-story-think-off
python3 benchmark.py --seed 0 --stratify-ops 20 --form literal --reasoning off \
    --out-dir experiments/NN-slug/runs/run-strat20-s0-literal-think-off

python3 charts.py experiments/NN-slug/runs/* \
    --out experiments/NN-slug/report/report.html --pdf
```

`--n N` samples uniformly; `--stratify-ops N` takes N pairs per total-operation bin
so difficulty curves resolve (uniform sampling lands almost entirely on 4+4-op
pairs). Rerunning the same `--out-dir` only retries api-error rows.

## Standing rules

These bind everything in the repo; see `CLAUDE.md` for the full list.

- Mechanical ground truth only. No LLM labels or judges any example.
- Vacuous laws (`x = x`, `x = y`) are excluded from every sample.
- Translation (checkform-graded) and implication-truth (True/False) are different
  tasks with different oracles; their numbers never share a table.
- Grader leniency is identical across arms and grammars.
- One variable per experiment, fixed seed 0, N reported everywhere.
- Frozen artifacts (`eval_v1`, `train_v1`, `train_v2`, `contrast_v1`, run
  directories) are never regenerated or edited; a change means a new version and
  a downstream rerun.
- Never train against the checker.

## Where to read next

- New to the benchmark: `informalizing-etp/README.md`, then
  `informalizing-etp/experiments/README.md`, then `experiments/07-two-stage-scale/EXPERIMENT.md`.
- Want the results: `main-results/README.md` and `paper/manuscript.tex`.
- Fine-tuning or probing: the `README.md` and `RESULTS.md` in each of those folders.
- Why a decision was made: `RESEARCH_LOG.md`.
