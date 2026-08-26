# REFACTOR SPEC — make ft-experiments/ runnable like the vlm-alignment repo

Context: I have fully reviewed https://github.com/idhantgulati/vlm-alignment (Shivam's paper repo).
Adopt its runnability patterns, listed explicitly below, WITHOUT inheriting its flaws, and WITHOUT
touching artifact bytes or in-flight runs. This spec is the authority for the refactor.

## Hard constraints (before anything)
1. Zero artifact changes: data/, eval_v1/, train_v1/, runs/ stay byte-identical; verify_artifacts
   must pass after the refactor exactly as before.
2. Use `git mv` for all moves; one refactor commit series, no logic changes mixed in.
3. Do not disturb the two-stage base evals currently running; reconcile their output paths after
   they land. All existing run_meta.json contents remain valid (add fields, never rename).
4. Frozen things stay frozen: prompt templates, eval protocol, greedy/temp-0, template SHAs.

## Target layout (mirrors the reference repo's stage-based design, mapped to our phases)

```
ft-experiments/
├── README.md                     # top-level: pipeline diagram + per-stage one-liners + repro order
├── requirements.txt              # root deps (mirror reference: one honest list, no pins we don't need)
├── prep.sh                       # uv venv + install + modal token check; idempotent; ends "Done"
├── config.py                     # THE registry (spec below)
│
├── data-gen/                     # ≈ their syn-data-gen/
│   ├── config.py                 # stage knobs only: seeds, tier quotas, depth ranges, paths via root config
│   ├── sair_fetch.py  build_sair_index.py  build_eval.py  build_train.py  verify_artifacts.py
│   ├── io/sample-train-row.json  io/sample-eval-row.json     # data contract by example (their io/ pattern)
│   └── README.md                 # exact commands, in order, copy-pasteable
│
├── eval/                         # ≈ their em-judge/
│   ├── config.py                 # eval knobs: arms, max_tokens per arm, timeouts, batch size
│   ├── modal_eval.py             # ONE runner, parameterized: --model {1b,8b,70b} --arm {story,literal,twostage}
│   │                             #   --adapter <path-or-hf-id>   (absent = base model)
│   ├── base_table.py             # assembles per-tier three-way tables from runs/
│   ├── run_eval.sh               # bash eval/run_eval.sh 8b story          (base)
│   │                             # bash eval/run_eval.sh 8b story --adapter runs/train/r1-l16-s0/ckpt-500
│   └── README.md                 # runbook: every eval variant as a literal command (their README-as-runbook)
│
├── training/                     # ≈ their root LoRA notebooks, plus script twin
│   ├── config.py                 # sweep definition lives HERE as data: RANKS=[1,2,8,16,32,64], LAYER=K,
│   │                             #   SEEDS=[0,1,2], LR, steps, alpha=r rule (their "rank==alpha" convention,
│   │                             #   documented in a comment like their notebook markdown cell)
│   ├── train_lora.py             # headless runner: --model 8b --rank 1 --layer 16 --seed 0
│   ├── train_lora.ipynb          # notebook face: header cell (goal + dataset link + source attribution,
│   │                             #   their style), then ONE clearly-marked CONFIG cell ("# just change the
│   │                             #   rank here" — their exact ergonomic), then cells that call into
│   │                             #   train_lora.py functions — the notebook never duplicates logic
│   ├── run_train.sh              # bash training/run_train.sh 8b --rank 1   (loops seeds if asked)
│   └── README.md
│
├── analysis/                     # ≈ their subspace-analysis/, prepared for our Phase 6+
│   ├── visuals.ipynb             # rank-vs-performance plot, per-tier deltas (their visuals.ipynb role)
│   ├── qualitative.ipynb         # base-vs-FT output pairs browser (their chat-ui.ipynb role)
│   └── README.md                 # notes future files: activation_extraction.py, svd.py, steering.py,
│                                 #   steering_inf.py — adopting their exact decomposition when we get there
│
└── data/  eval_v1/  train_v1/  runs/     # unchanged artifacts; all path references via root config.py
```

## Root config.py — the single registry (their pattern, minus their flaws)
```python
MODELS = {
  "1b":  {"hf_id": "unsloth/Llama-3.2-1B-Instruct",       "gpu": "A10G", "tp": 1},
  "8b":  {"hf_id": "unsloth/Meta-Llama-3.1-8B-Instruct",  "gpu": "A10G", "tp": 1},
  "70b": {"hf_id": "unsloth/Meta-Llama-3.3-70B-Instruct", "gpu": "H100:2", "tp": 2},  # gated: user go only
}
PATHS = { ... every artifact dir, prompts dir, runs dir ... }        # nothing path-like hardcoded elsewhere
EVAL  = { "temperature": 0.0, "max_tokens": 4096, "max_model_len": {"single": 8192, "twostage": 12288},
          "timeouts": {"single": 900, "twostage": 1800}, "template_shas": {...} }
GUARDRAILS = { "min_containers": 0, "scaledown_window": 120, "retries": 0 }
```
Rules (anti-flaw clauses, from my review of their config.py):
- NO commented-out filename graveyards as state. Run identity comes from CLI args + run_meta, never
  from hand-edited config lines.
- NO configuration encoded in magic output filenames. Run dirs are runs/<stage>/<model>-<arm>-<tag>/
  with the full config inside run_meta.json.
- Adding a model = adding ONE dict entry; every script resolves through the registry.

## Script conventions (their good habits, made uniform)
- Every runnable script: module docstring stating exactly what it does (their scripts do this),
  argparse with sensible defaults pulled from configs (their extract.py pattern), loud startup echo of
  the resolved config (their "=" banner in extract.py), and writes/updates run_meta.json.
- Base vs FT symmetry, improved: instead of their duplicated base_model_inference.py /
  ft_model_inference.py twins, ONE modal_eval.py where base = no --adapter. Structural fairness with
  zero code duplication. The adapter path/HF-id is stamped into run_meta.
- Incremental, resumable output writing for anything long-running (their filelock + per-item merge
  pattern): an interrupted eval or training job resumes without rerunning finished items.
- One output schema for ALL evaluated things — base, FT checkpoints, and (later) steered models emit
  identical per-row records so base_table.py and the analysis notebooks score everything through one
  path (their steering_inf.py "schema compatible with em-judge" trick, adopted project-wide).

## README conventions (their strongest feature)
Every stage README contains, in order: (1) what this stage does in two sentences, (2) inputs it
assumes exist and outputs it produces, (3) the literal copy-paste commands for every variant —
including the full future sweep, written out per rank like their em-judge README lists every vLLM
serve command per rank. The top-level README shows the whole pipeline as: prep.sh → data-gen →
eval (base) → training → eval (checkpoints) → analysis, each line being an actual command.

## Notebook conventions (their style)
Header markdown cell: purpose, dataset link, source attribution. One CONFIG cell near the top with
the single-knob comment. A markdown cell stating the sweep and the alpha=r convention. Inference
smoke-test cells at the end (their pattern of testing the model inline post-training). Notebooks call
library functions; they never fork logic from the .py twins.

## Also add (things they lack that we already do better — keep ours)
- verify_artifacts stays a first-class stage gate; hash manifests stay authoritative.
- run_meta.json completeness (GPU, image, vLLM version, template SHAs, guardrails) is non-negotiable.
- .gitignore: adopt their artifact-exclusion discipline (outputs*/, wandb/, checkpoints, caches) so
  bulky run outputs never enter git; committed artifacts remain the manifests + small tables.

## Order of work
1. Refactor + git mv + configs + .sh entry points + READMEs + io/ samples. Run verify_artifacts.
2. Reconcile the in-flight two-stage run outputs into the new runs/ layout when they land.
3. Regenerate the base table via the new entry point to prove the refactor is behavior-neutral
   (byte-compare against the committed table).
4. Show me: the new tree, one full README, and the output of `bash eval/run_eval.sh 1b story --limit 5`
   run through the new path. STOP there.
