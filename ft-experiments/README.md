# ft-experiments — grammar-only fine-tuning (MARS V)

Does grammar-only LoRA fine-tuning improve story→RG / literal→RG / two-stage
translation on unseen equations? Base vs FT compared on Oren's exact experiment
setup (his renderers, prompts, grader — `../informalizing-etp/`), evaluated on
the frozen SAIR-derived eval_v1, trained only on law-disjoint RG grammar text.

```
prep.sh ──> data-gen ──> eval (base) ──> training ──> eval (checkpoints) ──> analysis
```

Reproduce, in order (each line is a real command):

```sh
bash prep.sh                                       # venv + deps + modal auth check
# data (already frozen; rebuild is byte-identical — see data-gen/README.md)
python3 data-gen/verify_artifacts.py               # gate: must print ALL CHECKS PASSED
# base evals (score A)
bash eval/run_eval.sh 1b story                     # …and literal / two-stage; same for 8b
python3 eval/base_table.py                         # per-tier three-way base table
# fine-tune (Phase 4/5a presets or explicit sweep points)
bash training/run_train.sh 8b --preset phase5a
# FT-checkpoint evals (score B) + comparison
bash eval/run_eval.sh 8b story --adapter /models/checkpoints/phase5a-r16-all/final --adapter-rank 16
python3 eval/compare_table.py                      # base-vs-FT headline table
# analysis notebooks: analysis/visuals.ipynb · analysis/qualitative.ipynb
```

Stage READMEs are runbooks: `data-gen/` (SAIR fetch → eval_v1/train_v1 →
verification), `eval/` (every eval variant as a literal command), `training/`
(presets + the full rank sweep written out), `analysis/` (plots, qualitative
browser, planned mechanistic files). The registry `config.py` is the single
source for models, paths, protocol, guardrails; run identity = CLI args +
`run_meta.json`, never edited config lines. Artifacts (`data/ eval_v1/
train_v1/ runs/`) are frozen — `eval_v1` changes mean eval_v2 + rerun
everything. Experiment spec and standing rules: `../CLAUDE.md`.
