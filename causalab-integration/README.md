# causalab integration — ETP implication task

Source of truth for the `etp_implication` task for
[Goodfire's causalab](https://github.com/goodfire-ai/causalab), the
causal-abstraction framework (interchange interventions, DAS/DBM/PCA/SAE
subspaces, activation manifolds, manifold steering). The task turns our
certified implication data into causalab's native format so its analysis
chain runs on our stimuli unchanged.

causalab is an external repo, so the task lives here and is copied in:

```bash
./install.sh /path/to/causalab          # copies task + configs into the checkout
cd /path/to/causalab && uv sync         # once; installs torch etc.
./scripts/run_exp.sh etp_8b_pipeline    # baseline -> subspace -> activation_manifold
```

## The task in one paragraph

A premise law and a conclusion law over a magma are rendered in one of
several registers (`formal` equation, explicit `instance` prose,
`paraphrase`, `named`); the model answers True/False whether the premise
implies the conclusion. Ground truth is a certificate lookup — 20,563
certified ordered pairs over 157 laws (21 base + 32 per-op-bin synthetic
+ 104 derived substitution instances; 398 implies, 20,165
non-implications via finite countermodels; uncertified pairs excluded,
never guessed), stratified into four complexity bins by op count. The `template`
causal variable IS the register, so cross-register interventions are
native template interventions. Counterfactual generators map to the
project's research questions: `flip_premise`/`flip_conclusion` (does the
located representation causally drive the verdict?), `cross_register`
(is law identity register-invariant? — the story/literal/formal question
in miniature).

## Layout

| Path | What |
|---|---|
| `tasks/etp_implication/` | the task package (mirrors `causalab/tasks/<name>/` conventions; see its README for variables, generators, and the certified-sampling constraint) |
| `configs/task/etp_implication.yaml` | Hydra task config |
| `configs/runners/etp_implication/etp_8b_pipeline.yaml` | Llama-3.1-8B pipeline runner |
| `scripts/export_task_data.py` | regenerates `tasks/etp_implication/data/etp_pairs.json` |
| `install.sh` | copies everything into a causalab checkout |

## Regenerating the data

`export_task_data.py` needs `laws.py` and `magma.py` from this repo's
`certificate-pipeline` branch next to it:

```bash
git show certificate-pipeline:pipeline/laws.py  > /tmp/etp_export/laws.py
git show certificate-pipeline:pipeline/magma.py > /tmp/etp_export/magma.py
cp causalab-integration/scripts/export_task_data.py /tmp/etp_export/
cd /tmp/etp_export && python3 export_task_data.py   # prints counts + sha256
```

The JSON's `provenance` field records the generation convention
(exhaustive countermodel search ≤3, sampled n=4; degenerate laws and
self-pairs excluded). Treat the data file like a frozen instrument: if
you regenerate it, the printed sha256 changes and results across
versions are different experiments.

## Known constraints / next steps

- `task.resample_variable` must stay `"all"`: causalab's single-variable
  resampling bypasses the task generators and can produce uncertified
  pairs, which the mechanism rejects by design (fail loud, never guess).
  This is why the runner chain omits `locate` for now — integrating it
  needs a runner hook that consumes `flip_premise` (already yields
  certified single-variable pairs) or a certified-aware resampler.
- The True side is enriched via derived instance laws, which buys 398
  True pairs at the cost of a measurable lexical-overlap floor (v3: BoW
  AUROC ~0.70 pooled, 0.53-0.66 per complexity bin; see the task
  README). Probes must beat the per-bin floor under leave-pair-out
  splits, within complexity strata.
- Pin the causalab commit you run against and record it next to results
  (the framework is under active development).
