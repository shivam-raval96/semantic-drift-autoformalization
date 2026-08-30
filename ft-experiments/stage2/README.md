# stage2

Staged-vs-direct **translation** fine-tuning, built on stage-1 familiarity.
Stage 1 taught a model to recognize three rigid grammars by opaque label
(recognition only). Stage 2 asks whether that familiarity helps the model
*translate* stories into a grammar, and how far a single translation skill
carries to an unseen grammar and an unseen story theme.

Base model is **Llama-3.1-8B-Instruct** everywhere. F0 = the stage-1 adapter
(`stage1-8b-s0`). Staged runs start from **F0 merged into the base**, then a
fresh stage-2 LoRA on top; the direct run starts from the plain base. That is
the only difference between the two arms: did the starting weights already have
familiarity.

## Grammars

Same three from stage 1 plus a **held-out fourth**:

- `RG-1` = `a` prefix `op(a, b)` — the trained translation target
- `RG-2` = `b_near` prefix `f(a, b)` — added in the T2 curriculum
- `RG-3` = `b_far` infix `(a ∘ b)`
- `RG-4` = `sexpr` S-expression prefix `(⋆ left right)` — **never trained**,
  authored only as a prompt template + parser for the unseen-grammar eval

RG-4 lives in `../eval/grammars.py` (serializer + S-expression parser, graded
free through the `grade_b` transcode seam) and its prompt template is
`stage2/prompts/formalize_prompt_rg4.md` (kept in our tree, not Oren's).

## The runs

| run | start from | trains | data | purpose |
|---|---|---|---|---|
| **F0** | (stage-1 adapter) | — | — | familiarity-only baseline (exp-1) |
| **T1** | F0-merged base | RG-1, themes minus tea | `train_v2/train.jsonl` | hub: staged arm + generalization |
| **T2** | continues T1 | + RG-2 | `stage2/train_v2_rg2.jsonl` | two grammars vs one |
| **T3** | F0-merged base | RG-1, single theme (signal) upsampled | `stage2/train_v2_rg1_singletheme.jsonl` | one theme vs many |
| **D0** | plain base | RG-1, themes minus tea | `train_v2/train.jsonl` | direct arm (no familiarity) |

`tea` is the held-out theme (neither F0 nor `train_v2` ever saw it). T3 matches
T1's example budget by upsampling its single theme's 984 pairs to 2,772.

## Named, frozen datasets

- **`trans_eval_v1/`** — the 777 `eval_v1` problems, each rendered into all four
  grammars (`ref_rg1..ref_rg4`) with `canonical_e/f` for grading. One artifact
  serves every eval: exp-1 uses RG-1/2/3, stage 2 uses RG-1 (in-grammar) and
  RG-4 (unseen), theme generalization uses the `theme == tea` slice (190 rows).
- **`stage2/train_v2_rg2.jsonl`** — `train_v2` pairs, completions re-rendered
  into RG-2, for the T2 continuation.
- **`stage2/train_v2_rg1_singletheme.jsonl`** — one theme upsampled to 2,772,
  for T3.

Build + gate (deterministic; rebuilds are byte-identical):

```sh
python3 data-gen/build_trans_eval.py    # -> trans_eval_v1/{eval.jsonl,manifest.json}
python3 data-gen/verify_trans_eval.py   # gate -> TRANS-EVAL VERIFIED
python3 data-gen/build_stage2.py        # -> stage2/train_v2_rg2.jsonl + _singletheme.jsonl
python3 data-gen/verify_stage2.py       # gate -> STAGE2-TRAIN VERIFIED
```

## Run it (Lambda, one A100-40GB or A10G-24GB; 8B LoRA)

```sh
pip install -r requirements-lambda.txt
export HF_TOKEN=...            # gated Llama base + your stage-1/stage-2 repos

# 0. F0 baseline (exp-1): familiarity-only translation into RG-1/2/3
hf download deenais/mars-v-stage1 --include 'stage1-8b-s0/final/*' --local-dir checkpoints
python3 eval/stage2_eval_lambda.py --run-name f0 \
    --adapter checkpoints/stage1-8b-s0/final --grammars rg1,rg2,rg3

# 1. merge F0 into the base (shared start for T1/T2/T3)
python3 training/merge_adapter.py \
    --adapter-repo deenais/mars-v-stage1 --adapter-subfolder stage1-8b-s0/final \
    --out-dir checkpoints/llama-3.1-8b_f0-merged

# 2. T1 (staged, RG-1) then continue to T2 (RG-2)
python3 training/train_stage2_lambda.py --run-name t1-8b \
    --base checkpoints/llama-3.1-8b_f0-merged \
    --grammar rg1 --data train_v2/train.jsonl --hf-repo <you>/mars-v-stage2
python3 training/train_stage2_lambda.py --run-name t2-8b \
    --base checkpoints/llama-3.1-8b_f0-merged \
    --grammar rg2 --data stage2/train_v2_rg2.jsonl \
    --resume-adapter checkpoints/t1-8b/final --hf-repo <you>/mars-v-stage2

# 3. T3 (staged, single theme) and D0 (direct, plain base)
python3 training/train_stage2_lambda.py --run-name t3-8b \
    --base checkpoints/llama-3.1-8b_f0-merged \
    --grammar rg1 --data stage2/train_v2_rg1_singletheme.jsonl --hf-repo <you>/mars-v-stage2
python3 training/train_stage2_lambda.py --run-name d0-8b \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --grammar rg1 --data train_v2/train.jsonl --hf-repo <you>/mars-v-stage2

# 4. eval matrix (engine loads once per model; grammars loop)
python3 eval/stage2_eval_lambda.py --run-name t1-8b --adapter checkpoints/t1-8b/final --grammars rg1,rg4
python3 eval/stage2_eval_lambda.py --run-name t2-8b --adapter checkpoints/t2-8b/final --grammars rg4
python3 eval/stage2_eval_lambda.py --run-name t3-8b --adapter checkpoints/t3-8b/final --grammars rg1
python3 eval/stage2_eval_lambda.py --run-name d0-8b --adapter checkpoints/d0-8b/final --grammars rg1
```

Every eval writes `runs/stage2/eval-<run>-<label>.json` with overall +
per-tier + per-theme verdicts (correct / wrong / unparseable). Training loss
curves come free in each run's `record.json`.

The eval loads each adapter on the base it was **trained** against: it reads
`base_model_name_or_path` from the adapter's `adapter_config.json`, so the
staged adapters (T1/T2/T3) automatically evaluate on the F0-merged base rather
than plain Llama (evaluating them on plain Llama would silently drop F0).
Override with `--model-id <path-or-hub-id>` if the merged dir has moved.

## What each result answers

- **exp-1 (F0)** — does familiarity alone confer any translation ability?
  (expected low; it is the baseline the staged arm improves on)
- **staged vs direct** — T1 vs D0 into RG-1: accuracy and loss curves, with and
  without stage-1 familiarity as the starting point
- **grammar generalization** — T1 into RG-4 (unseen grammar, rules given in the
  prompt); T2 into RG-4 asks whether two trained grammars carry further than one
- **theme generalization** — T1 vs T3 into RG-1 on the `tea` slice: does one
  theme generalize as well as many

## Success metric

Cross-grammar / cross-theme translation accuracy (checkform verdict, `correct`),
base vs FT, sliced by tier and theme. N=777 per cell (190 for the tea slice);
tiers are 180–200 items, so ±3–4 points is noise.
