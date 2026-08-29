# stage1

Label-conditioned rigid-grammar familiarity, **recognition only** — the first
stage of the two-stage faithfulness-transfer experiment. A model learns to
associate an opaque label with each of the three RGs by two tasks, never by
producing a grammar (so it is not v1's grammar-only run and not v2's
translation run):

- **identify** — statement -> its label (`RG-1`/`RG-2`/`RG-3`)
- **validate** — statement + label -> `Yes`/`No` (negatives: wrong grammar,
  malformed syntax, mismatched line labels)

Labels are opaque on purpose (nothing in a prompt says what `RG-1` means), so
the association is learned, not read. Structural keys stay code-side:
`RG-1`=`a` (prefix `op`), `RG-2`=`b_near` (prefix `f`), `RG-3`=`b_far` (infix
`∘`); `RG-4` is reserved and unauthored (a stage-2 held-out grammar).

**Inputs:** train_v2 pairs (already SAIR-pair- and eval_v1-law-disjoint),
re-serialized through `../eval/grammars.py`. **Outputs:** `train.jsonl` (1,800),
`eval.jsonl` (495), `manifest.json`. Objective is completion-only (masked
prompt), r=16 all-layer LoRA — the v2 recipe, for comparability.

The eval split is held out by **law class**: a pair joins the eval pool only if
BOTH its laws fall in a deterministic ~30% hash bucket, and the train pool only
if NEITHER does (straddlers dropped), so recognition is measured on laws never
trained. Balanced: identify 300/300/300 per label; validate 50/50 Yes/No;
negatives even across the three corruption types.

## Build + gate (deterministic; rebuilds are byte-identical)

```sh
python3 data-gen/build_stage1.py     # -> stage1/{train,eval}.jsonl + manifest.json
python3 data-gen/verify_stage1.py    # gate — must print STAGE-1 CORPUS VERIFIED
```

`verify_stage1.py` re-derives every claim independently: determinism, balance,
train/eval law+pair disjointness, and per-row correctness (each identify
statement parses under its own grammar and no other; every `Yes` parses, every
`No` fails its parser).

## Train + eval

Two runtimes, one recipe and one grader. **Modal** (Harsh's workflow):

```sh
python3 training/train_stage1.py                       # 8B, checkpoints to the Volume
modal run eval/stage1_eval.py --model 8b               # base control
modal run eval/stage1_eval.py --model 8b --adapter /models/checkpoints/stage1-8b-s0/final
```

**Lambda** (in-process on the box GPU; ephemeral, so checkpoints push to HF):

```sh
pip install -r requirements-lambda.txt
export HF_TOKEN=...                                    # gated Llama base + your repo
python3 training/train_stage1_lambda.py --hf-repo <you>/mars-v-stage1
python3 eval/stage1_eval_lambda.py                     # base control
python3 eval/stage1_eval_lambda.py --adapter checkpoints/stage1-8b-s0/final
```

## HF checkpoint layout

The Lambda trainer creates the repo private and pushes as it trains, mirroring
`analysis/CHECKPOINTS.md`: one folder per run, `step-N/` adapters plus a
`record.json` (config, loss curve, base-vs-FT probe).

```
<you>/mars-v-stage1
└── stage1-8b-s0/
    ├── step-0/  step-100/ … final/     # PeftModel.from_pretrained targets
    └── record.json
```

## Success metric

Held-out recognition accuracy (identify top-1 + validate Yes/No), per label and
task, base vs FT. Opaque labels => the base control should sit near chance on
identify; the FT gain over base is the stage-1 signal that stage 2 builds on.
Out of scope here: grammar production, RG-4, the two-stage-vs-direct comparison,
and the 32B ability-damage check.
