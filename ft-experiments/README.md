# Does fine-tuning teach translation, or just a notation?

Part of MARS V (semantic faithfulness in autoformalization). We render Lean-verified
equational-theory implications into stories, ask a model to translate each story back into a
rigid formal grammar, and grade it mechanically — no LLM judges anywhere.

Two experiments live here:

- **v1 — grammar-only.** Fine-tune on bare `ASSUME:/ASK:` notation, nothing else. Does notation
  fluency transfer to translation?
- **v2 — task pairs.** Fine-tune on story → answer pairs, then ask for the answer in grammars the
  model never trained on. Did it learn to translate, or to emit one format?

## Findings

- **v1: grammar was not the bottleneck.** Syntax became perfect (unparseable 45% → 0%) and
  correctness did not move. On Qwen3-32B it *collapsed*, 34% → 3% — the model learned to produce
  fluent notation instead of reading the story.
- **v2: task pairs teach the skill.** 0.1 → 96.8% (8B), 14.0 → 89.4% (14B), 18.8 → 99.7% (32B) on
  the trained grammar, and it holds in a grammar with different keywords and operator (94–100%)
  and on the literal-NL input arm, which was never trained.
- **The limit is structural, and it is a capacity limit.** A restructured (infix) grammar reaches
  37% / 50% / 80% across 8B / 14B / 32B. The failures are syntactic — of the 14B's 152 failures on
  the deepest tier, 119 are unbalanced parentheses and 5 are wrong mathematics.
- **Transfer peaks early, then training specializes it away.** The 14B is level across grammars at
  step 100, peaks at 74.5% on the unseen grammar at step 300, and decays to 59% by the end while
  the trained grammar keeps climbing. The effect vanishes at 32B.
- **Fine-tuning installs an internal representation.** A correctness probe on hidden states goes
  0.52 → 0.95 (8B), 0.61 → 0.98 (14B), 0.60 → 0.92 (32B) AUROC on held-out equation families.
  v1's grammar-only training left it untouched.

Full tables per tier: [`RESULTS.md`](RESULTS.md). Pre-registered design and kill criteria:
[`DESIGN.md`](DESIGN.md).

## Setup

```bash
bash prep.sh
```

Needs a Modal account (`modal token new`) and an HF token with Llama access. Everything except
training and eval runs on a laptop.

## Where to start

The notebooks are the readable path through the project:

| notebook | what it does | needs |
|---|---|---|
| `data-gen/build-data.ipynb` | look at the training data, the eval set, and the three grammars; check train/eval disjointness | laptop |
| `train-pairs-8b.ipynb` | fine-tune the 8B end to end — inspect the masked batches, launch on Modal, plot the loss, grade the model's own output | Modal A100 |
| `analysis/results.ipynb` | the results table, per-tier numbers, checkpoint curves, and the raw model outputs behind them | laptop |

## Repository structure

```
├── config.py                     # models, paths, eval protocol, guardrails — the one registry
├── prep.sh                       # venv + deps + modal auth check
├── DESIGN.md                     # v2 spec: grammars, predictions, kill criteria (written first)
├── RESULTS.md                    # full results, both experiments
├── watch.sh                      # tail the live event log of a run
│
├── train-pairs-8b.ipynb          # v2 fine-tune, start to finish
│
├── data-gen/
│   ├── build_train.py            #   v1 corpus: bare grammar text
│   ├── build_train_v2.py         #   v2 corpus: story -> answer pairs (same equations, re-rendered)
│   ├── build_eval.py             #   eval_v1: 777 frozen problems, 4 tiers
│   ├── build_sair_index.py       #   hash index used for the train/eval disjointness gate
│   ├── verify_artifacts.py       #   independent re-derivation of every frozen artifact
│   ├── ftlib.py                  #   serializers + symmetry-aware hashing
│   └── build-data.ipynb
│
├── training/
│   ├── train_pairs.py            #   v2: completion-only loss on story -> answer pairs
│   ├── train_lora.py             #   v1: plain next-token loss on grammar text
│   └── config.py                 #   LoRA rank, lr, target modules, presets
│
├── eval/
│   ├── modal_eval.py             #   the eval runner: vLLM greedy, chunked + resumable
│   ├── grammars.py               #   the two never-trained grammars + their graders
│   ├── curve_eval.py             #   checkpoint sweeps: one engine load, adapters hot-swapped
│   ├── format_control.py         #   unrelated format-following tasks (is the model just narrower?)
│   ├── test_grammars.py          #   round-trip and cross-grammar tests
│   └── base_table.py, compare_table.py
│
├── analysis/
│   ├── make_figures.py           #   the two figures in RESULTS.md
│   ├── floor_samples.py
│   └── results.ipynb
│
├── eval_v1/  train_v1/  train_v2/  data/    # frozen, sha-pinned; generators are committed, bulk data is not
├── runs/                                    # every run: results.jsonl, summary.json, run_meta.json
└── assets/                                  # figures and the shareable reports
```

## Reproducing a run

```bash
# data (deterministic — same shas every time)
python3 data-gen/build_train_v2.py
python3 data-gen/verify_artifacts.py

# train
modal run training/train_pairs.py --preset smoke-v2          # 50 steps, sanity
modal run training/train_pairs.py --preset v2-8b             # full, ~65 min on an A100
TRAIN_MODEL_ID="Qwen/Qwen3-32B" modal run training/train_pairs.py --preset v2-32b

# eval: base, then the adapter, on any grammar
modal run eval/modal_eval.py --model 8b --arm story
modal run eval/modal_eval.py --model 8b --arm story-bfar \
  --adapter /models/checkpoints/v2-8b-s0/final --out-tag ft-v2

# checkpoint curve (one engine load for all checkpoints)
modal run eval/curve_eval.py --model 8b --run-name v2-8b-s0 \
  --checkpoints step-0,step-100,step-300,step-500,step-700,step-900 \
  --arms story,story-bfar --limit 200 --out-prefix curve-v2-v2-8b-s0

python3 -m pytest eval/test_grammars.py
```

Arms are `story`, `literal`, and the same two with `-bnear` / `-bfar` for the never-trained
grammars. Adding a model is one entry in `config.py`.
