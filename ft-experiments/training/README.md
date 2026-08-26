# training

LoRA fine-tuning of Llama-3.1-8B on the grammar-only train_v1 corpus:
plain next-token prediction on raw RG text (manual pack-1024 + EOS, loss
on all tokens, no chat template), bf16 base on Modal A10G, PEFT adapters
checkpointed to the Volume. The sweep is data in `config.py`
(RANKS/LAYER/SEEDS, alpha=r); the notebook is a face over
`train_lora.py` — it never forks logic.

**Inputs:** `../train_v1/` (train + holdout), `../config.py` +
`training/config.py`. **Outputs:** adapters at
`/models/checkpoints/<run_name>/{step-0,step-N,...,final}` on the Volume;
run record at `../runs/ft-v1/train-<run_name>.json` (config, loss curve,
trainable-param count, sanity probes, versions).

Commands:

```sh
bash training/run_train.sh 8b --preset smoke      # Phase 4: 200 samples, 50 steps, r=1, layer 16
bash training/run_train.sh 8b --preset phase5a    # Phase 5a: r=16, all layers, 3 epochs

# Phase 5b rank sweep — same single layer 16, rank varies, NOTHING else:
bash training/run_train.sh 8b --rank 1  --layer 16 --seed 0
bash training/run_train.sh 8b --rank 2  --layer 16 --seed 0
bash training/run_train.sh 8b --rank 8  --layer 16 --seed 0
bash training/run_train.sh 8b --rank 16 --layer 16 --seed 0
bash training/run_train.sh 8b --rank 32 --layer 16 --seed 0
bash training/run_train.sh 8b --rank 64 --layer 16 --seed 0

# seeds for the headline configs (Phase 5a config, rank 1, best rank):
SEEDS="0 1 2" bash training/run_train.sh 8b --rank 1 --layer 16
```

Every run asserts trainable-param isolation (single-layer runs must have
zero parameters outside the target block) and runs two sanity probes
(holdout-RG perplexity FT<base; raw `ASSUME:…\nASK:` completion must
parse under checkform). Interpret eval flatness only after probes pass —
see the format-bridge note in `../../CLAUDE.md`.
