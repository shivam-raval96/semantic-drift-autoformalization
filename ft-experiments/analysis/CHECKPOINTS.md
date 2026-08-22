# Fine-tuning checkpoints

LoRA adapters saved during training, for trajectory / PCA analysis.

## What's here

| run | model | checkpoints | recipe |
|---|---|---|---|
| `v2-8b-s0` | meta-llama/Llama-3.1-8B-Instruct | 13 (step-0 … 1041, final) | v2: story→RG pairs, completion-only loss |
| `v2-ministral-14b-s0` | mistralai/Ministral-3-14B-Instruct-2512-BF16 | 13 (step-0 … 1041, final) | v2, same |
| `v2-qwen3-32b-s0` | Qwen/Qwen3-32B | 10 (step-0 … 900) | v2, same |
| `phase5a-r16-all` | meta-llama/Llama-3.1-8B-Instruct | 10 (step-0 … 75) | v1: bare grammar text, loss on all tokens |
| `phase5a-32b-r16-all` | Qwen/Qwen3-32B | 10 (step-0 … 75) | v1, same |

Every run includes **step-0** (adapter initialized, nothing trained yet), so each
trajectory has its own origin.

All runs: LoRA r=16, alpha=16, dropout 0, on `q,k,v,o,gate,up,down_proj` in every
layer. lr 2e-4 cosine, warmup 3%, seed 0. v2 = 3 epochs over 2,772 pairs
(~1041 steps, saved every 100); v1 = 3 epochs over 2,772 grammar-text samples
(75 steps, saved every 10).

Two recipes exist for both the 8B and the 32B, so v1-vs-v2 is a controlled
comparison on the same model: same equations, same LoRA config, only the training
distribution differs. Behavioral outcome: v1 perfects syntax and loses semantics
(32B 34% → 3% correct), v2 teaches the task (32B 34% → 99.7%). See `../RESULTS.md`.

## Getting the weight change

Each checkpoint is a PEFT adapter — `adapter_model.safetensors` + `adapter_config.json`,
448 tensors for the 8B (~168 MB). The learned change for a module is:

```python
from safetensors import safe_open

with safe_open("step-500/adapter_model.safetensors", framework="pt") as sf:
    A = sf.get_tensor("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
    B = sf.get_tensor("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight")

dW = (B.float() @ A.float()) * (16 / 16)      # alpha / r
```

`dW` is rank-16 by construction, so the factors *are* the compact form — for PCA
across checkpoints you can work in the factored space rather than materializing
dense deltas (an 8B checkpoint is 168 MB factored, ~2 GB dense).

Or load onto the base model directly:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base, "step-500")
```

## trajectory.npz

Each run directory has one, written by `export_checkpoints.py`:

```python
import numpy as np

d = np.load("v2-8b-s0/trajectory.npz", allow_pickle=True)
d["steps"]    # (13,)          optimizer step per checkpoint
d["modules"]  # (448,)         module names, e.g. "model.layers.0.self_attn.q_proj"
d["norms"]    # (13, 448)      ||dW||_F per module per checkpoint
```

Enough to plot where and when the model changes without loading any adapter.

## Regenerating

```bash
python3 analysis/export_checkpoints.py --all --out ~/checkpoints-share
```

Pulls from the Modal volume `harsh-ft-grammar-weights` (path
`checkpoints/<run>/<step>/`) and rebuilds the npz files.

## Context worth knowing

- `v2-qwen3-32b-s0` stops at step-900, not a 3-epoch `final` — its training
  container was lost to a network failure at ~2.6 epochs. Every 32B number in
  RESULTS.md is from step-900. Its own checkpoint curve shows grammar A at 100%
  from step 500 onward, so the missing steps had nothing left to move.
- The v2 checkpoint curves show transfer to an unseen grammar peaking early and
  then decaying (14B: 74.5% at step 300 → 59% at step 900) while the trained
  grammar keeps improving — so the intermediate checkpoints are not
  interchangeable with the final one.
