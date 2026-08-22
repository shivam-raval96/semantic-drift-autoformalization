# Fine-tuning checkpoints — MARS V

LoRA adapters saved during training, for trajectory / PCA analysis. Task:
translating a natural-language story into a rigid formal grammar, graded
mechanically.

| folder | base model | trained on | checkpoints |
|---|---|---|---|
| `llama-3.1-8b_grammar-only` | Llama-3.1-8B-Instruct | bare grammar text | 9 (step-0 … 75, every 10) |
| `llama-3.1-8b_task-pairs` | Llama-3.1-8B-Instruct | story → grammar pairs | 12 (step-0 … 1041, every 100) |
| `ministral-3-14b_task-pairs` | Ministral-3-14B-Instruct | story → grammar pairs | 12 (step-0 … 1041, every 100) |
| `qwen3-32b_grammar-only` | Qwen3-32B | bare grammar text | 9 (step-0 … 75, every 10) |
| `qwen3-32b_task-pairs` | Qwen3-32B | story → grammar pairs | 10 (step-0 … 900, every 100) |

All runs: LoRA r=16, alpha=16, dropout 0, on `q,k,v,o,gate,up,down_proj` in every
layer. lr 2e-4 cosine, 3% warmup, seed 0, 3 epochs. Every run includes step-0
(initialized, untrained), so each trajectory has its own origin.

The 8B and the 32B each have **both** recipes — same model, same LoRA config, only
the training data differs — and the outcomes are opposite: grammar-only training
perfects syntax and destroys accuracy (32B 34% → 3%), task pairs teach the task
(32B 34% → 99.7%).

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base, "llama-3.1-8b_task-pairs/step-500")
```

## Weight deltas

`dW = (alpha/r) * B @ A`, rank-16 by construction — the factors are the compact
form (168 MB factored vs ~2 GB dense per 8B checkpoint).

```python
from safetensors import safe_open

with safe_open("step-500/adapter_model.safetensors", framework="pt") as sf:
    A = sf.get_tensor("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
    B = sf.get_tensor("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight")
dW = B.float() @ A.float()
```

Each folder also has `trajectory.npz` — `steps`, `modules`, `norms` (‖dW‖ per
module per checkpoint) — enough to see where and when a model changed without
downloading any adapter.

## Notes

- `qwen3-32b_task-pairs` stops at step-900, not a full 3 epochs: its training
  container was lost to a network failure at ~2.6 epochs. Its accuracy curve is
  flat from step 500, so little was left to change.
- Adapters only, no base weights redistributed. The two Llama runs inherit the
  Llama 3.1 Community License; the Qwen3 and Ministral runs are Apache 2.0.
