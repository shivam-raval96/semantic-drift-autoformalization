#!/usr/bin/env python3
"""Grammar-only LoRA fine-tuning on Modal (A10G, same volume as evals).

Objective (locked): plain next-token prediction on the raw RG `text`
field only — every sample tokenized, EOS appended, all samples
concatenated and packed into fixed 1024-token blocks, loss on ALL
tokens. No chat template, no prompts, no masking.

Packing and EOS are done manually here (not via a trainer's packing
flag) so the objective is byte-auditable and independent of TRL/Unsloth
API drift. LoRA via PEFT; bf16 base by default (matches the eval
engine's dtype), with `load_in_4bit` as an explicit OOM fallback.

Modes (driver):
  smoke : 200 seeded samples, 50 steps, r=1 on ONE layer's o_proj
          (layer 16 of 32) — verifies trainable-param isolation,
          checkpointing, and the two sanity probes.
  full  : Phase 5a — whole train_v1, r=16, standard modules, all
          layers, 3 epochs, checkpoints every 10 steps + step 0.

Sanity probes (returned, and asserted by the driver for smoke):
  (a) holdout RG perplexity, FT vs base (same weights, adapter
      disabled) — FT must be clearly lower;
  (b) raw greedy completion of "ASSUME: <law>\nASK:" — must be fluent
      RG (the driver parses it with checkform's parser).

Checkpoints (PEFT adapter format, vLLM-loadable) go to the shared
volume under /models/checkpoints/<run_name>/step-N/.
"""

import modal

app = modal.App("harsh-ft-grammar-train")

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "torch", "transformers", "peft", "accelerate", "bitsandbytes",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
BLOCK = 1024
PROBE_PROMPT = "ASSUME: op(x, y) = op(op(y, y), x)\nASK:"


@app.function(
    gpu="A10G",
    image=train_image,
    volumes={"/models": weights},
    secrets=[hf_secret],
    timeout=3600,
    retries=0,
    max_containers=1,
    scaledown_window=60,
)
def train(texts: list, holdout_texts: list, config: dict) -> dict:
    import math
    import os
    import time

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break

    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(config["seed"])
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # ---- dataset: tokenize + EOS + concatenate + fixed blocks ----
    ids = []
    for text in texts:
        ids.extend(tok(text, add_special_tokens=False)["input_ids"])
        ids.append(tok.eos_token_id)
    blocks = [ids[i : i + BLOCK] for i in range(0, len(ids) - BLOCK + 1, BLOCK)]
    dataset = [{"input_ids": b, "labels": list(b)} for b in blocks]

    t0 = time.monotonic()
    load_kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    fallback_4bit = False
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    except torch.cuda.OutOfMemoryError:
        fallback_4bit = True
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="cuda",
        )
    weights.commit()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=config["rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
        layers_to_transform=config.get("layers_to_transform"),
    )
    model = get_peft_model(model, lora)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if config.get("layers_to_transform") is not None:
        allowed = tuple(f"layers.{k}." for k in config["layers_to_transform"])
        stray = [n for n in trainable if not any(a in n for a in allowed)]
        assert not stray, f"trainable params outside restricted layers: {stray[:5]}"

    run_dir = f"/models/checkpoints/{config['run_name']}"
    os.makedirs(run_dir, exist_ok=True)
    model.save_pretrained(f"{run_dir}/step-0")  # pre-training snapshot

    def collate(features):
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features]),
        }

    args = TrainingArguments(
        output_dir=f"{run_dir}/trainer",
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=config.get("epochs", 1),
        max_steps=config.get("max_steps", -1),
        logging_steps=1,
        save_strategy="no",  # we snapshot adapters ourselves
        seed=config["seed"],
        bf16=True,
        report_to=[],
        dataloader_drop_last=False,
    )

    from transformers import TrainerCallback

    class AdapterSnapshot(TrainerCallback):
        def on_step_end(self, args_, state, control, **kw):
            if state.global_step % config["save_steps"] == 0 or state.global_step == state.max_steps:
                model.save_pretrained(f"{run_dir}/step-{state.global_step}")
                weights.commit()

    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=collate, callbacks=[AdapterSnapshot()],
    )
    train_out = trainer.train()
    model.save_pretrained(f"{run_dir}/final")
    weights.commit()
    train_s = time.monotonic() - t0

    # ---- sanity probes ----
    model.eval()

    @torch.no_grad()
    def holdout_loss() -> float:
        total, count = 0.0, 0
        for text in holdout_texts:
            enc = tok(text + tok.eos_token, return_tensors="pt", add_special_tokens=False).to("cuda")
            out = model(input_ids=enc["input_ids"], labels=enc["input_ids"])
            n = enc["input_ids"].shape[1] - 1
            total += out.loss.item() * n
            count += n
        return total / count

    @torch.no_grad()
    def complete(prompt: str) -> str:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        gen = model.generate(
            **enc, max_new_tokens=48, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    ft_loss = holdout_loss()
    ft_completion = complete(PROBE_PROMPT)
    with model.disable_adapter():
        base_loss = holdout_loss()
        base_completion = complete(PROBE_PROMPT)

    return {
        "run_name": config["run_name"],
        "config": config,
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "peft": str(peft.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "quantized_fallback_4bit": fallback_4bit,
        },
        "dataset": {
            "samples": len(texts),
            "holdout_samples": len(holdout_texts),
            "total_tokens": len(ids),
            "blocks_1024": len(blocks),
        },
        "trainable_param_count": trainable_count,
        "trainable_param_names_sample": trainable[:6],
        "steps": trainer.state.global_step,
        "train_loss_final": train_out.metrics.get("train_loss"),
        "loss_curve": [
            {"step": h["step"], "loss": h["loss"]}
            for h in trainer.state.log_history if "loss" in h
        ],
        "probes": {
            "holdout_loss_base": round(base_loss, 4),
            "holdout_loss_ft": round(ft_loss, 4),
            "holdout_ppl_base": round(math.exp(base_loss), 2),
            "holdout_ppl_ft": round(math.exp(ft_loss), 2),
            "raw_completion_base": base_completion,
            "raw_completion_ft": ft_completion,
        },
        "checkpoints_dir": run_dir,
        "train_seconds": round(train_s, 1),
    }


@app.local_entrypoint()
def main(mode: str = "smoke"):
    import json
    import random
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from ftlib import REPO  # noqa: F401
    from checkform import parse_prefix_equation

    train_rows = [
        json.loads(l) for l in (here / "train_v1" / "train.jsonl").read_text().splitlines()
    ]
    holdout_rows = [
        json.loads(l) for l in (here / "train_v1" / "holdout.jsonl").read_text().splitlines()
    ]
    holdout_texts = [r["text"] for r in holdout_rows]

    if mode == "smoke":
        rng = random.Random(0)
        texts = [r["text"] for r in rng.sample(train_rows, 200)]
        config = {
            "run_name": "smoke-r1-L16-oproj",
            "seed": 0,
            "rank": 1,
            "lora_alpha": 1,
            "target_modules": ["o_proj"],
            "layers_to_transform": [16],
            "batch_size": 1,
            "lr": 2e-4,
            "max_steps": 50,
            "save_steps": 25,
        }
    elif mode == "full":
        texts = [r["text"] for r in train_rows]
        config = {
            "run_name": "phase5a-r16-all",
            "seed": 0,
            "rank": 16,
            "lora_alpha": 16,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "layers_to_transform": None,
            "batch_size": 4,
            "lr": 2e-4,
            "epochs": 3,
            "save_steps": 10,
        }
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    t0 = time.monotonic()
    result = train.remote(texts, holdout_texts, config)
    result["wall_seconds"] = round(time.monotonic() - t0, 1)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    out = here / "runs" / "ft-v1" / f"train-{config['run_name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    p = result["probes"]
    print(f"\n== {config['run_name']} · {result['steps']} steps · "
          f"{result['train_seconds']}s train · params {result['trainable_param_count']}")
    print(f"loss first->last: {result['loss_curve'][0]['loss']:.3f} -> "
          f"{result['loss_curve'][-1]['loss']:.3f}")
    print(f"holdout ppl base {p['holdout_ppl_base']} -> ft {p['holdout_ppl_ft']}")
    print(f"completion base: {p['raw_completion_base'][:80]!r}")
    print(f"completion ft:   {p['raw_completion_ft'][:80]!r}")

    # probe assertions (smoke gate)
    assert p["holdout_loss_ft"] < p["holdout_loss_base"], "probe (a) FAILED: no ppl gain"
    ask = p["raw_completion_ft"].strip().splitlines()[0].strip()
    parse_prefix_equation(ask.rstrip("."))  # raises if not fluent RG
    print("PROBES PASSED")
