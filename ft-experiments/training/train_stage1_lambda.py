#!/usr/bin/env python3
"""Stage-1 fine-tune, Lambda-native (runs in-process on the box's GPU).

Same objective and LoRA recipe as train_stage1.py / train_pairs.py —
chat prompt masked, loss on the completion tokens only, EOS trained, no
packing, r=16 all-layer LoRA — but with no Modal: the training loop runs
here, and because a Lambda instance is ephemeral, every checkpoint is
saved locally AND pushed to a HuggingFace repo so nothing is lost on
teardown.

    # one-time on the box:
    #   pip install -r training/requirements-lambda.txt
    #   hf auth login                      (or export HF_TOKEN=...)
    #   export HF_TOKEN=...                (gated Llama base + your repo)
    python3 training/train_stage1_lambda.py --hf-repo <you>/mars-v-stage1
    python3 training/train_stage1_lambda.py --no-push            # local only
    python3 training/train_stage1_lambda.py --dry-run            # no GPU

Checkpoints land at <output-dir>/<run>/step-N/ locally and at
<run>/step-N/ inside the HF repo (private by default), mirroring the
existing CHECKPOINTS.md layout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as tcfg          # training/config.py — no modal import

PATHS = tcfg.PATHS
STAGE1 = PATHS["stage1"]
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RANK = 16
MAX_LEN = 2048
PROBE_HOLDOUT = 64


# ----- masking helpers (byte-for-byte the train_pairs.py objective) -----

def chat_kwargs_for(model_id: str):
    return {"enable_thinking": False} if "qwen3" in model_id.lower() else None


def _template_ids(tok, prompt_text, add_generation_prompt, chat_kwargs):
    out = tok.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=add_generation_prompt, tokenize=True, **(chat_kwargs or {}),
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return ids[0] if ids and isinstance(ids[0], list) else ids


def chat_prompt_ids(tok, prompt_text, chat_kwargs):
    return _template_ids(tok, prompt_text, True, chat_kwargs)


def build_examples(tok, rows, chat_kwargs, max_len):
    """Tokenize (prompt, completion) rows -> masked-label examples; an
    overlong row is dropped whole (the completion is never truncated)."""
    eos = tok.eos_token_id
    examples, n_dropped, lengths = [], 0, []
    for row in rows:
        prompt_ids = chat_prompt_ids(tok, row["prompt"], chat_kwargs)
        completion_ids = tok(row["completion"], add_special_tokens=False)["input_ids"]
        completion_ids.append(eos)
        if len(prompt_ids) + len(completion_ids) > max_len:
            n_dropped += 1
            continue
        examples.append({
            "input_ids": prompt_ids + completion_ids,
            "labels": [-100] * len(prompt_ids) + completion_ids,
            "n_prompt": len(prompt_ids),
        })
        lengths.append(len(prompt_ids) + len(completion_ids))
    return examples, n_dropped, lengths


# ----------------------------------------------------------- local side


def load_rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def payload(rows: list) -> list:
    return [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]


def model_tag(model_id: str) -> str:
    for key, entry in tcfg.MODELS.items():
        if entry["hf_id"] == model_id:
            return key
    return model_id.rsplit("/", 1)[-1].lower()


def build_config(model_id: str, seed: int, epochs: int, save_steps: int) -> dict:
    manifest = json.loads((STAGE1 / "manifest.json").read_text())
    return {
        "run_name": f"stage1-{model_tag(model_id)}-s{seed}",
        "model_id": model_id,
        "seed": seed,
        "rank": RANK,
        "lora_alpha": RANK,
        "target_modules": tcfg.ALL_LAYER_MODULES,
        "batch_size": 4,
        "grad_accum": 2,
        "lr": tcfg.LR,
        "save_steps": save_steps,
        "max_len": MAX_LEN,
        "epochs": epochs,
        "template_sha": manifest["prompt_sha16"],
        "platform": "lambda",
    }


# ------------------------------------------------------- the training run


def run_training(train_payload, holdout_payload, config, out_dir: Path,
                 hf_repo: str, push: bool) -> dict:
    import math
    import os

    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(config["seed"])
    model_id = config["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    chat_kwargs = chat_kwargs_for(model_id)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    dataset, n_dropped, lengths = build_examples(tok, train_payload, chat_kwargs, config["max_len"])
    assert dataset, "every row dropped as overlong — check MAX_LEN vs the corpus"
    print(f"= dataset: {len(dataset)} kept, {n_dropped} dropped overlong, "
          f"max seq {max(lengths)}")

    api = None
    if push:
        from huggingface_hub import HfApi

        api = HfApi()
        if not hf_repo:
            hf_repo = f"{api.whoami()['name']}/mars-v-stage1"
        api.create_repo(hf_repo, repo_type="model", private=True, exist_ok=True)
        print(f"= pushing checkpoints to https://huggingface.co/{hf_repo} (private)")

    run_dir = out_dir / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    def push_dir(local: Path, path_in_repo: str):
        if api is not None:
            api.upload_folder(folder_path=str(local), path_in_repo=path_in_repo,
                              repo_id=hf_repo, ignore_patterns=[".*"])

    fallback_4bit = False
    hf_cfg = AutoConfig.from_pretrained(model_id)
    multimodal = hasattr(hf_cfg, "vision_config")
    if multimodal:
        from transformers import AutoModelForImageTextToText as Loader
    else:
        Loader = AutoModelForCausalLM
    try:
        model = Loader.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    except torch.cuda.OutOfMemoryError:
        fallback_4bit = True
        from transformers import BitsAndBytesConfig

        model = Loader.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16),
            device_map="cuda",
        )
    print(f"= model loaded: {'nf4 4-bit fallback' if fallback_4bit else 'bf16'}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if multimodal:
        mods = "|".join(config["target_modules"])
        target_modules = rf".*language_model.*\.(?:{mods})"
    else:
        target_modules = config["target_modules"]
    lora = LoraConfig(
        r=config["rank"], lora_alpha=config["lora_alpha"], lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
    model = get_peft_model(model, lora)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert not any("vision" in n for n in trainable), "LoRA leaked into the vision tower"

    model.save_pretrained(f"{run_dir}/step-0")
    push_dir(run_dir / "step-0", f"{config['run_name']}/step-0")

    def collate(features):
        width = max(len(f["input_ids"]) for f in features)
        ids, labels, mask = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            ids.append(f["input_ids"] + [pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            mask.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    args = TrainingArguments(
        output_dir=f"{run_dir}/trainer",
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03,
        num_train_epochs=config["epochs"], logging_steps=1, save_strategy="no",
        seed=config["seed"], bf16=True, report_to=[], dataloader_drop_last=False,
    )

    class AdapterSnapshot(TrainerCallback):
        def on_step_end(self, args_, state, control, **kw):
            if state.global_step % config["save_steps"] == 0 or state.global_step == state.max_steps:
                step_dir = run_dir / f"step-{state.global_step}"
                model.save_pretrained(str(step_dir))
                push_dir(step_dir, f"{config['run_name']}/step-{state.global_step}")

    t0 = time.monotonic()
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
                      data_collator=collate, callbacks=[AdapterSnapshot()])
    train_out = trainer.train()
    model.save_pretrained(f"{run_dir}/final")
    push_dir(run_dir / "final", f"{config['run_name']}/final")
    train_s = time.monotonic() - t0

    # ---- probes ----
    model.eval()
    model.config.use_cache = True
    holdout_ds, _, _ = build_examples(tok, holdout_payload, chat_kwargs, config["max_len"])

    @torch.no_grad()
    def completion_loss() -> float:
        total, count = 0.0, 0
        for ex in holdout_ds:
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            labels = torch.tensor([ex["labels"]], device="cuda")
            out = model(input_ids=ids, labels=labels)
            n = int((labels != -100).sum())
            total += out.loss.item() * n
            count += n
        return total / count

    @torch.no_grad()
    def generate(prompt_text: str) -> str:
        ids = torch.tensor([chat_prompt_ids(tok, prompt_text, chat_kwargs)], device="cuda")
        gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=32, do_sample=False, pad_token_id=pad_id)
        return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

    probe_prompt = holdout_payload[0]["prompt"]
    ft_loss = completion_loss()
    ft_generation = generate(probe_prompt)
    with model.disable_adapter():
        base_loss = completion_loss()
        base_generation = generate(probe_prompt)

    record = {
        "run_name": config["run_name"], "config": config, "objective": "completion-only",
        "platform": "lambda",
        "hf_repo": hf_repo if push else None,
        "versions": {
            "torch": str(torch.__version__), "transformers": str(transformers.__version__),
            "peft": str(peft.__version__), "gpu": torch.cuda.get_device_name(0),
            "quantized_fallback_4bit": fallback_4bit,
        },
        "dataset": {"pairs": len(train_payload), "kept": len(dataset),
                    "n_dropped_overlong": n_dropped, "holdout_kept": len(holdout_ds)},
        "steps": trainer.state.global_step,
        "train_loss_final": train_out.metrics.get("train_loss"),
        "loss_curve": [{"step": h["step"], "loss": h["loss"]}
                       for h in trainer.state.log_history if "loss" in h],
        "probes": {
            "holdout_completion_loss_base": round(base_loss, 4),
            "holdout_completion_loss_ft": round(ft_loss, 4),
            "holdout_completion_ppl_base": round(math.exp(base_loss), 2),
            "holdout_completion_ppl_ft": round(math.exp(ft_loss), 2),
            "generation_reference": holdout_payload[0]["completion"],
            "generation_base": base_generation,
            "generation_ft": ft_generation,
        },
        "checkpoints_dir": str(run_dir),
        "train_seconds": round(train_s, 1),
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    if api is not None:
        api.upload_file(path_or_fileobj=(run_dir / "record.json").read_bytes(),
                        path_in_repo=f"{config['run_name']}/record.json", repo_id=hf_repo)
    return record


def run_dry(config: dict, train_payload: list, holdout_payload: list) -> None:
    for p in (train_payload + holdout_payload):
        assert p["prompt"] and p["completion"], "empty prompt/completion"
        assert "\n\n" in p["prompt"], "prompt missing statement block"
    print(f"= {len(train_payload)} train + {len(holdout_payload)} holdout rows assembled")
    print(f"= config: {json.dumps(config)}")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config["model_id"])
    except Exception as err:  # gated base or offline box
        print(f"= tokenizer unavailable ({type(err).__name__}); skipping masking check")
        print("DRY RUN OK (structural)")
        return
    chat_kwargs = chat_kwargs_for(config["model_id"])
    examples, n_dropped, _ = build_examples(tok, train_payload[:8], chat_kwargs, MAX_LEN)
    for pair, ex in zip(train_payload[:8], examples):
        n_prompt = ex["n_prompt"]
        completion_ids = tok(pair["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        assert all(l == -100 for l in ex["labels"][:n_prompt]), "prompt not fully masked"
        assert ex["input_ids"][n_prompt:] == completion_ids, "completion tokens mismatch"
        assert ex["labels"][n_prompt:] == completion_ids, "loss not on completion+EOS"
    assert n_dropped == 0, "unexpected overlong drops in dry run"
    print(f"= masking verified on {len(examples)} rows: prompt masked, completion+EOS trained")
    print("DRY RUN OK")


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Lambda-native stage-1 fine-tune.")
    cli.add_argument("--model-id", default=DEFAULT_MODEL)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--epochs", type=int, default=tcfg.EPOCHS)
    cli.add_argument("--save-steps", type=int, default=100)
    cli.add_argument("--hf-repo", default="", help="target repo; default <you>/mars-v-stage1")
    cli.add_argument("--no-push", action="store_true", help="save locally only, no HF push")
    cli.add_argument("--output-dir", type=Path, default=HERE.parent / "checkpoints")
    cli.add_argument("--dry-run", action="store_true", help="assemble + check masking, no GPU")
    args = cli.parse_args(argv)

    train_rows = load_rows(STAGE1 / "train.jsonl")
    eval_rows = load_rows(STAGE1 / "eval.jsonl")
    train_payload = payload(train_rows)
    import random
    holdout_payload = payload(random.Random(args.seed).sample(eval_rows, PROBE_HOLDOUT))
    config = build_config(args.model_id, args.seed, args.epochs, args.save_steps)

    print(f"= stage-1 (lambda) {config['run_name']}: {len(train_payload)} rows, "
          f"model {config['model_id']}, r{config['rank']}, {config['epochs']} epochs")

    if args.dry_run:
        run_dry(config, train_payload, holdout_payload)
        return 0

    record = run_training(train_payload, holdout_payload, config,
                          args.output_dir, args.hf_repo, push=not args.no_push)

    runs_dir = PATHS["runs"] / "stage1"
    runs_dir.mkdir(parents=True, exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    (runs_dir / f"train-{config['run_name']}-lambda.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    p = record["probes"]
    print(f"\n== {config['run_name']} · {record['steps']} steps · {record['train_seconds']}s")
    print(f"loss first->last: {record['loss_curve'][0]['loss']:.3f} -> "
          f"{record['loss_curve'][-1]['loss']:.3f}")
    print(f"holdout completion loss base {p['holdout_completion_loss_base']} -> "
          f"ft {p['holdout_completion_loss_ft']}")
    print(f"generation ft: {p['generation_ft'][:80]!r}")
    assert p["holdout_completion_loss_ft"] < p["holdout_completion_loss_base"], \
        "probe FAILED: FT did not reduce held-out completion loss over base"
    print("PROBE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
