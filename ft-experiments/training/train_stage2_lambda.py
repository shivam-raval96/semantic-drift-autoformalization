#!/usr/bin/env python3
"""Stage-2 translation fine-tune, Lambda-native (in-process on the GPU).

Same completion-only objective and r=16 all-layer LoRA recipe as
train_pairs.py / train_stage1_lambda.py, but the prompt is the frozen
translation template (story -> rigid grammar), built through the repo's
build_prompt + wrap_prompt so training and eval see byte-identical inputs.

Two things stage 2 adds over stage 1:
  --base            plain Llama (direct arm, D0) or the F0-merged dir
                    (staged arm, T1/T3) — see training/merge_adapter.py.
  --resume-adapter  continue an existing adapter instead of a fresh LoRA,
                    for the T1 -> T2 A-then-B curriculum (one adapter).

    export HF_TOKEN=...
    # T1: staged, RG-1, tea held out
    python3 training/train_stage2_lambda.py --run-name t1-8b \
        --base checkpoints/llama-3.1-8b_f0-merged \
        --grammar rg1 --data train_v2/train.jsonl --hf-repo <you>/mars-v-stage2
    # T2: continue T1 on RG-2
    python3 training/train_stage2_lambda.py --run-name t2-8b \
        --base checkpoints/llama-3.1-8b_f0-merged \
        --grammar rg2 --data stage2/train_v2_rg2.jsonl \
        --resume-adapter checkpoints/t1-8b/final --hf-repo <you>/mars-v-stage2
    # T3: staged, RG-1, single theme upsampled
    python3 training/train_stage2_lambda.py --run-name t3-8b \
        --base checkpoints/llama-3.1-8b_f0-merged \
        --grammar rg1 --data stage2/train_v2_rg1_singletheme.jsonl --hf-repo <you>/mars-v-stage2
    # D0: direct, plain base, RG-1
    python3 training/train_stage2_lambda.py --run-name d0-8b \
        --base meta-llama/Llama-3.1-8B-Instruct \
        --grammar rg1 --data train_v2/train.jsonl --hf-repo <you>/mars-v-stage2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import importlib.util

HERE = Path(__file__).resolve().parent
DATA_GEN = HERE.parent / "data-gen"
if str(DATA_GEN) not in sys.path:
    sys.path.insert(0, str(DATA_GEN))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tcfg = _load("ft_training_config", HERE / "config.py")  # training/config.py
import stage2lib as s2         # data-gen/stage2lib.py — shared prompt/grade

PATHS = tcfg.PATHS
ROOT = tcfg.ftc.ROOT           # ft-experiments/
DEFAULT_BASE = "meta-llama/Llama-3.1-8B-Instruct"
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

def resolve_path(value: str) -> str:
    """A local dir under ft-experiments resolves to an absolute path; an HF
    id (has no such dir) is passed through untouched."""
    candidate = (ROOT / value) if not Path(value).is_absolute() else Path(value)
    return str(candidate) if candidate.exists() else value


def load_rows(path: Path) -> list:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        assert "story" in r and "completion" in r, f"bad translation row in {path}"
    return rows


def build_payload(rows: list, key: str, model_id: str) -> list:
    return [{"prompt": s2.build_translation_prompt(r["story"], key, model_id),
             "completion": r["completion"],
             "canonical_e": r["canonical_e"], "canonical_f": r["canonical_f"]}
            for r in rows]


def build_config(args, key: str) -> dict:
    return {
        "run_name": args.run_name,
        "base": args.base,
        "grammar": key,
        "label": s2.GRAMMAR_TO_LABEL[key],
        "data": args.data,
        "resume_adapter": args.resume_adapter or None,
        "seed": args.seed,
        "rank": RANK,
        "lora_alpha": RANK,
        "target_modules": tcfg.ALL_LAYER_MODULES,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": tcfg.LR,
        "save_steps": args.save_steps,
        "max_len": MAX_LEN,
        "epochs": args.epochs,
        "template_sha": s2.template_sha(key),
        "platform": "lambda",
    }


# ------------------------------------------------------- the training run

def run_training(train_payload, holdout_payload, config, out_dir: Path,
                 base_dir: str, resume_adapter, hf_repo: str, push: bool) -> dict:
    import os

    import torch
    import transformers
    import peft
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.manual_seed(config["seed"])
    model_id = config["model_id"]
    tok = AutoTokenizer.from_pretrained(base_dir)
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
            hf_repo = f"{api.whoami()['name']}/mars-v-stage2"
        api.create_repo(hf_repo, repo_type="model", private=True, exist_ok=True)
        print(f"= pushing checkpoints to https://huggingface.co/{hf_repo} (private)")

    run_dir = out_dir / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    def push_dir(local: Path, path_in_repo: str):
        if api is not None:
            # Skip PEFT's model-card README: its base_model YAML is the local
            # merged-base path, which HF's upload validator rejects (it wants a
            # hub id). The adapter is always loaded with an explicit --base, so
            # the card is cosmetic; adapter_config.json is not YAML-validated.
            api.upload_folder(folder_path=str(local), path_in_repo=path_in_repo,
                              repo_id=hf_repo, ignore_patterns=[".*", "README.md"])

    fallback_4bit = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype=torch.bfloat16, device_map="cuda")
    except torch.cuda.OutOfMemoryError:
        fallback_4bit = True
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            base_dir,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16),
            device_map="cuda",
        )
    print(f"= base loaded: {'nf4 4-bit fallback' if fallback_4bit else 'bf16'} from {base_dir}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if resume_adapter:
        adapter_dir = resolve_path(resume_adapter)
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        print(f"= resumed adapter (continuing) from {adapter_dir}")
    else:
        lora = LoraConfig(
            r=config["rank"], lora_alpha=config["lora_alpha"], lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=config["target_modules"])
        model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"= trainable params: {trainable}")

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

    # explicit int warmup keeps this version-robust (some transformers builds
    # reject warmup_ratio); 3% of the total optimizer steps as elsewhere.
    steps_per_epoch = math.ceil(len(dataset) / (config["batch_size"] * config["grad_accum"]))
    total_steps = steps_per_epoch * config["epochs"]
    warmup_steps = max(1, round(0.03 * total_steps))

    args = TrainingArguments(
        output_dir=f"{run_dir}/trainer",
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"], lr_scheduler_type="cosine", warmup_steps=warmup_steps,
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
                             max_new_tokens=128, do_sample=False, pad_token_id=pad_id)
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
        "warmup_steps": warmup_steps,
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
            "probe_note": "in-distribution sanity probe (sampled from the training data)",
        },
        "checkpoints_dir": str(run_dir),
        "train_seconds": round(train_s, 1),
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    if api is not None:
        api.upload_file(path_or_fileobj=(run_dir / "record.json").read_bytes(),
                        path_in_repo=f"{config['run_name']}/record.json", repo_id=hf_repo)
    return record


def run_dry(config, base_dir, train_payload, holdout_payload) -> None:
    for p in (train_payload + holdout_payload):
        assert p["prompt"] and p["completion"], "empty prompt/completion"
        assert "translate" in p["prompt"].lower(), "prompt missing translation template"
    print(f"= {len(train_payload)} train + {len(holdout_payload)} holdout rows assembled")
    print(f"= grammar {config['label']} ({config['grammar']}), template sha16 "
          f"{config['template_sha'][:16]}, base {base_dir}")
    # every completion must grade exact under its target grammar
    for p in train_payload[:64]:
        v = s2.grade(p["completion"], config["grammar"], p["canonical_e"], p["canonical_f"])
        assert v["status"] == "correct", ("completion not exact", v)
    print(f"= completions grade exact under {config['label']}")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(base_dir)
    except Exception as err:
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
    print(f"= masking verified on {len(examples)} rows: prompt masked, completion+EOS trained")
    print("DRY RUN OK")


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Lambda-native stage-2 translation fine-tune.")
    cli.add_argument("--run-name", required=True, help="e.g. t1-8b / t2-8b / t3-8b / d0-8b")
    cli.add_argument("--base", default=DEFAULT_BASE, help="HF id or local merged-F0 dir")
    cli.add_argument("--grammar", default="rg1", help="rg1 / rg2 / rg3 / rg4 (target grammar)")
    cli.add_argument("--data", required=True, help="translation jsonl (story + completion)")
    cli.add_argument("--resume-adapter", default="", help="continue this adapter (T1 -> T2)")
    cli.add_argument("--model-id", default=DEFAULT_BASE, help="tokenizer/chat family of the base")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--epochs", type=int, default=tcfg.EPOCHS)
    cli.add_argument("--save-steps", type=int, default=100)
    cli.add_argument("--batch-size", type=int, default=4)
    cli.add_argument("--grad-accum", type=int, default=2)
    cli.add_argument("--hf-repo", default="", help="target repo; default <you>/mars-v-stage2")
    cli.add_argument("--no-push", action="store_true", help="save locally only")
    cli.add_argument("--output-dir", type=Path, default=HERE.parent / "checkpoints")
    cli.add_argument("--dry-run", action="store_true", help="assemble + check masking, no GPU")
    args = cli.parse_args(argv)

    s2.assert_frozen_templates()
    key = s2.key_for(args.grammar)
    base_dir = resolve_path(args.base)

    data_path = Path(resolve_path(args.data))
    rows = load_rows(data_path)
    config = build_config(args, key)
    config["model_id"] = args.model_id

    train_payload = build_payload(rows, key, config["model_id"])
    holdout_rows = random.Random(args.seed).sample(rows, min(PROBE_HOLDOUT, len(rows)))
    holdout_payload = build_payload(holdout_rows, key, config["model_id"])

    print(f"= stage-2 (lambda) {config['run_name']}: {len(train_payload)} rows -> {config['label']}, "
          f"base {base_dir}{' +resume' if args.resume_adapter else ''}, "
          f"r{config['rank']}, {config['epochs']} epochs")

    if args.dry_run:
        run_dry(config, base_dir, train_payload, holdout_payload)
        return 0

    record = run_training(train_payload, holdout_payload, config, args.output_dir,
                          base_dir, args.resume_adapter, args.hf_repo, push=not args.no_push)

    runs_dir = PATHS["runs"] / "stage2"
    runs_dir.mkdir(parents=True, exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    (runs_dir / f"train-{config['run_name']}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    p = record["probes"]
    print(f"\n== {config['run_name']} · {record['steps']} steps · {record['train_seconds']}s")
    print(f"loss first->last: {record['loss_curve'][0]['loss']:.3f} -> "
          f"{record['loss_curve'][-1]['loss']:.3f}")
    print(f"holdout completion loss base {p['holdout_completion_loss_base']} -> "
          f"ft {p['holdout_completion_loss_ft']}")
    print(f"generation ft: {p['generation_ft'][:100]!r}")
    assert p["holdout_completion_loss_ft"] < p["holdout_completion_loss_base"], \
        "probe FAILED: FT did not reduce held-out completion loss over base"
    print("PROBE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
