#!/usr/bin/env python3
"""Task-pair LoRA fine-tuning on Modal (v2): story -> RG, completion-only loss.

Why v2 differs from v1 (training/train_lora.py): v1 taught the grammar
alone — raw RG text, packed 1024-token blocks, loss on every token, no
prompts. v2 teaches the TASK. Each sample is the frozen story-arm eval
prompt, built byte-identically to eval/modal_eval.py's path
(checkform.build_prompt on the sha-pinned formalize_prompt.md, then
benchmark.wrap_prompt(base, "off", model_id, "story"), then the model's
chat template with add_generation_prompt=True — no-think for Qwen3),
followed by the reference RG as the assistant completion.

Objective changes that follow from that:
  - Loss on completion tokens only (prompt labels -100): the story is
    context to condition on, not text to model.
  - EOS appended to the completion and trained, so generations stop.
  - NO packing: samples are whole (prompt, completion) episodes; block
    packing would splice unrelated stories into one context.
  - Overlong rows (> MAX_LEN tokens) are DROPPED and counted, never
    truncated — a clipped completion would train a malformed answer.

LoRA recipe and schedule are v1's exactly (r=16, alpha=16, all layers
q/k/v/o/gate/up/down, dropout 0, bias none, LR 2e-4 cosine 3% warmup,
bf16, grad checkpointing) so v1-vs-v2 differ only in data + objective.

Sanity probes (returned in the run record; asserted by the driver):
  (a) mean completion-token loss on train_v2/holdout.jsonl, adapter on
      vs off — FT must be lower;
  (b) greedy generation from one holdout prompt (full chat prompt, as
      trained) — the driver extracts ASSUME/ASK and parses them with
      checkform's parser.

Checkpoints (PEFT adapter format, vLLM-loadable) go to the shared
volume under /models/checkpoints/<run_name>/step-N/.

Usage:
  python3 ft-experiments/v2/train_pairs.py --dry-run          # local, no GPU
  modal run ft-experiments/v2/train_pairs.py --preset smoke-v2
  modal run ft-experiments/v2/train_pairs.py --preset v2-8b
  TRAIN_MODEL_ID="Qwen/Qwen3-32B" \
    modal run ft-experiments/v2/train_pairs.py --preset v2-32b
"""

import os

import modal

app = modal.App("harsh-ft-v2-train")

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

MODEL_ID = os.environ.get("TRAIN_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
TRAIN_GPU = os.environ.get("TRAIN_GPU", "A100-80GB")
MAX_LEN = 2048
RANK = 16  # v1 phase5a recipe, held fixed in v2 (alpha == rank)

# First line of the pinned story template; the dry run asserts the
# assembled prompt actually contains the template, not just the story.
TEMPLATE_HEADER = "You will read a short story and translate what it describes"

# Both models train on A100-80GB in v2. v2-32b: launch with
# TRAIN_MODEL_ID="Qwen/Qwen3-32B"; smaller micro-batch, same effective
# batch of 8. The bf16->nf4 fallback below covers the 32B if bf16 +
# activations don't fit (which path was taken is logged in the record).
PRESETS = {
    "smoke-v2": {"samples": 200, "max_steps": 50, "save_steps": 25,
                 "batch_size": 4, "grad_accum": 2},
    "v2-8b": {"samples": 0, "epochs": 3, "save_steps": 100,
              "batch_size": 4, "grad_accum": 2},
    "v2-32b": {"samples": 0, "epochs": 3, "save_steps": 100,
               "batch_size": 2, "grad_accum": 4},
    # Third model (screen 2026-08-19: story 11.1% = signal zone). Launch with
    # TRAIN_MODEL_ID="mistralai/Ministral-3-14B-Instruct-2512-BF16".
    "v2-14b": {"samples": 0, "epochs": 3, "save_steps": 100,
               "batch_size": 4, "grad_accum": 2},
}


# Shared by the container and the local dry run — pure python, no torch.

def chat_kwargs_for(model_id: str):
    # Qwen3 templates think by default; the frozen protocol is no-think.
    # Mirrors eval/modal_eval.py exactly.
    return {"enable_thinking": False} if "qwen3" in model_id.lower() else None


def _template_ids(tok, prompt_text: str, add_generation_prompt: bool, chat_kwargs) -> list:
    out = tok.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=add_generation_prompt, tokenize=True, **(chat_kwargs or {}),
    )
    # transformers <5 returns the id list; >=5 returns a BatchEncoding
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return ids[0] if ids and isinstance(ids[0], list) else ids


def chat_prompt_ids(tok, prompt_text: str, chat_kwargs) -> list:
    return _template_ids(tok, prompt_text, True, chat_kwargs)


def build_examples(tok, rows: list, chat_kwargs, max_len: int) -> tuple:
    """Tokenize (prompt, completion) rows -> masked-label examples.

    Returns (examples, n_dropped, lengths). A row that would exceed
    max_len is dropped whole — the completion is never truncated.
    """
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


def seq_stats(lengths: list) -> dict:
    return {
        "seq_len_max": max(lengths) if lengths else 0,
        "seq_len_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
    }


@app.function(
    gpu=TRAIN_GPU,
    image=train_image,
    volumes={"/models": weights},
    secrets=[hf_secret],
    timeout=21600,  # 32B at batch 2/accum 4 needs ~3.5h wall; 3h killed it once
    retries=0,
    max_containers=1,
    scaledown_window=60,
)
def train(rows: list, holdout_rows: list, config: dict) -> dict:
    import json
    import math
    import os
    import time

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    model_id = config["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    chat_kwargs = chat_kwargs_for(model_id)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    dataset, n_dropped, lengths = build_examples(tok, rows, chat_kwargs, config["max_len"])
    assert dataset, "every row dropped as overlong — check MAX_LEN vs the corpus"
    stats = seq_stats(lengths)
    print(f"= dataset: {len(dataset)} kept, {n_dropped} dropped overlong, "
          f"seq len max {stats['seq_len_max']} mean {stats['seq_len_mean']}")

    t0 = time.monotonic()
    fallback_4bit = False
    # Multimodal checkpoints (e.g. Ministral-3: Mistral3Config = vision tower
    # + language model) refuse AutoModelForCausalLM; load the composite class
    # and restrict LoRA to language-model layers below. Text-only training
    # works unchanged — the vision tower simply never sees an input.
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(model_id)
    multimodal = hasattr(hf_cfg, "vision_config")
    if multimodal:
        from transformers import AutoModelForImageTextToText as AutoLoader
    else:
        AutoLoader = AutoModelForCausalLM
    try:
        model = AutoLoader.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda")
    except torch.cuda.OutOfMemoryError:
        fallback_4bit = True
        from transformers import BitsAndBytesConfig
        model = AutoLoader.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map="cuda",
        )
    print(f"= model loaded: {'nf4 4-bit fallback' if fallback_4bit else 'bf16'}"
          f"{' (multimodal wrapper, text-only training)' if multimodal else ''}")
    weights.commit()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if multimodal:
        # Regex fullmatch keeps every adapter inside the language model; a bare
        # suffix list would also hit the vision tower's q/k/v/o projections.
        mods = "|".join(config["target_modules"])
        target_modules = rf".*language_model.*\.(?:{mods})"
    else:
        target_modules = config["target_modules"]
    lora = LoraConfig(
        r=config["rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert not any("vision" in n for n in trainable), \
        f"LoRA leaked into the vision tower: {[n for n in trainable if 'vision' in n][:3]}"

    run_dir = f"/models/checkpoints/{config['run_name']}"
    os.makedirs(run_dir, exist_ok=True)
    model.save_pretrained(f"{run_dir}/step-0")  # pre-training snapshot

    def collate(features):
        width = max(len(f["input_ids"]) for f in features)
        batch_ids, batch_labels, batch_mask = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            batch_ids.append(f["input_ids"] + [pad_id] * pad)
            batch_labels.append(f["labels"] + [-100] * pad)
            batch_mask.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(batch_ids),
            "labels": torch.tensor(batch_labels),
            "attention_mask": torch.tensor(batch_mask),
        }

    args = TrainingArguments(
        output_dir=f"{run_dir}/trainer",
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("grad_accum", 1),
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
    model.config.use_cache = True  # training is done; generation wants the cache

    holdout_ds, holdout_dropped, _ = build_examples(
        tok, holdout_rows, chat_kwargs, config["max_len"])

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
        gen = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=128, do_sample=False, pad_token_id=pad_id,
        )
        return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

    probe_prompt = holdout_rows[0]["prompt"]
    ft_loss = completion_loss()
    ft_generation = generate(probe_prompt)
    with model.disable_adapter():
        base_loss = completion_loss()
        base_generation = generate(probe_prompt)

    record = {
        "run_name": config["run_name"],
        "config": config,
        "objective": "completion-only",
        "template_sha": config["template_sha"],
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "peft": str(peft.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "quantized_fallback_4bit": fallback_4bit,
        },
        "dataset": {
            "pairs": len(rows),
            "kept": len(dataset),
            "n_dropped_overlong": n_dropped,
            "holdout_pairs": len(holdout_rows),
            "holdout_kept": len(holdout_ds),
            "holdout_dropped_overlong": holdout_dropped,
            **stats,
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
            "holdout_completion_loss_base": round(base_loss, 4),
            "holdout_completion_loss_ft": round(ft_loss, 4),
            "holdout_completion_ppl_base": round(math.exp(base_loss), 2),
            "holdout_completion_ppl_ft": round(math.exp(ft_loss), 2),
            "generation_reference": holdout_rows[0]["completion"],
            "generation_base": base_generation,
            "generation_ft": ft_generation,
        },
        "checkpoints_dir": run_dir,
        "train_seconds": round(train_s, 1),
    }
    # Also persist the record beside the checkpoints: a --detach run whose
    # local client has died would otherwise lose it with the return value.
    with open(f"{run_dir}/record.json", "w") as fh:
        json.dump(record, fh, indent=2)
    weights.commit()
    return record


# ---------------------------------------------------------- local side


def load_local():
    """Resolve configs + repo imports; assert the pinned template sha.

    Returns (cfg, repo) where cfg is training/config.py (LR, modules,
    EPOCHS, PATHS, MODELS via ftc) and repo is a dict of the repo
    functions the local side needs. Every prompt is built through
    checkform.build_prompt + benchmark.wrap_prompt — the exact calls
    eval/modal_eval.py makes — so training and eval byte-match.
    """
    import hashlib
    import importlib.util
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "ft_training_config", here.parent / "training" / "config.py")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    sys.path.insert(0, str(cfg.ftc.REPO))
    from benchmark import wrap_prompt
    from checkform import PROMPT_PATH, build_prompt, extract_answer, parse_prefix_equation

    want = cfg.ftc.EVAL["template_shas"]["formalize_prompt.md"]
    got = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    assert got == want, f"frozen template changed on disk: formalize_prompt.md {got}"

    return cfg, {
        "template_path": PROMPT_PATH,
        "template_sha": want,
        "build_prompt": build_prompt,
        "wrap_prompt": wrap_prompt,
        "extract_answer": extract_answer,
        "parse_prefix_equation": parse_prefix_equation,
    }


def load_rows(path) -> list:
    import json

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for row in rows:
        assert "story" in row and "completion" in row, f"bad train_v2 row in {path}"
    return rows


def build_payload(rows: list, repo: dict, model_id: str) -> list:
    """train_v2 rows -> the (prompt, completion) pairs the container trains on."""
    payload = []
    for row in rows:
        base = repo["build_prompt"]({"story": row["story"]},
                                    template_path=repo["template_path"])
        payload.append({
            "prompt": repo["wrap_prompt"](base, "off", model_id, "story"),
            "completion": row["completion"],
        })
    return payload


def model_tag(cfg, model_id: str) -> str:
    for key, entry in cfg.MODELS.items():
        if entry["hf_id"] == model_id:
            return key
    return model_id.rsplit("/", 1)[-1].lower()


def build_config(preset: str, seed: int, samples: int, epochs: int,
                 max_steps: int, run_name: str, cfg, template_sha: str) -> tuple:
    """Resolve a training config: preset defaults, explicit args override.

    Sentinels mark "unset" (seed -1, samples -1, epochs 0, max_steps 0,
    run_name ""). Rank/modules/LR are not knobs in v2 — the recipe is
    pinned to v1's phase5a config so only data + objective differ.
    """
    assert preset in PRESETS, f"give --preset from {sorted(PRESETS)}"
    base: dict = dict(PRESETS[preset])
    unset = {"seed": -1, "samples": -1, "epochs": 0, "max_steps": 0}
    for key, val in (("seed", seed), ("samples", samples),
                     ("epochs", epochs), ("max_steps", max_steps)):
        if val != unset[key]:
            base[key] = val
    base.setdefault("seed", 0)
    tag = model_tag(cfg, MODEL_ID)
    if preset == "v2-32b":  # batch sizing is for the 32B; catch a forgotten env
        assert "32b" in tag, f"preset v2-32b but TRAIN_MODEL_ID resolves to {tag!r}"
    stub = "smoke-v2" if preset == "smoke-v2" else "v2"
    config = {
        "run_name": run_name or f"{stub}-{tag}-s{base['seed']}",
        "model_id": MODEL_ID,
        "seed": base["seed"],
        "rank": RANK,
        "lora_alpha": RANK,  # alpha=r convention
        "target_modules": cfg.ALL_LAYER_MODULES,
        "batch_size": base["batch_size"],
        "grad_accum": base["grad_accum"],
        "lr": cfg.LR,
        "save_steps": base["save_steps"],
        "max_len": MAX_LEN,
        "template_sha": template_sha,
    }
    if base.get("max_steps"):
        config["max_steps"] = base["max_steps"]
    else:
        config["epochs"] = base.get("epochs", cfg.EPOCHS)
    return config, base.get("samples", 0)


def live_log(cfg, run_name: str, msg: str):
    from datetime import datetime

    path = cfg.PATHS["runs"] / "ft-v2" / "live.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(f"{datetime.now().strftime('%H:%M:%S')} [train-{run_name}] {msg}\n")


# ------------------------------------------------------------- dry run

# Synthetic stand-ins for train_v2 rows (the real file is generated by
# the data pipeline); completions are valid RG so the probe-side parse
# path can be exercised end to end.
_DRYRUN_ROWS = [
    ("In a certain paint workshop, the colorist pours crimson into ochre "
     "and compares it with pouring ochre into ochre, then that into crimson. "
     "The two batches always match. Must pouring in the other order match too?",
     "ASSUME: op(x, y) = op(op(y, y), x)\nASK: op(x, y) = op(y, x)"),
    ("In a tea house, the brewer steeps jasmine into oolong and it always "
     "equals oolong steeped into jasmine. Must a double steep of jasmine "
     "equal jasmine alone?",
     "ASSUME: op(x, y) = op(y, x)\nASK: op(x, x) = x"),
    ("A gardener grafts rose onto fern, then the result onto rose again; "
     "it always matches plain rose. Must grafting fern onto rose match fern?",
     "ASSUME: op(op(x, y), x) = x\nASK: op(y, x) = y"),
    ("A smith alloys iron with copper, always matching copper alloyed with "
     "the same mix. Must the triple alloy match the single?",
     "ASSUME: op(x, op(y, op(x, y))) = x\nASK: op(x, x) = op(x, op(x, x))"),
    ("In a dye works, folding indigo into madder twice always gives madder. "
     "Must one fold give indigo?",
     "ASSUME: op(op(x, x), y) = y\nASK: op(x, y) = x"),
]


def run_dry():
    """Local, GPU-free verification of the full data path.

    Synthesizes train_v2-shaped rows in a temp dir, loads them through
    the real loader, builds prompts through the real repo functions, and
    tokenizes with the real tokenizer — then asserts the masking, EOS,
    template, and no-truncation properties the objective depends on.
    """
    import json
    import tempfile
    from pathlib import Path

    cfg, repo = load_local()
    print(f"= template sha OK: {repo['template_sha'][:16]}…")

    tmp = Path(tempfile.mkdtemp(prefix="train-v2-dryrun-"))
    with (tmp / "train.jsonl").open("w") as fh:
        for i, (story, completion) in enumerate(_DRYRUN_ROWS):
            fh.write(json.dumps({
                "story": story, "completion": completion,
                "pair_hash": f"dryrun-{i}", "tier": "dryrun", "source": "dryrun",
            }) + "\n")
    rows = load_rows(tmp / "train.jsonl")
    assert len(rows) == len(_DRYRUN_ROWS)
    payload = build_payload(rows, repo, MODEL_ID)

    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as err:  # gated repo without local auth
        raise SystemExit(
            f"could not load the {MODEL_ID} tokenizer locally ({err}); "
            "authenticate with `hf auth login` (or set HF_TOKEN) and rerun"
        )
    chat_kwargs = chat_kwargs_for(MODEL_ID)
    eos = tok.eos_token_id

    examples, n_dropped, lengths = build_examples(tok, payload, chat_kwargs, MAX_LEN)
    assert n_dropped == 0 and len(examples) == len(payload), "unexpected drops"

    for pair, ex in zip(payload, examples):
        n_prompt = ex["n_prompt"]
        ids, labels = ex["input_ids"], ex["labels"]
        completion_ids = tok(pair["completion"], add_special_tokens=False)["input_ids"] + [eos]

        # masking exactly covers the prompt; no row truncates the completion
        assert all(l == -100 for l in labels[:n_prompt])
        first_target = next(i for i, l in enumerate(labels) if l != -100)
        assert first_target == n_prompt
        assert labels[n_prompt] == completion_ids[0]
        assert ids[n_prompt:] == completion_ids and labels[n_prompt:] == completion_ids
        assert ids[-1] == eos and labels[-1] == eos  # EOS present and trained

        # prompt really is the template + story + regime suffix, chat-formatted
        decoded_prompt = tok.decode(ids[:n_prompt])
        assert TEMPLATE_HEADER in decoded_prompt
        assert "Respond with only the two required lines" in decoded_prompt

        # and it ends with the tokenizer's generation prompt (what eval
        # generates after) — derived, not hardcoded per model family
        no_gen = _template_ids(tok, pair["prompt"], False, chat_kwargs)
        assert ids[: len(no_gen)] == no_gen and n_prompt > len(no_gen)
        gen_tail = tok.decode(ids[len(no_gen):n_prompt])
        assert decoded_prompt.endswith(gen_tail)

    # the overlong branch drops whole rows rather than truncating
    forced, forced_dropped, _ = build_examples(tok, payload[:1], chat_kwargs, max_len=64)
    assert forced == [] and forced_dropped == 1

    # completions parse under the grader's own parser (probe-b path)
    for pair in payload:
        assume, ask = repo["extract_answer"](pair["completion"])
        repo["parse_prefix_equation"](assume)
        repo["parse_prefix_equation"](ask)

    stats = seq_stats(lengths)
    ex0, pair0 = examples[0], payload[0]
    print(f"= {len(examples)} rows built, 0 dropped, "
          f"seq len max {stats['seq_len_max']} mean {stats['seq_len_mean']}, "
          f"prompt tokens {ex0['n_prompt']}, "
          f"completion tokens {len(ex0['input_ids']) - ex0['n_prompt']}")
    print("\n========== assembled example (row 0) ==========")
    print("---------- prompt (labels = -100) ----------")
    print(tok.decode(ex0["input_ids"][: ex0["n_prompt"]]))
    print("---------- completion (loss on these tokens) ----------")
    print(tok.decode(ex0["input_ids"][ex0["n_prompt"]:]))
    print("========== end example ==========\n")
    print("DRY RUN OK: template sha pinned; masking covers exactly the prompt; "
          "completion + EOS trained; chat prompt ends with the generation prompt; "
          "overlong rows drop whole (never truncate); completions parse.")


# ------------------------------------------------------------ entrypoint


@app.local_entrypoint()
def main(
    preset: str = "",
    seed: int = -1,       # -1 = unset (preset/default 0)
    samples: int = -1,    # -1 = unset; 0 = full corpus
    epochs: int = 0,      # 0 = unset
    max_steps: int = 0,   # 0 = unset (use epochs)
    run_name: str = "",   # "" = derived v2-{model}-s{seed}
    dry_run: bool = False,
):
    import json
    import random
    import time
    from datetime import datetime, timezone

    if dry_run:
        run_dry()
        return

    cfg, repo = load_local()
    config, n_samples = build_config(
        preset, seed, samples, epochs, max_steps, run_name, cfg, repo["template_sha"])
    print(f"= resolved training config: {json.dumps(config)}  samples={n_samples or 'all'}")

    train_v2 = cfg.ftc.ROOT / "train_v2"
    train_rows = load_rows(train_v2 / "train.jsonl")
    holdout_rows = load_rows(train_v2 / "holdout.jsonl")
    if n_samples:
        rng = random.Random(config["seed"])
        train_rows = rng.sample(train_rows, n_samples)
    payload = build_payload(train_rows, repo, config["model_id"])
    holdout_payload = build_payload(holdout_rows, repo, config["model_id"])

    live_log(cfg, config["run_name"],
             f"started: {config['model_id']} on {TRAIN_GPU}, {len(payload)} pairs, "
             f"preset={preset}, completion-only loss")
    t0 = time.monotonic()
    result = train.remote(payload, holdout_payload, config)
    result["wall_seconds"] = round(time.monotonic() - t0, 1)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    out = cfg.PATHS["runs"] / "ft-v2" / f"train-{config['run_name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    d, p = result["dataset"], result["probes"]
    print(f"\n== {config['run_name']} · {result['steps']} steps · "
          f"{result['train_seconds']}s train · params {result['trainable_param_count']} · "
          f"{'nf4 fallback' if result['versions']['quantized_fallback_4bit'] else 'bf16'}")
    print(f"dataset: {d['kept']} kept / {d['n_dropped_overlong']} dropped overlong · "
          f"seq len max {d['seq_len_max']} mean {d['seq_len_mean']}")
    print(f"loss first->last: {result['loss_curve'][0]['loss']:.3f} -> "
          f"{result['loss_curve'][-1]['loss']:.3f}")
    print(f"holdout completion loss base {p['holdout_completion_loss_base']} -> "
          f"ft {p['holdout_completion_loss_ft']}")
    print(f"generation ft:   {p['generation_ft'][:100]!r}")
    print(f"generation base: {p['generation_base'][:100]!r}")

    # probe assertions (the record above is already on disk either way)
    try:
        assert p["holdout_completion_loss_ft"] < p["holdout_completion_loss_base"], \
            "probe (a) FAILED: no completion-loss gain over base"
        assume, ask = repo["extract_answer"](p["generation_ft"])
        repo["parse_prefix_equation"](assume)  # raises if not fluent RG
        repo["parse_prefix_equation"](ask)
    except Exception as err:
        live_log(cfg, config["run_name"], f"done ({result['steps']} steps) but PROBE FAILED: {err}")
        raise
    live_log(cfg, config["run_name"],
             f"done: {result['steps']} steps in {result['train_seconds']}s, "
             f"loss {result['loss_curve'][0]['loss']:.3f}->{result['loss_curve'][-1]['loss']:.3f}, "
             f"holdout completion loss {p['holdout_completion_loss_base']}->"
             f"{p['holdout_completion_loss_ft']}, "
             f"{d['n_dropped_overlong']} dropped overlong, probes passed; record {out.name}")
    print("PROBES PASSED")


if __name__ == "__main__":
    import sys

    if "--dry-run" in sys.argv:
        run_dry()
    else:
        raise SystemExit("use `modal run ft-experiments/v2/train_pairs.py --preset ...` "
                         "to train, or `--dry-run` for the local check")
