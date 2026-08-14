#!/usr/bin/env python3
"""Behavioral verification gate: can the model answer "is this formalization
correct?" above the ~0.65 threshold on a balanced contrast_v1 sample?

Greedy chat generation, one-word answer, parsed by the first standalone
yes/no token. Qwen3 runs with thinking disabled (no-think protocol).
Sample: problems_per_tier per tier, seeded, both texts per problem.

    bash behavior/run_verify.sh <model-key> [--limit N]

Model keys and GPUs come from config.VERIFY; run_verify.sh sets VERIFY_GPU
so the Modal function is defined on the right hardware. Run record:
runs/verify-v1/<model-key>.json.
"""

import hashlib
import importlib.util
import json
import os
import random
import re
from pathlib import Path

import modal

STAGE = Path(__file__).resolve().parent
PX_ROOT = STAGE.parent

GPU = os.environ.get("VERIFY_GPU", "A10G")


def load_config():
    """Local-side only: this module is also imported inside the container,
    where the repo tree does not exist, so config must load lazily."""
    spec = importlib.util.spec_from_file_location(
        "px_root_config", PX_ROOT / "config.py"
    )
    pxc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pxc)
    return pxc

app = modal.App("harsh-probe-verify")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "torch", "transformers", "accelerate", "huggingface_hub[hf_transfer]"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


def extract(text: str):
    m = re.search(r"\b(yes|no)\b", text.lower())
    return m.group(1) if m else None


@app.function(
    gpu=GPU,
    image=image,
    volumes={"/models": weights},
    secrets=[hf_secret],
    timeout=3600,
    retries=0,
    max_containers=1,
    scaledown_window=60,
)
def verify(items: list, config: dict) -> dict:
    import time

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config["model_id"]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        eos = tok.eos_token_id
        tok.pad_token = tok.convert_ids_to_tokens(
            eos[0] if isinstance(eos, (list, tuple)) else eos
        )
    kw = (
        {"dtype": torch.bfloat16}
        if int(transformers.__version__.split(".")[0]) >= 5
        else {"torch_dtype": torch.bfloat16}
    )
    multi_gpu = ":" in config["gpu"]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto" if multi_gpu else "cuda", **kw
    )
    model.eval()
    load_s = time.time() - t0

    chat_kw = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}],
            tokenize=False, add_generation_prompt=True, **chat_kw,
        )
        for it in items
    ]

    outputs = []
    t1 = time.time()
    bs = config["batch_size"]
    for b0 in range(0, len(prompts), bs):
        enc = tok(prompts[b0 : b0 + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=config["max_new_tokens"], do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        outputs.extend(
            tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for g in gen
        )
    gen_s = time.time() - t1

    rows = []
    for it, out in zip(items, outputs):
        pred = extract(out)
        rows.append({"id": it["id"], "label": it["label"], "pred": pred,
                     "raw": out.strip()[:120]})
    parsed = [r for r in rows if r["pred"] is not None]
    correct = [r for r in parsed if (r["pred"] == "yes") == (r["label"] == 1)]
    strict_correct = len(correct)
    by_tier = {}
    for t in sorted({it["tier"] for it in items}):
        sub = [r for r, it in zip(rows, items) if it["tier"] == t]
        ok = sum(1 for r in sub if r["pred"] is not None
                 and (r["pred"] == "yes") == (r["label"] == 1))
        by_tier[t] = round(ok / len(sub), 4)
    return {
        "n": len(rows),
        "unparseable": len(rows) - len(parsed),
        "acc_strict": round(strict_correct / len(rows), 4),
        "acc_parsed": round(len(correct) / len(parsed), 4) if parsed else None,
        "yes_rate": round(sum(1 for r in parsed if r["pred"] == "yes")
                          / len(parsed), 4) if parsed else None,
        "by_tier_strict": by_tier,
        "gpu_seconds": {"load": round(load_s, 1), "generate": round(gen_s, 1)},
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
        "rows": rows,
    }


@app.local_entrypoint()
def main(model: str, limit: int = 0, dry_run: bool = False):
    pxc = load_config()
    mcfg = pxc.VERIFY["models"][model]
    template = (STAGE / "verify_prompt.md").read_text(encoding="utf-8")

    rows = [
        json.loads(line)
        for line in (PX_ROOT / "contrast_v1" / "contrast.jsonl").open()
    ]
    rng = random.Random(pxc.VERIFY["sample_seed"])
    sample = []
    for t in ("easy", "medium", "hard"):
        pool = [r for r in rows if r["tier"] == t]
        sample.extend(rng.sample(pool, pxc.VERIFY["problems_per_tier"]))
    if limit:
        sample = sample[:limit]

    items = []
    for row in sample:
        for kind, label in (("correct", 1), ("wrong", 0)):
            items.append({
                "id": f"{row['problem_id']}::{kind}",
                "prompt": template.replace("{story}", row["story"])
                                  .replace("{rg}", row[f"{kind}_rg"]),
                "label": label,
                "tier": row["tier"],
            })

    config = {
        "model_key": model,
        "model_id": mcfg["id"],
        "gpu": GPU,
        "batch_size": 8 if ":" in GPU else 16,
        "max_new_tokens": pxc.VERIFY["max_new_tokens"],
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "sample_seed": pxc.VERIFY["sample_seed"],
        "problems_per_tier": pxc.VERIFY["problems_per_tier"],
        "limit": limit,
        "threshold": pxc.VERIFY["threshold"],
    }
    if dry_run:
        labels = [it["label"] for it in items]
        print(f"items: {len(items)}  correct/wrong: {sum(labels)}/{len(labels) - sum(labels)}")
        print(f"tiers: {sorted({it['tier'] for it in items})}")
        print(f"config: {json.dumps(config, indent=2)}")
        print(f"--- first prompt ---\n{items[0]['prompt']}")
        return

    result = verify.remote(items, config)

    out_dir = PX_ROOT / "runs" / "verify-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"stage": "verify-gate", **config, **result}
    tag = model if not limit else f"{model}-limit{limit}"
    (out_dir / f"{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
    keep = {k: v for k, v in record.items() if k != "rows"}
    print(json.dumps(keep, indent=2))
