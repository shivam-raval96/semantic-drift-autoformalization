#!/usr/bin/env python3
"""Capture Llama-3.1-8B-Instruct residual-stream activations over contrast_v1.

Reader mode: the model reads   story + "\n\n" + candidate RG answer   as bare
text (no chat template — the direction must not bake in instruction wording).
Per text, two sites at every layer (embeddings + 32 blocks = 33 rows):

  last — hidden state at the final token of the answer
  mean — mean over the answer-region tokens (positions after the story prefix)

Capture code follows certificate-pipeline's pipeline/hf_backend.py
(output_hidden_states, float16, npz keyed by item id). Modal scaffolding
follows ft-experiments/training/train_lora.py (same image recipe and the same
weights volume, so the model is already cached at /models/hf).

Outputs on volume harsh-probe-activations under /acts/contrast_v1/<tag>/:
  acts-last.npz, acts-mean.npz    item_id -> (33, 4096) float16
  meta.json                       ids, labels, config, token stats
Run record: probe-experiments/runs/capture-v1/<tag>.json

  modal run capture/modal_capture.py --limit 4 --tag smoke    # smoke first
  modal run capture/modal_capture.py                          # full 2,000 texts
"""

import importlib.util
import json
import os
from pathlib import Path

import modal

STAGE = Path(__file__).resolve().parent
PX_ROOT = STAGE.parent

GPU = os.environ.get("CAPTURE_GPU", "A10G")


def load_config():
    """Local-side only: this module is also imported inside the container,
    where the repo tree does not exist, so config must load lazily."""
    spec = importlib.util.spec_from_file_location(
        "px_root_config", PX_ROOT / "config.py"
    )
    pxc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pxc)
    return pxc


app = modal.App("harsh-probe-capture")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "torch", "transformers", "accelerate", "numpy", "peft", "huggingface_hub[hf_transfer]"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
acts_vol = modal.Volume.from_name("harsh-probe-activations", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

@app.function(
    gpu=GPU,
    image=image,
    volumes={"/models": weights, "/acts": acts_vol},
    secrets=[hf_secret],
    timeout=3600,
    retries=0,
    max_containers=1,
    scaledown_window=60,
)
def capture(items: list, config: dict) -> dict:
    import time

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break

    # Big-vocab buffers were the FT run's OOM too; same allocator fix.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    model_id = config["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
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
    # Transformer body only: we need hidden states, never logits — the LM
    # head would add a (batch, seq, 128256) buffer per forward plus ~1GB
    # of weights, which is exactly what OOMed the first full run.
    multi_gpu = ":" in config["gpu"]
    if config.get("adapter"):
        # LoRA adapters carry CausalLM key paths: load the LM, merge the
        # adapter into the weights, then keep only the transformer body.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        lm = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto" if multi_gpu else "cuda", **kw
        )
        lm = PeftModel.from_pretrained(lm, config["adapter"]).merge_and_unload()
        model = lm.model
    else:
        model = AutoModel.from_pretrained(
            model_id, device_map="auto" if multi_gpu else "cuda", **kw
        )
    model.eval()
    load_s = time.time() - t0

    if config.get("chat_mode"):
        chat_kw = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
        for it in items:
            it["text"] = tok.apply_chat_template(
                [{"role": "user", "content": it["text"]}],
                tokenize=False, add_generation_prompt=True, **chat_kw,
            )
            it["prefix"] = it["text"][: len(it["text"]) // 2]  # span = tail half

    # Tokenize once, unpadded; the answer span starts where the prefix ends.
    ids_all = [tok(it["text"]).input_ids for it in items]
    span_starts = [len(tok(it["prefix"]).input_ids) for it in items]
    for it, ids, start in zip(items, ids_all, span_starts):
        assert start < len(ids), f"{it['id']}: empty answer span"

    order = sorted(range(len(items)), key=lambda i: len(ids_all[i]))
    acts_last, acts_mean = {}, {}
    t1 = time.time()
    bs = config["batch_size"]
    for b0 in range(0, len(order), bs):
        batch = order[b0 : b0 + bs]
        enc = tok.pad(
            {"input_ids": [ids_all[i] for i in batch]}, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple(n_layers+1) of (B, L, d)
        mask = enc["attention_mask"]
        for row, i in enumerate(batch):
            real = int(mask[row].sum())
            start = span_starts[i]
            acts_last[items[i]["id"]] = np.stack(
                [h[row, real - 1].float().cpu().numpy().astype(np.float16) for h in hs]
            )
            acts_mean[items[i]["id"]] = np.stack(
                [
                    h[row, start:real].float().mean(dim=0).cpu().numpy().astype(np.float16)
                    for h in hs
                ]
            )
        del out, hs
    fwd_s = time.time() - t1

    out_dir = f"/acts/contrast_v1/{config['tag']}"
    os.makedirs(out_dir, exist_ok=True)
    t2 = time.time()
    np.savez_compressed(f"{out_dir}/acts-last.npz", **acts_last)
    np.savez_compressed(f"{out_dir}/acts-mean.npz", **acts_mean)
    lengths = [len(x) for x in ids_all]
    ans_lengths = [len(x) - s for x, s in zip(ids_all, span_starts)]
    meta = {
        "model": model_id,
        "n_texts": len(items),
        "layers": len(next(iter(acts_last.values()))),
        "d_model": int(next(iter(acts_last.values())).shape[1]),
        "sites": ["last", "mean"],
        "text_template": "story + '\\n\\n' + rg (bare, no chat template)",
        "ids": [it["id"] for it in items],
        "labels": {it["id"]: it["label"] for it in items},
        "token_len": {"min": min(lengths), "max": max(lengths),
                      "mean": round(sum(lengths) / len(lengths), 1)},
        "answer_token_len": {"min": min(ans_lengths), "max": max(ans_lengths),
                             "mean": round(sum(ans_lengths) / len(ans_lengths), 1)},
        "config": config,
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    with open(f"{out_dir}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    # A corrupt archive once reached the volume; never publish unverified.
    import zipfile
    for name in ("acts-last.npz", "acts-mean.npz"):
        bad = zipfile.ZipFile(f"{out_dir}/{name}").testzip()
        assert bad is None, f"{name} corrupt at {bad}"
    acts_vol.commit()
    save_s = time.time() - t2

    return {
        "out_dir": out_dir,
        "n_texts": len(items),
        "shape_per_item": list(next(iter(acts_last.values())).shape),
        "token_len": meta["token_len"],
        "answer_token_len": meta["answer_token_len"],
        "gpu_seconds": {"load": round(load_s, 1), "forward": round(fwd_s, 1),
                        "save": round(save_s, 1)},
        "versions": meta["versions"],
    }


@app.local_entrypoint()
def main(model: str = "llama-3.1-8b", limit: int = 0, tag: str = "",
         dry_run: bool = False, adapter: str = "", asked: bool = False):
    root = PX_ROOT
    pxc = load_config()
    mcfg = pxc.MODELS[model]
    manifest = json.loads((root / "contrast_v1" / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (root / "contrast_v1" / "contrast.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if limit and not asked:
        rows = rows[:limit]
    tag = tag or (f"{model}-full" if not limit else f"{model}-limit{limit}")
    if asked and not tag.startswith("asked"):
        tag = f"asked-{tag}"

    items = []
    if asked:
        template = (root / "behavior" / "verify_prompt.md").read_text(encoding="utf-8")
        import random as _random
        rng = _random.Random(pxc.VERIFY["sample_seed"])
        sample = []
        for t in ("easy", "medium", "hard"):
            pool = [r for r in rows if r["tier"] == t]
            sample.extend(rng.sample(pool, pxc.VERIFY["problems_per_tier"]))
        rows = sample
        if limit:
            rows = rows[:limit]
    for row in rows:
        for kind, label in (("correct", 1), ("wrong", 0)):
            if asked:
                items.append({
                    "id": f"{row['problem_id']}::{kind}",
                    "text": template.replace("{story}", row["story"])
                                    .replace("{rg}", row[f"{kind}_rg"]),
                    "prefix": "",
                    "label": label,
                })
            else:
                prefix = row["story"] + "\n\n"
                items.append({
                    "id": f"{row['problem_id']}::{kind}",
                    "text": prefix + row[f"{kind}_rg"],
                    "prefix": prefix,
                    "label": label,
                })

    config = {
        "tag": tag,
        "model_key": model,
        "model_id": mcfg["id"],
        "adapter": adapter or None,
        "gpu": GPU,
        "limit": limit,
        "contrast_version": "v1",
        "contrast_sha256": manifest["files"]["contrast.jsonl"],
        "chat_mode": asked,
        "batch_size": 4 if ":" in GPU else 8,
    }
    if dry_run:
        print(f"items: {len(items)}")
        print(f"config: {json.dumps(config, indent=2)}")
        print(f"--- first text (tail) ---\n...{items[0]['text'][-200:]}")
        return

    summary = capture.remote(items, config)

    runs_dir = root / "runs" / "capture-v1"
    runs_dir.mkdir(parents=True, exist_ok=True)
    record = {"stage": "capture", **config, **summary}
    (runs_dir / f"{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
