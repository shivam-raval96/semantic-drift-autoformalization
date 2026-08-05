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

import json
from pathlib import Path

import modal

app = modal.App("harsh-probe-capture")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "torch", "transformers", "accelerate", "numpy", "huggingface_hub[hf_transfer]"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
acts_vol = modal.Volume.from_name("harsh-probe-activations", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 8


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/models": weights, "/acts": acts_vol},
    secrets=[hf_secret],
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=60,
)
def capture(items: list, config: dict) -> dict:
    import os
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
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
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
    model = AutoModel.from_pretrained(MODEL_ID, **kw).to("cuda")
    model.eval()
    load_s = time.time() - t0

    # Tokenize once, unpadded; the answer span starts where the prefix ends.
    ids_all = [tok(it["text"]).input_ids for it in items]
    span_starts = [len(tok(it["prefix"]).input_ids) for it in items]
    for it, ids, start in zip(items, ids_all, span_starts):
        assert start < len(ids), f"{it['id']}: empty answer span"

    order = sorted(range(len(items)), key=lambda i: len(ids_all[i]))
    acts_last, acts_mean = {}, {}
    t1 = time.time()
    for b0 in range(0, len(order), BATCH_SIZE):
        batch = order[b0 : b0 + BATCH_SIZE]
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
        "model": MODEL_ID,
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
def main(limit: int = 0, tag: str = ""):
    stage = Path(__file__).resolve().parent
    root = stage.parent
    manifest = json.loads((root / "contrast_v1" / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (root / "contrast_v1" / "contrast.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if limit:
        rows = rows[:limit]
    tag = tag or ("full" if not limit else f"limit{limit}")

    items = []
    for row in rows:
        prefix = row["story"] + "\n\n"
        for kind, label in (("correct", 1), ("wrong", 0)):
            items.append(
                {
                    "id": f"{row['problem_id']}::{kind}",
                    "text": prefix + row[f"{kind}_rg"],
                    "prefix": prefix,
                    "label": label,
                }
            )

    config = {
        "tag": tag,
        "limit": limit,
        "contrast_version": "v1",
        "contrast_sha256": manifest["files"]["contrast.jsonl"],
        "batch_size": BATCH_SIZE,
    }
    summary = capture.remote(items, config)

    runs_dir = root / "runs" / "capture-v1"
    runs_dir.mkdir(parents=True, exist_ok=True)
    record = {"stage": "capture", **config, **summary}
    (runs_dir / f"{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
