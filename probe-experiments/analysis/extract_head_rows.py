#!/usr/bin/env python3
"""Extract Qwen3-32B final-norm weight + yes/no lm_head rows (tiny tensors)
for the depth-resolved margin lens. Runs on Modal CPU against the cached
checkpoint; saves ~100KB npz locally via stdout-safe volume path."""

import json
import os
from pathlib import Path

import modal

app = modal.App("harsh-head-extract")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "numpy", "safetensors",
                    "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
acts_vol = modal.Volume.from_name("harsh-probe-activations", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(image=image, volumes={"/models": weights, "/acts": acts_vol},
              secrets=[hf_secret], timeout=1200, retries=0)
def extract() -> dict:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break
    import numpy as np
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    model_id = "Qwen/Qwen3-32B"
    tok = AutoTokenizer.from_pretrained(model_id)
    variants = lambda w: sorted({
        tok(v, add_special_tokens=False).input_ids[0]
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize())
    })
    yes_ids, no_ids = variants("yes"), variants("no")

    local = snapshot_download(model_id, allow_patterns=["*.safetensors.index.json"])
    index = json.loads((Path(local) / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    need = {"model.norm.weight", "lm_head.weight"}
    shards = {wmap[k] for k in need}
    local = snapshot_download(model_id, allow_patterns=[*shards, "*.index.json"])

    out = {}
    for name in need:
        with safe_open(str(Path(local) / wmap[name]), framework="pt") as f:
            if name == "lm_head.weight":
                t = f.get_slice(name)
                out["head_rows"] = {
                    i: t[i].float().numpy() for i in (*yes_ids, *no_ids)
                }
            else:
                out["norm_weight"] = f.get_tensor(name).float().numpy()

    os.makedirs("/acts/lens", exist_ok=True)
    np.savez("/acts/lens/qwen3-32b-head-rows.npz",
             norm_weight=out["norm_weight"],
             yes_ids=np.array(yes_ids), no_ids=np.array(no_ids),
             **{f"row_{i}": v for i, v in out["head_rows"].items()})
    acts_vol.commit()
    return {"yes_ids": yes_ids, "no_ids": no_ids,
            "d": int(out["norm_weight"].shape[0])}


@app.local_entrypoint()
def main():
    print(json.dumps(extract.remote(), indent=2))
