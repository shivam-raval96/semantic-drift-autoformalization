#!/usr/bin/env python3
"""Extract yes/no causal directions from the pre-fitted J-lens and R-lens for
Qwen3.6-27B, plus that model's readout tensors.

J-lens vector for token t at layer L is row t of W_U J_L, i.e. W_U[t] @ J_L
(Anthropic workspace paper; same convention as anthropics/jacobian-lens).
The yes/no contrast direction at layer L is therefore

    v_L = (mean W_U[yes_ids] - mean W_U[no_ids]) @ J_L

which is the residual-stream direction that most raises the model's
disposition to eventually emit "yes" over "no" from layer L. Compare this to
our supervised probe direction: if they are near-orthogonal, the probe reads
something the verbalization pathway does not use, which is the workspace
explanation for our steering null.

Downloads (~6.6GB of lenses) happen in the container; only ~1MB of directions
comes back. Runs on CPU.
"""

import json
import os
from pathlib import Path

import modal

app = modal.App("harsh-lens-directions")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch", "transformers>=5.5", "numpy", "safetensors", "datasets",
        "huggingface_hub[hf_transfer]",
        "git+https://github.com/anthropics/jacobian-lens.git",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
acts_vol = modal.Volume.from_name("harsh-probe-activations", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL_ID = "Qwen/Qwen3.6-27B"
LENSES = {
    "jlens": ("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt"),
    "rlens": ("camilablank/workspace-lenses", "qwen3.6-27b/r-lens/lens.pt"),
}


@app.function(image=image, volumes={"/models": weights, "/acts": acts_vol},
              secrets=[hf_secret], timeout=3600, retries=0, cpu=4, memory=32768)
def extract() -> dict:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from jlens import JacobianLens
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    variants = lambda w: sorted({
        tok(v, add_special_tokens=False).input_ids[0]
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize())
    })
    yes_ids, no_ids = variants("yes"), variants("no")

    idx_dir = snapshot_download(MODEL_ID, allow_patterns=["*.index.json"])
    index = json.loads((Path(idx_dir) / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    need = {k: wmap[k] for k in ("model.norm.weight", "lm_head.weight")
            if k in wmap}
    tied = "lm_head.weight" not in wmap
    if tied:
        need["model.embed_tokens.weight"] = wmap["model.embed_tokens.weight"]
    local = snapshot_download(MODEL_ID, allow_patterns=[*set(need.values()),
                                                        "*.index.json"])
    head_key = "model.embed_tokens.weight" if tied else "lm_head.weight"
    with safe_open(str(Path(local) / need[head_key]), framework="pt") as f:
        head = f.get_slice(head_key)
        yes_rows = torch.stack([head[i].float() for i in yes_ids])
        no_rows = torch.stack([head[i].float() for i in no_ids])
    with safe_open(str(Path(local) / need["model.norm.weight"]), framework="pt") as f:
        norm_w = f.get_tensor("model.norm.weight").float()

    contrast = (yes_rows.mean(0) - no_rows.mean(0))  # [d]
    out = {"norm_weight": norm_w.numpy(),
           "yes_ids": np.array(yes_ids), "no_ids": np.array(no_ids),
           "yes_rows": yes_rows.numpy(), "no_rows": no_rows.numpy(),
           "contrast_unembed": contrast.numpy()}
    summary = {"model": MODEL_ID, "tied_embeddings": bool(tied),
               "yes_ids": yes_ids, "no_ids": no_ids, "lenses": {}}

    for name, (repo, filename) in LENSES.items():
        lens = JacobianLens.from_pretrained(repo, filename=filename)
        layers = sorted(lens.jacobians.keys())
        dirs, norms = [], []
        for L in layers:
            J = lens.jacobians[L].float()          # [d, d]
            v = contrast @ J                        # row-vector convention
            dirs.append(v.numpy())
            norms.append(float(v.norm()))
        out[f"{name}_dirs"] = np.stack(dirs)
        out[f"{name}_layers"] = np.array(layers)
        summary["lenses"][name] = {
            "repo": repo, "filename": filename, "n_prompts": int(lens.n_prompts),
            "d_model": int(lens.d_model), "layers": layers,
            "dir_norm_min": round(min(norms), 3),
            "dir_norm_max": round(max(norms), 3),
        }
        del lens

    os.makedirs("/acts/lens", exist_ok=True)
    np.savez("/acts/lens/qwen3.6-27b-lens-dirs.npz", **out)
    acts_vol.commit()
    return summary


@app.local_entrypoint()
def main():
    s = extract.remote()
    print(json.dumps(s, indent=2))
    out = Path(__file__).resolve().parents[1] / "runs" / "lens-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lens_directions_meta.json").write_text(json.dumps(s, indent=2) + "\n")
