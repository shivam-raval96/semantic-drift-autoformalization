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

# Qwen3-32B is our primary model (all probe/steering results live there) AND
# a clean dense transformer, so steering is position-local. Free J-lens from
# the Neuronpedia batch (n=1000, wikitext, Anthropic jlens recipe).
TARGETS = {
    "qwen3-32b": {
        "model_id": "Qwen/Qwen3-32B",
        "lenses": {"jlens": ("neuronpedia/jacobian-lens",
                             "qwen3-32b/jlens/Salesforce-wikitext/"
                             "Qwen3-32B_jacobian_lens.pt")},
    },
    "qwen3.6-27b": {
        "model_id": "Qwen/Qwen3.6-27B",
        "lenses": {"jlens": ("camilablank/workspace-lenses",
                             "qwen3.6-27b/j-lens/lens.pt"),
                   "rlens": ("camilablank/workspace-lenses",
                             "qwen3.6-27b/r-lens/lens.pt")},
    },
}


@app.function(image=image, volumes={"/models": weights, "/acts": acts_vol},
              secrets=[hf_secret], timeout=3600, retries=0, cpu=4, memory=32768)
def extract(target: str) -> dict:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from jlens import JacobianLens
    from safetensors import safe_open
    from transformers import AutoTokenizer

    cfg = TARGETS[target]
    model_id, lenses = cfg["model_id"], cfg["lenses"]
    tok = AutoTokenizer.from_pretrained(model_id)
    variants = lambda w: sorted({
        tok(v, add_special_tokens=False).input_ids[0]
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize())
    })
    yes_ids, no_ids = variants("yes"), variants("no")

    idx_dir = snapshot_download(model_id, allow_patterns=["*.index.json"])
    index = json.loads((Path(idx_dir) / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    # Key names differ across wrappers (plain vs ForConditionalGeneration,
    # which nests the text stack under model.language_model.*). Detect.
    norm_key = next(k for k in wmap
                    if k.endswith("norm.weight") and "layers." not in k)
    head_key = ("lm_head.weight" if "lm_head.weight" in wmap
                else next(k for k in wmap if k.endswith("embed_tokens.weight")))
    tied = head_key != "lm_head.weight"
    need = {norm_key: wmap[norm_key], head_key: wmap[head_key]}
    local = snapshot_download(model_id, allow_patterns=[*set(need.values()),
                                                        "*.index.json"])
    with safe_open(str(Path(local) / need[head_key]), framework="pt") as f:
        head = f.get_slice(head_key)
        yes_rows = torch.stack([head[i].float() for i in yes_ids])
        no_rows = torch.stack([head[i].float() for i in no_ids])
    with safe_open(str(Path(local) / need[norm_key]), framework="pt") as f:
        norm_w = f.get_tensor(norm_key).float()

    contrast = (yes_rows.mean(0) - no_rows.mean(0))  # [d]
    out = {"norm_weight": norm_w.numpy(),
           "yes_ids": np.array(yes_ids), "no_ids": np.array(no_ids),
           "yes_rows": yes_rows.numpy(), "no_rows": no_rows.numpy(),
           "contrast_unembed": contrast.numpy()}
    summary = {"model": model_id, "target": target, "norm_key": norm_key,
               "head_key": head_key, "tied_embeddings": bool(tied),
               "yes_ids": yes_ids, "no_ids": no_ids, "lenses": {}}

    for name, (repo, filename) in lenses.items():
        # Two on-Hub formats: a saved lens ({"J": ...}) and a raw fit()
        # checkpoint ({"jacobian_sum", "n_done", ...}). The lens is the
        # running mean, so the checkpoint converts exactly.
        path = hf_hub_download(repo, filename)
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if "J" in blob:
            lens = JacobianLens.load(path)
            jac, n_prompts = lens.jacobians, int(lens.n_prompts)
        elif "jacobian_sum" in blob:
            n_prompts = int(blob["n_done"])
            jac = {int(L): (v.float() / n_prompts)
                   for L, v in blob["jacobian_sum"].items()}
            print(f"[{name}] fit-checkpoint -> mean over {n_prompts} prompts")
        else:
            raise ValueError(f"{filename}: unknown lens format {list(blob)[:6]}")
        lens = type("L", (), {"jacobians": jac, "n_prompts": n_prompts,
                              "d_model": next(iter(jac.values())).shape[0]})()
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
    np.savez(f"/acts/lens/{target}-lens-dirs.npz", **out)
    acts_vol.commit()
    return summary


@app.local_entrypoint()
def main(target: str = "qwen3-32b"):
    s = extract.remote(target)
    print(json.dumps(s, indent=2))
    out = Path(__file__).resolve().parents[1] / "runs" / "lens-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lens_directions_{target}.json").write_text(
        json.dumps(s, indent=2) + "\n")
