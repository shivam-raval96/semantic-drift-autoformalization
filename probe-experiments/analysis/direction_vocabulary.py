#!/usr/bin/env python3
"""What would the model be disposed to SAY along our correctness direction?

The J-lens readout for a residual direction v at block L is  (W_U J_L) v :
one score per vocabulary token, the degree to which v raises the model's
disposition to eventually emit that token. Applying it to our supervised
probe direction turns an abstract geometric object into words.

Three readouts, all at the same block:
  probe direction        - what our correctness direction "means" verbally
  negated probe          - the other pole
  random direction       - the null: what an arbitrary direction reads as

Interpretation. If the probe direction's top tokens are semantically related
to correctness/error, the direction IS verbalizable (inside the workspace)
even though it is orthogonal to the yes/no axis. If they look like arbitrary
vocabulary, the direction is outside the verbalizable workspace entirely -
which is the stronger form of the "represented but not read" claim.

Runs on Modal CPU: needs the full lm_head (151936 x 5120) and one Jacobian.
"""

import json
import os
from pathlib import Path

import modal

STAGE = Path(__file__).resolve().parent
PX_ROOT = STAGE.parent

app = modal.App("harsh-direction-vocab")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch", "transformers>=5.5", "numpy", "safetensors",
                    "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
acts_vol = modal.Volume.from_name("harsh-probe-activations", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL_ID = "Qwen/Qwen3-32B"
LENS = ("neuronpedia/jacobian-lens",
        "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt")


@app.function(image=image, volumes={"/models": weights, "/acts": acts_vol},
              secrets=[hf_secret], timeout=3600, retries=0, cpu=8, memory=65536)
def readout(direction: list, block: int, topk: int = 30,
            extra: dict | None = None) -> dict:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    idx = snapshot_download(MODEL_ID, allow_patterns=["*.index.json"])
    wmap = json.loads((Path(idx) / "model.safetensors.index.json").read_text())["weight_map"]
    shard = snapshot_download(MODEL_ID, allow_patterns=[wmap["lm_head.weight"],
                                                        "*.index.json"])
    with safe_open(str(Path(shard) / wmap["lm_head.weight"]), framework="pt") as f:
        W_U = f.get_tensor("lm_head.weight").float()      # [vocab, d]

    blob = torch.load(hf_hub_download(*LENS), map_location="cpu", weights_only=True)
    n = int(blob["n_done"])
    J = (blob["jacobian_sum"][block].float() / n)          # [d, d]

    v_probe = torch.tensor(direction, dtype=torch.float32)
    v_probe = v_probe / v_probe.norm()
    g = torch.Generator().manual_seed(0)
    v_rand = torch.randn(len(direction), generator=g)
    v_rand = v_rand / v_rand.norm()

    out = {"model": MODEL_ID, "block": block, "lens_n_prompts": n,
           "readouts": {}}
    cands = [("probe", v_probe), ("probe_negated", -v_probe),
             ("random", v_rand)]
    # POSITIVE CONTROL: a direction we KNOW is verbalizable. The J-lens yes/no
    # direction must read out as yes/no-ish tokens; if it does not, the
    # readout code (not the probe direction) is what is broken.
    for k, vec in (extra or {}).items():
        t = torch.tensor(vec, dtype=torch.float32)
        cands.append((k, t / t.norm()))
    for name, v in cands:
        scores = W_U @ (J @ v)                             # [vocab]
        top = torch.topk(scores, topk)
        out["readouts"][name] = {
            "top_tokens": [tok.decode([int(i)]) for i in top.indices],
            "top_scores": [round(float(s), 4) for s in top.values],
            "score_std": round(float(scores.std()), 4),
            "score_max": round(float(scores.max()), 4),
        }
    return out


@app.local_entrypoint()
def main(block: int = 60, topk: int = 30):
    import numpy as np

    d = np.load(PX_ROOT / "runs" / "probe-32b" / "direction_L61_mean.npz")
    lens = np.load(PX_ROOT / "capture" / "qwen3-32b-lens-dirs.npz")
    layers = list(lens["jlens_layers"])
    jdir = lens["jlens_dirs"][layers.index(block)].astype(float)
    extra = {"jlens_yesno_POSITIVE_CONTROL": jdir.tolist(),
             "unembed_yesno_raw": lens["contrast_unembed"].astype(float).tolist()}
    res = readout.remote(d["direction"].astype(float).tolist(), block, topk,
                         extra)
    o = PX_ROOT / "runs" / "lens-v1"
    o.mkdir(parents=True, exist_ok=True)
    (o / "direction_vocabulary.json").write_text(json.dumps(res, indent=2) + "\n")
    for name, r in res["readouts"].items():
        print(f"\n[{name}] max {r['score_max']} sd {r['score_std']}")
        print("  " + " | ".join(repr(t) for t in r["top_tokens"][:15]))
    print(f"\n-> {o / 'direction_vocabulary.json'}")
