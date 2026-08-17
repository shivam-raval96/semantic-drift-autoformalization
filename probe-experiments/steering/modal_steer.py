#!/usr/bin/env python3
"""Steering: does the layer-61 correctness direction causally move the
model's own verification readout?

Injects alpha * mean_residual_norm * unit_direction into the output of
decoder layer 60 of Qwen3-32B (= capture row 61) at ALL positions during a
forward pass over the 300 gate texts, and reads the yes/no logit margin.
Conditions: the probe direction at several alphas (negatives = negated-
vector control) and a norm-matched random direction. One engine load, all
conditions in one remote call. Readout per condition: margin AUROC vs
labels + sign accuracy. Design follows mech-interp exp-1 Part 3
conventions (normalized scale, negated + random controls).

    modal run steering/modal_steer.py --dry-run
    modal run steering/modal_steer.py --limit 10     # smoke
    modal run steering/modal_steer.py                # full 300 texts
"""

import bisect
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path

import modal

STAGE = Path(__file__).resolve().parent
PX_ROOT = STAGE.parent

GPU = os.environ.get("STEER_GPU", "A100-80GB")

ALPHAS = [0.0, 0.25, 0.5, 1.0, -0.5, -1.0]
RANDOM_ALPHAS = [0.5, 1.0]


def load_config():
    spec = importlib.util.spec_from_file_location(
        "px_root_config", PX_ROOT / "config.py"
    )
    pxc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pxc)
    return pxc


app = modal.App("harsh-probe-steer")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "torch", "transformers", "accelerate", "numpy", "huggingface_hub[hf_transfer]"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


def auroc(rows):
    pos = sorted(r["margin"] for r in rows if r["label"] == 1)
    neg = sorted(r["margin"] for r in rows if r["label"] == 0)
    if not pos or not neg:
        return None
    ranks = sum(
        bisect.bisect_left(neg, x)
        + (bisect.bisect_right(neg, x) - bisect.bisect_left(neg, x)) / 2.0
        for x in pos
    )
    return round(ranks / (len(pos) * len(neg)), 4)


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
def steer(items: list, config: dict) -> dict:
    import time

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
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
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", **kw)
    model.eval()
    layers = model.model.layers
    assert len(layers) == config["n_layers"], f"{len(layers)} layers, want {config['n_layers']}"
    load_s = time.time() - t0

    direction = torch.tensor(config["direction"], dtype=torch.bfloat16, device="cuda")
    rng = np.random.RandomState(config["random_seed"])
    rand = rng.standard_normal(len(config["direction"]))
    rand = torch.tensor(rand / np.linalg.norm(rand), dtype=torch.bfloat16, device="cuda")

    inject = {"vec": None}

    def hook(_module, _inp, out):
        if inject["vec"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + inject["vec"],) + out[1:]
        return out + inject["vec"]

    handle = layers[config["hook_layer"]].register_forward_hook(hook)

    variants = lambda w: sorted({
        tok(v, add_special_tokens=False).input_ids[0]
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize())
    })
    yes_ids, no_ids = variants("yes"), variants("no")

    chat_kw = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}],
            tokenize=False, add_generation_prompt=True, **chat_kw,
        )
        for it in items
    ]

    conditions = [("direction", a) for a in config["alphas"]] + [
        ("random", a) for a in config["random_alphas"]
    ]
    results = {}
    t1 = time.time()
    bs = config["batch_size"]
    for kind, alpha in conditions:
        vec = direction if kind == "direction" else rand
        inject["vec"] = None if alpha == 0.0 else (
            float(alpha) * config["scale"] * vec
        )
        rows = []
        for b0 in range(0, len(prompts), bs):
            enc = tok(prompts[b0:b0 + bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            m = (logits[:, yes_ids].max(dim=1).values
                 - logits[:, no_ids].max(dim=1).values)
            for it, mg in zip(items[b0:b0 + bs], m.cpu().tolist()):
                rows.append({"id": it["id"], "label": it["label"],
                             "margin": round(mg, 4)})
        results[f"{kind}@{alpha:+.2f}"] = {
            "auroc": auroc(rows),
            "yes_sign_rate": round(sum(1 for r in rows if r["margin"] > 0)
                                   / len(rows), 4),
            "mean_margin": round(sum(r["margin"] for r in rows) / len(rows), 4),
            "rows": rows,
        }
    handle.remove()
    return {
        "conditions": results,
        "gpu_seconds": {"load": round(load_s, 1),
                        "forward": round(time.time() - t1, 1)},
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
    }


@app.local_entrypoint()
def main(limit: int = 0, dry_run: bool = False, direction_src: str = "probe"):
    import numpy as np

    pxc = load_config()
    model_key = "qwen3-32b"
    mcfg = pxc.MODELS[model_key]

    d = np.load(PX_ROOT / "runs" / "probe-32b" / "direction_L61_mean.npz")
    capture_layer = int(d["layer"])          # capture row index (emb + blocks)
    hook_layer = capture_layer - 1           # decoder layer whose OUTPUT is that row
    if direction_src == "probe":
        direction = d["direction"].astype(float)
    elif direction_src == "jlens":
        # POSITIVE CONTROL. The J-lens yes/no direction is, by construction,
        # the residual direction that most raises the disposition to say
        # "yes" over "no" from this layer. If steering along it does not move
        # the margin, the steering harness itself is not working and the
        # probe-direction null is uninformative.
        lens = np.load(PX_ROOT / "capture" / "qwen3-32b-lens-dirs.npz")
        layers = list(lens["jlens_layers"])
        idx = layers.index(hook_layer)
        v = lens["jlens_dirs"][idx].astype(float)
        direction = v / np.linalg.norm(v)
    else:
        raise SystemExit(f"unknown direction_src {direction_src}")
    direction = direction.tolist()

    acts_meta = json.loads((PX_ROOT / "capture" / "acts-qwen3-32b" / "meta.json").read_text())
    npz = np.load(PX_ROOT / "capture" / "acts-qwen3-32b" / "acts-last.npz")
    scale = float(np.mean([
        np.linalg.norm(npz[i][capture_layer].astype(np.float32))
        for i in acts_meta["ids"][:400]
    ]))

    template = (PX_ROOT / "behavior" / "verify_prompt.md").read_text(encoding="utf-8")
    rows = [json.loads(l) for l in (PX_ROOT / "contrast_v1" / "contrast.jsonl").open()]
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
            })

    config = {
        "model_key": model_key,
        "model_id": mcfg["id"],
        "n_layers": 64,
        "hook_layer": hook_layer,
        "capture_row": capture_layer,
        "site": "all positions, forward-only margin readout",
        "scale": scale,
        "alphas": ALPHAS,
        "random_alphas": RANDOM_ALPHAS,
        "random_seed": 0,
        "batch_size": 16,
        "direction_src": direction_src,
        "direction_sha256": hashlib.sha256(
            np.array(direction, dtype=np.float32).tobytes()).hexdigest()[:16],
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "limit": limit,
        "direction": direction,
    }
    if dry_run:
        show = {k: v for k, v in config.items() if k != "direction"}
        print(f"items: {len(items)}  conditions: "
              f"{[f'direction@{a}' for a in ALPHAS] + [f'random@{a}' for a in RANDOM_ALPHAS]}")
        print(json.dumps(show, indent=2))
        return

    result = steer.remote(items, config)

    out_dir = PX_ROOT / "runs" / "steer-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"qwen3-32b-{direction_src}" + (f"-limit{limit}" if limit else "")
    slim = {k: v for k, v in config.items() if k != "direction"}
    record = {"stage": "steering", **slim, **result}
    (out_dir / f"{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
    for cond, r in result["conditions"].items():
        print(f"{cond:16s} auroc {r['auroc']}  yes_rate {r['yes_sign_rate']}  "
              f"mean_margin {r['mean_margin']:+.3f}")
    print(f"-> {out_dir / f'{tag}.json'}")
