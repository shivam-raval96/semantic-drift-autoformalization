#!/usr/bin/env python3
"""Sanity-check the margin-lens reimplementation.

The lens computes  RMSNorm(h) * norm_w @ lm_head.T  from saved residual
states. Applied to the FINAL layer's state this must reproduce the model's
own logits. If it does not, every margin-lens number is suspect.

Also reports, for the same texts, the margin computed the two ways, so a
reviewer can see they agree exactly (not just correlate).

    modal run analysis/validate_lens.py --model qwen3-32b
"""

import importlib.util
import json
import os
from pathlib import Path

import modal

STAGE = Path(__file__).resolve().parent
PX_ROOT = STAGE.parent
GPU = os.environ.get("VALIDATE_GPU", "A100-80GB")

app = modal.App("harsh-lens-validate")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("torch", "transformers", "accelerate", "numpy",
                    "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


def load_config():
    spec = importlib.util.spec_from_file_location("px_root_config",
                                                  PX_ROOT / "config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@app.function(gpu=GPU, image=image, volumes={"/models": weights},
              secrets=[hf_secret], timeout=1800, retries=0, max_containers=1,
              scaledown_window=60)
def validate(model_id: str, texts: list) -> dict:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    kw = ({"dtype": torch.bfloat16}
          if int(transformers.__version__.split(".")[0]) >= 5
          else {"torch_dtype": torch.bfloat16})
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", **kw)
    model.eval()

    variants = lambda w: sorted({
        tok(v, add_special_tokens=False).input_ids[0]
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize())})
    yes_ids, no_ids = variants("yes"), variants("no")

    norm_w = model.model.norm.weight.detach().float()
    head = model.lm_head.weight.detach().float()
    eps = float(getattr(model.model.norm, "variance_epsilon",
                        getattr(model.config, "rms_norm_eps", 1e-6)))

    rows = []
    for text in texts:
        enc = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        true_logits = out.logits[0, -1].float()
        h = out.hidden_states[-1][0, -1].float()
        # HYPOTHESIS UNDER TEST: HF appends the last hidden state AFTER the
        # final norm, so hidden_states[-1] is already normed. Compare both.
        hn = h / torch.sqrt((h * h).mean() + eps) * norm_w
        lens_logits = hn @ head.T                     # with norm (our lens)
        raw_logits = h @ head.T                       # assuming pre-normed
        d_norm = (lens_logits - true_logits).abs().max().item()
        d_raw = (raw_logits - true_logits).abs().max().item()
        # second probe: is hidden_states[-2] a raw residual? norm it and see
        h2 = out.hidden_states[-2][0, -1].float()
        h2n = h2 / torch.sqrt((h2 * h2).mean() + eps) * norm_w
        d_prev_normed = (h2n @ head.T - true_logits).abs().max().item()
        # margins the two ways
        m_true = (true_logits[yes_ids].max() - true_logits[no_ids].max()).item()
        m_lens = (lens_logits[yes_ids].max() - lens_logits[no_ids].max()).item()
        diff = (lens_logits - true_logits).abs()
        rows.append({
            "n_tokens": int(enc["input_ids"].shape[1]),
            "diff_with_norm": round(d_norm, 4),
            "diff_no_norm": round(d_raw, 4),
            "diff_prevlayer_normed": round(d_prev_normed, 4),
            "max_abs_logit_diff": round(float(diff.max()), 4),
            "mean_abs_logit_diff": round(float(diff.mean()), 5),
            "true_logit_scale": round(float(true_logits.abs().mean()), 3),
            "margin_true": round(m_true, 4),
            "margin_lens": round(m_lens, 4),
            "margin_abs_diff": round(abs(m_true - m_lens), 5),
            "argmax_matches": bool(true_logits.argmax() == lens_logits.argmax()),
        })
    worst = max(r["max_abs_logit_diff"] for r in rows)
    worst_margin = max(r["margin_abs_diff"] for r in rows)
    return {
        "model": model_id, "n_texts": len(rows),
        "eps": eps, "yes_ids": yes_ids, "no_ids": no_ids,
        "worst_max_abs_logit_diff": worst,
        "worst_margin_abs_diff": worst_margin,
        "all_argmax_match": all(r["argmax_matches"] for r in rows),
        "worst_diff_no_norm": max(r["diff_no_norm"] for r in rows),
        "worst_diff_with_norm": max(r["diff_with_norm"] for r in rows),
        "diagnosis": ("hidden_states[-1] is PRE-NORMED (double-norm bug confirmed)"
                      if max(r["diff_no_norm"] for r in rows)
                      < max(r["diff_with_norm"] for r in rows) else
                      "hidden_states[-1] is a raw residual; bug is elsewhere"),
        "verdict": ("PASS" if worst_margin < 0.05 and
                    all(r["argmax_matches"] for r in rows) else "FAIL"),
        "rows": rows,
        "versions": {"torch": torch.__version__,
                     "transformers": transformers.__version__},
    }


@app.local_entrypoint()
def main(model: str = "qwen3-32b", n: int = 6, asked: bool = True):
    pxc = load_config()
    mcfg = pxc.MODELS[model]
    rows = [json.loads(l) for l in
            (PX_ROOT / "contrast_v1" / "contrast.jsonl").open()][:n]
    template = (PX_ROOT / "behavior" / "verify_prompt.md").read_text()
    texts = []
    for r in rows:
        if asked:
            texts.append(template.replace("{story}", r["story"])
                                 .replace("{rg}", r["correct_rg"]))
        else:
            texts.append(r["story"] + "\n\n" + r["correct_rg"])
    res = validate.remote(mcfg["id"], texts)
    out = PX_ROOT / "runs" / "lens-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"validate_lens_{model}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
