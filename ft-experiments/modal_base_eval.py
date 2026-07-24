#!/usr/bin/env python3
"""Phase 3 base evals on Modal: vLLM batch generation + local grading.

Remote side (Modal): one batch-generation function per GPU tier — A10G
for Llama-3.2-1B/3B, a single A100-40GB for Llama-3.1-8B — vLLM engine,
bf16, greedy (temperature 0, fixed max_tokens), HF weights cached on a
Volume, fast scaledown. The container never sees repo code; it maps
prompts to raw completions and reports backend versions.

Local side (this file's entrypoint, run via `modal run`): builds the
frozen-protocol prompts from eval_v1 (unchanged repo templates + the
byte-exact reasoning-off wrapper), sends them, grades raw responses with
checkform, and writes a benchmark-style run directory under
ft-experiments/runs/base-v1/ with backend + version + sampling params in
run_meta.json.

    modal run modal_base_eval.py --model 1b --arm story --limit 5   # smoke
    modal run modal_base_eval.py --model 8b --arm literal           # full
"""

import modal

app = modal.App("harsh-ft-base-eval")

# vLLM 0.25.x JIT-compiles flashinfer kernels at warmup, which needs the
# full CUDA toolkit (nvcc) — hence the devel base image, not debian_slim.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODELS = {
    "1b": ("meta-llama/Llama-3.2-1B-Instruct", "a10g"),
    "8b": ("meta-llama/Llama-3.1-8B-Instruct", "a100"),
    "70b": ("meta-llama/Llama-3.1-70B-Instruct", "a100_80_x4"),
}
MAX_TOKENS = 4096  # repo convention for the reasoning-off regime
MAX_MODEL_LEN = 8192  # prompts are ~2k tokens; caps KV allocation


def _generate(
    model_id: str, conversations: list, max_tokens: int, tensor_parallel: int = 1
) -> dict:
    import os
    import time

    # The shared workspace secret may expose the token under any of the
    # common names; vLLM/huggingface_hub want HF_TOKEN.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    t0 = time.monotonic()
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=tensor_parallel,
        enforce_eager=False,
    )
    load_s = time.monotonic() - t0
    weights.commit()  # persist freshly downloaded weights promptly

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t1 = time.monotonic()
    outputs = llm.chat(conversations, params)
    generate_s = time.monotonic() - t1

    rows = [
        {
            "text": out.outputs[0].text,
            "finish_reason": out.outputs[0].finish_reason,
            "completion_tokens": len(out.outputs[0].token_ids),
            "prompt_tokens": len(out.prompt_token_ids),
        }
        for out in outputs
    ]
    return {
        "rows": rows,
        "backend": {
            "engine": "vllm",
            "vllm_version": str(vllm.__version__),
            "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "max_model_len": MAX_MODEL_LEN,
            "chat_template": "model default via llm.chat, add_generation_prompt",
        },
        "timing": {"model_load_s": round(load_s, 1), "generate_s": round(generate_s, 1)},
    }


common = dict(
    image=image,
    volumes={"/models": weights},
    secrets=[hf_secret],
    timeout=3600,
    scaledown_window=60,
)


@app.function(gpu="A10G", **common)
def generate_a10g(model_id: str, conversations: list, max_tokens: int) -> dict:
    return _generate(model_id, conversations, max_tokens)


@app.function(gpu="A100-40GB", **common)
def generate_a100(model_id: str, conversations: list, max_tokens: int) -> dict:
    return _generate(model_id, conversations, max_tokens)


# 70B bf16 is ~140GB of weights: tensor-parallel over 4x A100-80GB.
@app.function(gpu="A100-80GB:4", **{**common, "timeout": 7200})
def generate_a100_80_x4(model_id: str, conversations: list, max_tokens: int) -> dict:
    return _generate(model_id, conversations, max_tokens, tensor_parallel=4)


@app.local_entrypoint()
def main(model: str = "1b", arm: str = "story", limit: int = 0):
    import hashlib
    import json
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from ftlib import REPO  # noqa: F401  (puts informalizing-etp on sys.path)
    from benchmark import bucket_of, wrap_prompt
    from checkform import PROMPT_PATH as STORY_TEMPLATE
    from checkform import build_prompt, grade

    LITERAL_TEMPLATE = REPO / "prompts" / "literal_prompt.md"
    assert arm in ("story", "literal"), f"unknown arm {arm!r}"
    model_id, gpu_tier = MODELS[model]
    template = STORY_TEMPLATE if arm == "story" else LITERAL_TEMPLATE

    rows = []
    for tier in ("normal", "hard", "extra_hard", "order5"):
        for line in (here / "eval_v1" / f"eval_{tier}.jsonl").read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]

    prompts = []
    for r in rows:
        base = build_prompt({"story": r[arm]}, template_path=template)
        prompts.append(wrap_prompt(base, "off", model_id, arm))
    conversations = [[{"role": "user", "content": p}] for p in prompts]

    fn = {
        "a10g": generate_a10g,
        "a100": generate_a100,
        "a100_80_x4": generate_a100_80_x4,
    }[gpu_tier]
    t0 = time.monotonic()
    result = fn.remote(model_id, conversations, MAX_TOKENS)
    wall_s = time.monotonic() - t0

    out_dir = here / "runs" / "base-v1" / f"{model}-{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_rows = []
    for r, prompt, gen in zip(rows, prompts, result["rows"]):
        verdict = grade(gen["text"], {"canonical_e": r["canonical_e"], "canonical_f": r["canonical_f"]})
        row = {
            "pair_id": r["problem_id"],
            "tier": r["tier"],
            "form": arm,
            "model": model_id,
            "regime": "off",
            "ops_total": r["ops_total"],
            "depth": r["max_depth"],
            "pair_hash": r["pair_hash"],
            "sent_prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:12],
            "response": gen["text"],
            "verdict": verdict,
            "finish_reason": gen["finish_reason"],
            "completion_tokens": gen["completion_tokens"],
            "prompt_tokens": gen["prompt_tokens"],
            "api_error": None,
        }
        row["bucket"] = bucket_of(row)
        results_rows.append(row)

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in results_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    tiers = sorted({row["tier"] for row in results_rows})
    buckets = ("exact", "correct-swapped", "correct-dualized", "wrong", "unparseable")
    summary = {}
    for tier in tiers:
        mine = [row for row in results_rows if row["tier"] == tier]
        counts = {b: sum(1 for row in mine if row["bucket"] == b) for b in buckets}
        correct = counts["exact"] + counts["correct-swapped"] + counts["correct-dualized"]
        summary[tier] = {
            "n": len(mine),
            **counts,
            "correct": correct,
            "correct_pct": round(100 * correct / len(mine), 1),
            "unparseable_pct": round(100 * counts["unparseable"] / len(mine), 1),
            "length_finishes": sum(1 for row in mine if row["finish_reason"] == "length"),
        }

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "3-base-eval",
        "model": model_id,
        "arm": arm,
        "regime": "off",
        "eval_version": "v1",
        "n": len(rows),
        "limit": limit or None,
        "sampling": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "greedy": True},
        "backend": result["backend"],
        "gpu_requested": {
            "a10g": "A10G",
            "a100": "A100-40GB",
            "a100_80_x4": "A100-80GB:4 (TP=4)",
        }[gpu_tier],
        "prompt_template": template.name,
        "prompt_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "regime_suffix": wrap_prompt("", "off", model_id, arm),
        "timing": {**result["timing"], "wall_s": round(wall_s, 1)},
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{model_id} · {arm} · n={len(rows)} · wall {wall_s:.0f}s "
          f"(load {result['timing']['model_load_s']}s, gen {result['timing']['generate_s']}s)")
    for tier, s in summary.items():
        print(f"  {tier:12s} n={s['n']:3d}  correct {s['correct_pct']:5.1f}%  "
              f"exact {s['exact']:3d}  swapped {s['correct-swapped']:3d}  "
              f"dual {s['correct-dualized']:2d}  wrong {s['wrong']:3d}  "
              f"unparseable {s['unparseable']:3d} ({s['unparseable_pct']}%)  "
              f"len-cap {s['length_finishes']}")
