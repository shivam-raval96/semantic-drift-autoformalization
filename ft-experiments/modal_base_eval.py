#!/usr/bin/env python3
"""Phase 3 base evals on Modal: vLLM batch generation + local grading.

Remote side (Modal): ONE GPU class (A10G), one image, one config for
every eval from here on — base and future FT-checkpoint evals alike
(decision 2026-07-24). vLLM engine, bf16, greedy (temperature 0, fixed
max_tokens), HF weights cached on a Volume. Guardrails: min_containers=0,
scaledown_window 60s, retries=0, 15-min hard timeout, max 1 container
per run. 70B is parked until an explicit go (it cannot fit A10G; its
config gets decided then). The container never sees repo code; it maps
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

IMAGE_TAG = "nvidia/cuda:12.8.1-devel-ubuntu22.04 + python3.12 + uv:vllm"

MODELS = {
    "1b": ("meta-llama/Llama-3.2-1B-Instruct", "a10g"),
    "8b": ("meta-llama/Llama-3.1-8B-Instruct", "a10g"),
    "70b": ("meta-llama/Llama-3.1-70B-Instruct", "parked"),
}
MAX_TOKENS = 4096  # repo convention for the reasoning-off regime
MAX_MODEL_LEN = 8192  # prompts are ~2k tokens; caps KV allocation


def _generate(
    model_id: str,
    conversations: list,
    max_tokens: int,
    tensor_parallel: int = 1,
    max_model_len: int = MAX_MODEL_LEN,
    two_stage: dict | None = None,
    lora: dict | None = None,
) -> dict:
    """Batch-generate; with two_stage, feed each raw stage-1 response
    verbatim into the stage-2 template (mechanical {story} replacement,
    exactly checkform.build_prompt's semantics) and grade-relevant
    output is the second pass. Engine loads once for both passes."""
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

    lora_request = None
    lora_kwargs = {}
    if lora is not None:
        from vllm.lora.request import LoRARequest

        lora_kwargs = {"enable_lora": True, "max_lora_rank": lora["rank"]}
        lora_request = LoRARequest("ft-adapter", 1, lora["path"])

    t0 = time.monotonic()
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel,
        enforce_eager=False,
        **lora_kwargs,
    )
    load_s = time.monotonic() - t0
    weights.commit()  # persist freshly downloaded weights promptly

    def run_pass(convs: list) -> list:
        outputs = llm.chat(
            convs,
            SamplingParams(temperature=0.0, max_tokens=max_tokens),
            lora_request=lora_request,
        )
        return [
            {
                "text": out.outputs[0].text,
                "finish_reason": out.outputs[0].finish_reason,
                "completion_tokens": len(out.outputs[0].token_ids),
                "prompt_tokens": len(out.prompt_token_ids),
            }
            for out in outputs
        ]

    t1 = time.monotonic()
    rows = run_pass(conversations)
    stage1_rows = None
    if two_stage is not None:
        stage1_rows = rows
        stage2_convs = [
            [{
                "role": "user",
                "content": two_stage["template"].replace("{story}", r["text"])
                + two_stage["suffix"],
            }]
            for r in stage1_rows
        ]
        rows = run_pass(stage2_convs)
    generate_s = time.monotonic() - t1

    return {
        "rows": rows,
        "stage1_rows": stage1_rows,
        "backend": {
            "engine": "vllm",
            "vllm_version": str(vllm.__version__),
            "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "max_model_len": max_model_len,
            "chat_template": "model default via llm.chat, add_generation_prompt",
        },
        "timing": {"model_load_s": round(load_s, 1), "generate_s": round(generate_s, 1)},
    }


GUARDRAILS = dict(
    min_containers=0,
    max_containers=1,
    scaledown_window=60,  # user cap: <= 120s
    retries=0,
    timeout=900,  # 15-minute hard cap per run
)


_fn_common = dict(
    gpu="A10G", image=image, volumes={"/models": weights}, secrets=[hf_secret]
)


@app.function(**_fn_common, **GUARDRAILS)
def generate_a10g(
    model_id: str, conversations: list, max_tokens: int, lora: dict | None = None
) -> dict:
    return _generate(model_id, conversations, max_tokens, lora=lora)


# Two-stage runs two full generation passes (8B: ~7 min each), which
# cannot fit the 15-min single-pass cap; 30-min timeout here, disclosed
# and stamped in run_meta. Stage-2 prompts embed a whole stage-1
# response, so the sequence cap is raised to 12288.
TWO_STAGE_MAX_MODEL_LEN = 12288
TWO_STAGE_GUARDRAILS = {**GUARDRAILS, "timeout": 1800}


@app.function(**_fn_common, **TWO_STAGE_GUARDRAILS)
def generate_a10g_two_stage(
    model_id: str,
    conversations: list,
    max_tokens: int,
    two_stage: dict,
    lora: dict | None = None,
) -> dict:
    return _generate(
        model_id,
        conversations,
        max_tokens,
        max_model_len=TWO_STAGE_MAX_MODEL_LEN,
        two_stage=two_stage,
        lora=lora,
    )


@app.local_entrypoint()
def main(
    model: str = "1b",
    arm: str = "story",
    limit: int = 0,
    adapter: str = "",
    adapter_rank: int = 16,
    out_tag: str = "base-v1",
):
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
    ABSTRACT_TEMPLATE = REPO / "prompts" / "abstract_prompt.md"
    assert arm in ("story", "literal", "two-stage"), f"unknown arm {arm!r}"
    model_id, gpu_tier = MODELS[model]
    # exp-07 plumbing: two-stage sends the STORY under abstract_prompt.md
    # (stage 1); stage 2 is literal_prompt.md filled with the raw stage-1
    # response, wrapped with the literal two-lines suffix.
    template = {
        "story": STORY_TEMPLATE, "literal": LITERAL_TEMPLATE, "two-stage": ABSTRACT_TEMPLATE,
    }[arm]
    wrap_form = "abstract" if arm == "two-stage" else arm
    source_field = "literal" if arm == "literal" else "story"

    rows = []
    for tier in ("normal", "hard", "extra_hard", "order5"):
        for line in (here / "eval_v1" / f"eval_{tier}.jsonl").read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]

    prompts = []
    for r in rows:
        base = build_prompt({"story": r[source_field]}, template_path=template)
        prompts.append(wrap_prompt(base, "off", model_id, wrap_form))
    conversations = [[{"role": "user", "content": p}] for p in prompts]

    if gpu_tier == "parked":
        raise SystemExit(f"{model_id} is parked pending an explicit go — not runnable")
    stage2_template_text = LITERAL_TEMPLATE.read_text(encoding="utf-8")
    stage2_suffix = wrap_prompt("", "off", model_id, "literal")
    lora = {"path": adapter, "rank": adapter_rank} if adapter else None
    t0 = time.monotonic()
    if arm == "two-stage":
        result = generate_a10g_two_stage.remote(
            model_id, conversations, MAX_TOKENS,
            {"template": stage2_template_text, "suffix": stage2_suffix},
            lora=lora,
        )
    else:
        result = generate_a10g.remote(model_id, conversations, MAX_TOKENS, lora=lora)
    wall_s = time.monotonic() - t0

    out_dir = here / "runs" / out_tag / f"{model}-{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_rows = []
    stage1_rows = result.get("stage1_rows") or [None] * len(rows)
    for r, prompt, gen, s1 in zip(rows, prompts, result["rows"], stage1_rows):
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
        if s1 is not None:
            stage2_prompt = stage2_template_text.replace("{story}", s1["text"]) + stage2_suffix
            row["sent_prompt_hash"] = hashlib.sha256(stage2_prompt.encode()).hexdigest()[:12]
            row.update(
                stage1_sent_prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:12],
                stage1_response=s1["text"],
                stage1_finish_reason=s1["finish_reason"],
                stage1_completion_tokens=s1["completion_tokens"],
            )
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
        "gpu_requested": "A10G",
        "adapter": ({"path": adapter, "rank": adapter_rank} if adapter else None),
        "image": IMAGE_TAG,
        "guardrails": TWO_STAGE_GUARDRAILS if arm == "two-stage" else GUARDRAILS,
        "max_model_len": TWO_STAGE_MAX_MODEL_LEN if arm == "two-stage" else MAX_MODEL_LEN,
        "prompt_template": template.name,
        "prompt_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "regime_suffix": wrap_prompt("", "off", model_id, wrap_form),
        **(
            {
                "stage2_template": LITERAL_TEMPLATE.name,
                "stage2_template_sha256": hashlib.sha256(
                    LITERAL_TEMPLATE.read_bytes()
                ).hexdigest(),
                "stage2_regime_suffix": stage2_suffix,
            }
            if arm == "two-stage"
            else {}
        ),
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
