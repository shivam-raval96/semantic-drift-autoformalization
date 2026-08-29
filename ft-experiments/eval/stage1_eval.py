#!/usr/bin/env python3
"""Stage-1 recognition eval: does the model recognize grammars by label?

Runs the held-out stage1/eval.jsonl rows (law-disjoint from training) and
grades identification (statement -> RG-N) and validation (statement + label
-> Yes/No). Base vs FT is a per-invocation choice (--adapter); the base run
is the control that opaque labels are not guessable pre-training.

    modal run eval/stage1_eval.py --model 8b                 # base control
    modal run eval/stage1_eval.py --model 8b --adapter /models/checkpoints/stage1-8b-s0/final
    python3 eval/stage1_eval.py --dry-run                    # local grader self-test

Greedy, no-think, temperature 0 — same protocol family as modal_eval.py.
"""

import importlib.util
import json
import re
from pathlib import Path

import modal

app = modal.App("harsh-stage1-eval")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

IMAGE_TAG = "nvidia/cuda:12.8.1-devel-ubuntu22.04 + python3.12 + uv:vllm"
MAX_TOKENS = 32       # answers are a label or Yes/No; also caps runaway fluency
MAX_MODEL_LEN = 4096


def _generate(model_id, conversations, max_tokens, lora=None, chat_kwargs=None):
    """Batch greedy generation — mirrors format_control._generate."""
    import os
    import time

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
        lora_request = LoRARequest("stage1-adapter", 1, lora["path"])

    t0 = time.monotonic()
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=1,
        enforce_eager=False,
        **lora_kwargs,
    )
    load_s = time.monotonic() - t0
    weights.commit()

    t1 = time.monotonic()
    outputs = llm.chat(
        conversations,
        SamplingParams(temperature=0.0, max_tokens=max_tokens),
        lora_request=lora_request,
        **({"chat_template_kwargs": chat_kwargs} if chat_kwargs else {}),
    )
    rows = [
        {
            "text": out.outputs[0].text,
            "finish_reason": out.outputs[0].finish_reason,
            "completion_tokens": len(out.outputs[0].token_ids),
        }
        for out in outputs
    ]
    generate_s = time.monotonic() - t1

    return {
        "rows": rows,
        "backend": {
            "engine": "vllm",
            "vllm_version": str(vllm.__version__),
            "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "max_model_len": MAX_MODEL_LEN,
        },
        "timing": {"model_load_s": round(load_s, 1), "generate_s": round(generate_s, 1)},
    }


GUARDRAILS = dict(min_containers=0, max_containers=1, scaledown_window=60,
                  retries=0, timeout=1800)
_fn_common = dict(image=image, volumes={"/models": weights}, secrets=[hf_secret],
                  max_inputs=1)


@app.function(gpu="A10G", **_fn_common, **GUARDRAILS)
def generate_a10g(model_id, conversations, max_tokens, lora=None, chat_kwargs=None):
    return _generate(model_id, conversations, max_tokens, lora=lora, chat_kwargs=chat_kwargs)


@app.function(gpu="A100-80GB", **_fn_common, **GUARDRAILS)
def generate_a100(model_id, conversations, max_tokens, lora=None, chat_kwargs=None):
    return _generate(model_id, conversations, max_tokens, lora=lora, chat_kwargs=chat_kwargs)


# --------------------------------------------------------------- grading
# The grader lives in data-gen/stage1lib.py (shared with the builder and the
# Lambda eval), so extraction + scoring can never drift between paths.


def _load_stage1lib(root: Path):
    import sys

    path = root / "data-gen" / "stage1lib.py"
    spec = importlib.util.spec_from_file_location("stage1lib", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage1lib"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------- local dry-run gate


def run_dry(root: Path):
    """GPU-free grader self-test: gold answers grade correct, wrong answers
    grade wrong, and the extractors handle messy responses."""
    s1 = _load_stage1lib(root)
    rows = [json.loads(l) for l in (root / "stage1" / "eval.jsonl").read_text().splitlines() if l.strip()]
    assert rows, "no eval rows"

    for row in rows:
        # gold response grades correct
        _, correct, answered = s1.grade_row(row, row["completion"])
        assert correct and answered, ("gold not correct", row)
        # a clearly wrong response grades incorrect
        wrong = "RG-1" if row["completion"] != "RG-1" else "RG-2"
        if row["task"] == "validate":
            wrong = "No" if row["completion"] == "Yes" else "Yes"
        _, correct_w, _ = s1.grade_row(row, wrong)
        assert not correct_w, ("wrong graded correct", row)

    # messy-response extraction (models think aloud, last answer wins)
    assert s1.extract_identify("hmm RG-1 ... actually RG-3") == "RG-3"
    assert s1.extract_validate("Let me see. Yes, wait, No.") == "No"
    assert s1.extract_identify("I cannot tell") is None       # no-answer state
    print(f"= dry run: {len(rows)} eval rows; gold grades correct, wrong grades wrong, "
          "messy extraction OK")
    print("DRY RUN OK")


# ------------------------------------------------------------ entrypoint


@app.local_entrypoint()
def main(model: str = "8b", adapter: str = "", adapter_rank: int = 16, dry_run: bool = False):
    import time
    from datetime import datetime, timezone

    here = Path(__file__).resolve().parent
    root = here.parent
    if dry_run:
        run_dry(root)
        return

    spec = importlib.util.spec_from_file_location("ft_root_config", root / "config.py")
    ftc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ftc)
    assert IMAGE_TAG == ftc.MODAL["image_tag"], "image drifted from config.MODAL"

    entry = ftc.MODELS[model]
    if entry.get("parked"):
        raise SystemExit(f"{entry['hf_id']} is parked — not runnable")
    model_id = entry["hf_id"]
    s1 = _load_stage1lib(root)

    run_dry(root)  # never spend GPU on a grader failing its own controls

    rows = [json.loads(l) for l in (root / "stage1" / "eval.jsonl").read_text().splitlines() if l.strip()]
    conversations = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    lora = {"path": adapter, "rank": adapter_rank} if adapter else None
    chat_kwargs = {"enable_thinking": False} if "qwen3" in model_id.lower() else None
    fn = generate_a100 if entry["gpu"] == "A100-80GB" else generate_a10g
    print(f"= stage1 eval: model={model_id} adapter={adapter or 'NONE (base control)'} "
          f"n={len(rows)}")

    t0 = time.monotonic()
    result = fn.remote(model_id, conversations, MAX_TOKENS, lora=lora, chat_kwargs=chat_kwargs)
    wall_s = time.monotonic() - t0

    graded = []
    for row, gen in zip(rows, result["rows"]):
        predicted, correct, answered = s1.grade_row(row, gen["text"])
        graded.append({
            "task": row["task"], "grammar": row["grammar"], "label": row["label"],
            "polarity": row["polarity"], "corruption": row["corruption"],
            "gold": row["completion"], "predicted": predicted,
            "correct": correct, "answered": answered,
            "response": gen["text"], "completion_tokens": gen["completion_tokens"],
        })
    summary = s1.summarize(graded)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": "stage-1 recognition eval (identify + validate), held-out laws",
        "model": model_id, "model_key": model,
        "adapter": ({"path": adapter, "rank": adapter_rank} if adapter else None),
        "is_base_control": adapter == "",
        "n": len(rows),
        "sampling": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "greedy": True},
        "chat_template_kwargs": chat_kwargs,
        "backend": result["backend"],
        "timing": {**result["timing"], "wall_s": round(wall_s, 1)},
        "summary": summary,
        "rows": graded,
    }
    out_dir = ftc.PATHS["runs"] / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval-{model}{'-ft' if adapter else '-base'}.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{model_id} · stage1 · n={len(rows)} · wall {wall_s:.0f}s -> {out_path}")
    print(f"  overall  acc {summary['overall']['accuracy_pct']:5.1f}%  "
          f"answered {summary['overall']['answered_pct']:5.1f}%")
    print(f"  identify acc {summary['identify']['accuracy_pct']:5.1f}%  "
          f"(by label {{'RG-1': {summary['identify_by_label'].get('RG-1', {}).get('accuracy_pct')}, "
          f"'RG-2': {summary['identify_by_label'].get('RG-2', {}).get('accuracy_pct')}, "
          f"'RG-3': {summary['identify_by_label'].get('RG-3', {}).get('accuracy_pct')}}})")
    print(f"  validate acc {summary['validate']['accuracy_pct']:5.1f}%  "
          f"(yes {summary['validate_by_polarity']['yes']['accuracy_pct']}, "
          f"no {summary['validate_by_polarity']['no']['accuracy_pct']})")


if __name__ == "__main__":
    import sys

    if "--dry-run" in sys.argv:
        run_dry(Path(__file__).resolve().parent.parent)
    else:
        raise SystemExit("use `modal run eval/stage1_eval.py --model 8b [--adapter ...]`, "
                         "or `python3 eval/stage1_eval.py --dry-run` for the local grader check")