#!/usr/bin/env python3
"""Stage-1 recognition eval, Lambda-native (local vLLM on the box's GPU).

Same held-out rows, grader, and protocol as eval/stage1_eval.py, without
Modal: the vLLM engine runs here. Base vs FT is a per-invocation choice
(--adapter); the base run is the control that opaque labels are not
guessable pre-training.

    python3 eval/stage1_eval_lambda.py                                   # base control
    python3 eval/stage1_eval_lambda.py --adapter ../checkpoints/stage1-8b-s0/final
    python3 eval/stage1_eval_lambda.py --dry-run                         # local grader self-test

--adapter takes a LOCAL adapter folder. On a fresh box, first fetch it:
    hf download <you>/mars-v-stage1 --include 'stage1-8b-s0/final/*' --local-dir ../checkpoints
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 32
MAX_MODEL_LEN = 4096


def load_stage1lib():
    path = ROOT / "data-gen" / "stage1lib.py"
    spec = importlib.util.spec_from_file_location("stage1lib", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage1lib"] = module
    spec.loader.exec_module(module)
    return module


def load_rows() -> list:
    return [json.loads(l) for l in (ROOT / "stage1" / "eval.jsonl").read_text().splitlines() if l.strip()]


def run_dry(s1) -> None:
    rows = load_rows()
    assert rows, "no eval rows"
    for row in rows:
        _, correct, answered = s1.grade_row(row, row["completion"])
        assert correct and answered, ("gold not correct", row)
        wrong = "RG-1" if row["completion"] != "RG-1" else "RG-2"
        if row["task"] == "validate":
            wrong = "No" if row["completion"] == "Yes" else "Yes"
        _, correct_w, _ = s1.grade_row(row, wrong)
        assert not correct_w, ("wrong graded correct", row)
    assert s1.extract_identify("hmm RG-1 ... actually RG-3") == "RG-3"
    assert s1.extract_validate("Let me see. Yes, wait, No.") == "No"
    assert s1.extract_identify("I cannot tell") is None
    print(f"= dry run: {len(rows)} eval rows; gold correct, wrong wrong, messy OK")
    print("DRY RUN OK")


def generate(model_id: str, conversations: list, adapter: str, adapter_rank: int,
             chat_kwargs: dict | None) -> dict:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    lora_request, lora_kwargs = None, {}
    if adapter:
        from vllm.lora.request import LoRARequest

        lora_kwargs = {"enable_lora": True, "max_lora_rank": adapter_rank}
        lora_request = LoRARequest("stage1-adapter", 1, adapter)

    t0 = time.monotonic()
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              tensor_parallel_size=1, enforce_eager=False, **lora_kwargs)
    load_s = time.monotonic() - t0

    t1 = time.monotonic()
    outputs = llm.chat(
        conversations,
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
        lora_request=lora_request,
        **({"chat_template_kwargs": chat_kwargs} if chat_kwargs else {}),
    )
    gen_rows = [
        {"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
         "completion_tokens": len(o.outputs[0].token_ids)}
        for o in outputs
    ]
    return {
        "rows": gen_rows,
        "backend": {"engine": "vllm", "vllm_version": str(vllm.__version__),
                    "torch_version": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16"},
        "timing": {"model_load_s": round(load_s, 1),
                   "generate_s": round(time.monotonic() - t1, 1)},
    }


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Lambda-native stage-1 recognition eval.")
    cli.add_argument("--model-id", default=DEFAULT_MODEL)
    cli.add_argument("--adapter", default="", help="local adapter folder; empty = base control")
    cli.add_argument("--adapter-rank", type=int, default=16)
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args(argv)

    s1 = load_stage1lib()
    if args.dry_run:
        run_dry(s1)
        return 0
    run_dry(s1)  # never spend GPU on a grader failing its own controls

    if args.adapter and not Path(args.adapter).is_dir():
        raise SystemExit(f"--adapter {args.adapter!r} is not a local folder; download it first "
                         "(see the header) or pass a local path")

    rows = load_rows()
    conversations = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    chat_kwargs = {"enable_thinking": False} if "qwen3" in args.model_id.lower() else None
    print(f"= stage1 eval (lambda): model={args.model_id} "
          f"adapter={args.adapter or 'NONE (base control)'} n={len(rows)}")

    t0 = time.monotonic()
    result = generate(args.model_id, conversations, args.adapter, args.adapter_rank, chat_kwargs)
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
        "platform": "lambda",
        "model": args.model_id,
        "adapter": args.adapter or None,
        "is_base_control": args.adapter == "",
        "n": len(rows),
        "sampling": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "greedy": True},
        "backend": result["backend"],
        "timing": {**result["timing"], "wall_s": round(wall_s, 1)},
        "summary": summary,
        "rows": graded,
    }
    out_dir = ROOT / "runs" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "ft" if args.adapter else "base"
    out_path = out_dir / f"eval-lambda-{tag}.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{args.model_id} · stage1 · n={len(rows)} · wall {wall_s:.0f}s -> {out_path}")
    print(f"  overall  acc {summary['overall']['accuracy_pct']:5.1f}%  "
          f"answered {summary['overall']['answered_pct']:5.1f}%")
    ident = summary["identify_by_label"]
    print(f"  identify acc {summary['identify']['accuracy_pct']:5.1f}%  "
          f"(RG-1 {ident.get('RG-1', {}).get('accuracy_pct')}, "
          f"RG-2 {ident.get('RG-2', {}).get('accuracy_pct')}, "
          f"RG-3 {ident.get('RG-3', {}).get('accuracy_pct')})")
    print(f"  validate acc {summary['validate']['accuracy_pct']:5.1f}%  "
          f"(yes {summary['validate_by_polarity']['yes']['accuracy_pct']}, "
          f"no {summary['validate_by_polarity']['no']['accuracy_pct']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
