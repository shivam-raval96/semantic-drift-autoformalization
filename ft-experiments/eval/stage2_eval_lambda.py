#!/usr/bin/env python3
"""Stage-2 cross-grammar translation eval, Lambda-native (local vLLM).

Runs one model over trans_eval_v1 and grades its story->grammar output for
one or more rigid grammars. The engine loads once and every requested
grammar reuses it, so a full RG-1/2/3/4 sweep is one model load. Grading
goes through the shared transcode seam (grammars.grade_b), giving the same
three-way verdict as the benchmark: correct / wrong / unparseable. Results
are sliced by tier and by theme (tea is the held-out theme).

    # exp-1: F0 (stage-1 familiarity) translating into RG-1/2/3
    python3 eval/stage2_eval_lambda.py --run-name f0 \
        --adapter ../checkpoints/stage1-8b-s0/final --grammars rg1,rg2,rg3

    # T1: in-grammar (RG-1) + held-out grammar (RG-4)
    python3 eval/stage2_eval_lambda.py --run-name t1-8b \
        --adapter ../checkpoints/t1-8b/final --grammars rg1,rg4

    # base control (no adapter)
    python3 eval/stage2_eval_lambda.py --run-name base --grammars rg1

    python3 eval/stage2_eval_lambda.py --dry-run    # grader self-test, no GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_GEN = ROOT / "data-gen"
if str(DATA_GEN) not in sys.path:
    sys.path.insert(0, str(DATA_GEN))

import stage2lib as s2  # noqa: E402  (adds repo prompt fns + grammars)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 512
MAX_MODEL_LEN = 4096
TIERS = ("normal", "hard", "extra_hard", "order5")


def load_rows() -> list:
    path = s2.ftc.PATHS["trans_eval_v1"] / "eval.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_grammars(spec: str) -> list:
    return [s2.key_for(tok) for tok in spec.split(",") if tok.strip()]


# ------------------------------------------------------------- grading

def _acc(subset: list) -> dict:
    n = len(subset)
    correct = sum(1 for r in subset if r["status"] == "correct")
    wrong = sum(1 for r in subset if r["status"] == "wrong")
    answered = sum(1 for r in subset if r["status"] != "unparseable")
    return {
        "n": n,
        "correct_pct": round(100 * correct / n, 1) if n else 0.0,
        "wrong_pct": round(100 * wrong / n, 1) if n else 0.0,
        "unparseable_pct": round(100 * (n - answered) / n, 1) if n else 0.0,
        "answered_pct": round(100 * answered / n, 1) if n else 0.0,
    }


def summarize(graded: list) -> dict:
    return {
        "overall": _acc(graded),
        "by_tier": {t: _acc([r for r in graded if r["tier"] == t]) for t in TIERS},
        "by_theme": {t: _acc([r for r in graded if r["theme"] == t])
                     for t in sorted({r["theme"] for r in graded})},
    }


def grade_all(rows: list, key: str, texts: list) -> list:
    graded = []
    for row, text in zip(rows, texts):
        verdict = s2.grade(text["text"], key, row["canonical_e"], row["canonical_f"])
        graded.append({
            "problem_id": row["problem_id"], "tier": row["tier"], "theme": row["theme"],
            "status": verdict["status"], "transform": verdict.get("transform"),
            "response": text["text"], "completion_tokens": text["completion_tokens"],
        })
    return graded


# ------------------------------------------------------------- dry run

def run_dry(rows: list, grammars: list) -> None:
    assert rows, "no eval rows"
    ref_field = {"a": "ref_rg1", "b_near": "ref_rg2", "b_far": "ref_rg3", "sexpr": "ref_rg4"}
    for key in grammars:
        graded = grade_all(
            rows, key,
            [{"text": r[ref_field[key]], "completion_tokens": 0} for r in rows])
        summ = summarize(graded)["overall"]
        assert summ["correct_pct"] == 100.0, (key, "gold refs not all correct", summ)
        # a broken reference must be unparseable, not silently correct
        broken = rows[0][ref_field[key]]
        broken = broken[:broken.rfind(")")] + broken[broken.rfind(")") + 1:] \
            if ")" in broken else broken.replace("=", "")
        bad = s2.grade(broken, key, rows[0]["canonical_e"], rows[0]["canonical_f"])
        assert bad["status"] == "unparseable", (key, "broken ref not unparseable", bad)
    print(f"= dry run: {len(rows)} rows; {[s2.GRAMMAR_TO_LABEL[k] for k in grammars]} "
          "gold refs 100% correct, broken refs unparseable")
    print("DRY RUN OK")


# ------------------------------------------------------------- generate

def load_engine(model_id: str, adapter: str, adapter_rank: int):
    import torch
    import vllm
    from vllm import LLM

    lora_kwargs = {}
    if adapter:
        lora_kwargs = {"enable_lora": True, "max_lora_rank": adapter_rank}
    t0 = time.monotonic()
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              tensor_parallel_size=1, enforce_eager=False, **lora_kwargs)
    backend = {"engine": "vllm", "vllm_version": str(vllm.__version__),
               "torch_version": str(torch.__version__),
               "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16"}
    return llm, backend, round(time.monotonic() - t0, 1)


def generate_grammar(llm, rows, key, model_id, adapter, chat_kwargs):
    from vllm import SamplingParams

    lora_request = None
    if adapter:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("stage2-adapter", 1, adapter)
    conversations = [
        [{"role": "user", "content": s2.build_translation_prompt(r["story"], key, model_id)}]
        for r in rows
    ]
    t0 = time.monotonic()
    outputs = llm.chat(
        conversations,
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
        lora_request=lora_request,
        **({"chat_template_kwargs": chat_kwargs} if chat_kwargs else {}),
    )
    gen = [{"text": o.outputs[0].text, "completion_tokens": len(o.outputs[0].token_ids)}
           for o in outputs]
    return gen, round(time.monotonic() - t0, 1)


# ------------------------------------------------------------- entry

def derive_run_name(adapter: str) -> str:
    if not adapter:
        return "base"
    parts = Path(adapter).parts
    # .../<run>/final -> <run>; .../<run> -> <run>
    return parts[-2] if parts and parts[-1] in ("final", "") else parts[-1]


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Lambda-native stage-2 translation eval.")
    cli.add_argument("--model-id", default=DEFAULT_MODEL)
    cli.add_argument("--adapter", default="", help="local adapter folder; empty = base control")
    cli.add_argument("--adapter-rank", type=int, default=16)
    cli.add_argument("--grammars", default="rg1,rg2,rg3,rg4", help="comma list, e.g. rg1,rg4")
    cli.add_argument("--run-name", default="", help="output tag; default derived from --adapter")
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args(argv)

    s2.assert_frozen_templates()
    grammars = parse_grammars(args.grammars)
    rows = load_rows()

    if args.dry_run:
        run_dry(rows, grammars)
        return 0
    run_dry(rows, grammars)  # never spend GPU on a grader failing its own controls

    if args.adapter and not Path(args.adapter).is_dir():
        raise SystemExit(f"--adapter {args.adapter!r} is not a local folder; download it first")

    run_name = args.run_name or derive_run_name(args.adapter)
    chat_kwargs = {"enable_thinking": False} if "qwen3" in args.model_id.lower() else None
    labels = [s2.GRAMMAR_TO_LABEL[k] for k in grammars]
    print(f"= stage2 eval (lambda): model={args.model_id} "
          f"adapter={args.adapter or 'NONE (base control)'} run={run_name} "
          f"grammars={labels} n={len(rows)}")

    wall0 = time.monotonic()
    llm, backend, load_s = load_engine(args.model_id, args.adapter, args.adapter_rank)
    print(f"= engine loaded in {load_s}s")

    out_dir = s2.ftc.PATHS["runs"] / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = {}
    for key, label in zip(grammars, labels):
        gen, gen_s = generate_grammar(llm, rows, key, args.model_id, args.adapter, chat_kwargs)
        graded = grade_all(rows, key, gen)
        summary = summarize(graded)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": "stage-2 translation eval (story -> rigid grammar)",
            "platform": "lambda",
            "model": args.model_id, "adapter": args.adapter or None,
            "is_base_control": args.adapter == "",
            "run_name": run_name, "grammar": key, "label": label,
            "template_sha": s2.template_sha(key),
            "n": len(rows),
            "sampling": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "greedy": True},
            "backend": backend,
            "timing": {"model_load_s": load_s, "generate_s": gen_s},
            "summary": summary, "rows": graded,
        }
        out_path = out_dir / f"eval-{run_name}-{label}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        matrix[label] = summary
        print(f"  {label}: correct {summary['overall']['correct_pct']:5.1f}%  "
              f"wrong {summary['overall']['wrong_pct']:5.1f}%  "
              f"unparseable {summary['overall']['unparseable_pct']:5.1f}%  "
              f"(gen {gen_s}s) -> {out_path.name}")

    print(f"\n{args.model_id} · run {run_name} · n={len(rows)} · "
          f"wall {time.monotonic() - wall0:.0f}s")
    for label in labels:
        ov = matrix[label]["overall"]
        tea = matrix[label]["by_theme"].get("tea", {})
        print(f"  {label:5s} correct {ov['correct_pct']:5.1f}%  "
              f"tea correct {tea.get('correct_pct', 'n/a')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
