#!/usr/bin/env python3
"""Experiment 13 runner: truth judgment of ETP implications across arms.

For each sampled (E, F) pair — stratified by total operation count with
each bin split exactly 50/50 between Lean-proved-true and proved-false
implications (truthdata.py) — render the implication in the chosen arm
(--form story | literal | symbolic), fill the arm's prove_*_prompt.md
template, send it to each model via OpenRouter, and grade the final
"ANSWER: True|False" line against the ground truth with proveform.grade.

The sampler never consults the form, so all arms of one seed cover the
identical pair set. Equation labels (E387) exist only in metadata; the
runner hard-fails if any prompt leaks a label or an ETP identifier.

Artifacts match the benchmark.py run-dir schema (run_meta.json,
samples.jsonl, results.jsonl, summary.json/md); rerunning the same
--out-dir resumes, retrying only api-error rows. --dry-run grades
synthesized correct answers offline and must score 100%.

Run from the repo's informalizing-etp/ directory:

    set -a; source ../.env; set +a
    python3 experiments/13-implication-truth/run_experiment.py \
        --form story --per-bin 10 --bins 2:8 --seed 0 --reasoning on
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark import (  # noqa: E402
    build_reasoning_payload,
    call_openrouter,
    load_completed,
    load_equations,
    validate_models,
)
from checkform import build_prompt  # noqa: E402
from genform import parse_bins  # noqa: E402
from literalform import render_description  # noqa: E402
from proveform import BUCKETS, PROMPT_PATHS, grade  # noqa: E402
from storyform import render_story  # noqa: E402
from symbolform import render_symbolic  # noqa: E402
from truthdata import ensure_matrix, sample_truth_balanced, truth_availability  # noqa: E402

RENDERERS = {
    "story": render_story,
    "literal": render_description,
    "symbolic": render_symbolic,
}

# Three open-weight models and one lightweight closed model.
DEFAULT_MODELS = (
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3-32b",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-5-mini",
)

EXPERIMENT_DIR = Path(__file__).resolve().parent

# Uniform per-regime wrappers, one wording for every arm (the task is
# identical; only the input format differs). benchmark.py's wrappers
# talk about "the two required lines" and cannot be reused here.
REGIME_PREFIX = {
    "on": (
        "Work through the problem step by step first — decide whether the "
        "questioned regularity is forced in every structure satisfying the "
        "assumption or can fail in one — and only then finish with the "
        "single required ANSWER line.\n\n"
    ),
}
REGIME_SUFFIX = {
    "off": (
        "\n\nRespond with only the single required ANSWER line, and no "
        "other text before it."
    ),
}

# Nothing tied to the ETP may reach a model: labels invite recall of the
# published implication database rather than reasoning about the text.
_LABEL_RE = re.compile(r"\bE\d+\b")
BANNED_IN_PROMPTS = ("magma", "Magma", "Lean", "ETP", "Equational")


def wrap_prompt(prompt: str, regime: Optional[str], model: str = "") -> str:
    text = REGIME_PREFIX.get(regime, "") + prompt + REGIME_SUFFIX.get(regime, "")
    # Qwen3's vendor-documented soft switch, as in benchmark.wrap_prompt.
    if regime == "off" and model.startswith("qwen/qwen3"):
        text += "\n/no_think"
    return text


def check_no_leakage(prompt: str, pair_id: str) -> None:
    if _LABEL_RE.search(prompt):
        raise SystemExit(f"{pair_id}: equation label leaked into the prompt")
    for word in BANNED_IN_PROMPTS:
        if word in prompt:
            raise SystemExit(f"{pair_id}: banned word {word!r} in the prompt")


def make_sample(equations: List[str], pair: dict, form: str, template_path: Path) -> dict:
    """Render one truth-labeled pair in the given form and build its prompt."""
    e_num, f_num = pair["e_num"], pair["f_num"]
    story, metadata = RENDERERS[form](equations[e_num - 1], equations[f_num - 1])
    metadata["label_e"] = f"E{e_num}"
    metadata["label_f"] = f"E{f_num}"
    metadata["truth"] = pair["truth"]
    metadata["status"] = pair["status"]
    prompt = build_prompt({"story": story, "metadata": metadata}, template_path=template_path)
    pair_id = f"E{e_num}-E{f_num}"
    check_no_leakage(prompt, pair_id)
    return {
        "pair_id": pair_id,
        "form": form,
        "story": story,
        "metadata": metadata,
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
        "truth": pair["truth"],
        "status": pair["status"],
        "ops_e": pair["ops_e"],
        "ops_f": pair["ops_f"],
        "ops_total": pair["ops_total"],
        "depth": pair["depth"],
    }


def bucket_of(row: dict) -> str:
    if row["response"] is None and row["api_error"] is not None:
        return "api-error"
    return row["verdict"]["status"]


def run_one(sample: dict, model: str, caller, regime: Optional[str]) -> dict:
    sent_prompt = wrap_prompt(sample["prompt"], regime, model)
    call = caller(model, sent_prompt, sample)
    response = call["content"]
    if response is not None:
        verdict = grade(response, sample["truth"])
    elif call["error"] is None:
        # The call succeeded but the model produced no visible content —
        # in practice, the whole token budget went to reasoning
        # (finish_reason "length") or the provider returned an empty
        # message. That is the model failing to answer, not an
        # infrastructure error: grade it unparseable so it stays in the
        # denominator and is never silently retried away.
        verdict = {
            "status": "unparseable",
            "answer": None,
            "error": f"empty response (finish_reason={call.get('finish_reason')})",
        }
    else:
        verdict = None
    row = {
        "pair_id": sample["pair_id"],
        "form": sample["form"],
        "label_e": sample["metadata"]["label_e"],
        "label_f": sample["metadata"]["label_f"],
        "theme": sample["metadata"]["theme"],
        "model": model,
        "regime": regime,
        "truth": sample["truth"],
        "status": sample["status"],
        "ops_e": sample["ops_e"],
        "ops_f": sample["ops_f"],
        "ops_total": sample["ops_total"],
        "depth": sample["depth"],
        "prompt_hash": sample["prompt_hash"],
        "sent_prompt_hash": hashlib.sha256(sent_prompt.encode("utf-8")).hexdigest()[:12],
        "response": response,
        "answer": verdict["answer"] if verdict else None,
        "verdict": verdict,
        "api_error": call["error"],
        "usage": call["usage"],
        "latency_s": call["latency_s"],
        "routed_model": call.get("routed_model"),
        "provider": call.get("provider"),
        "finish_reason": call.get("finish_reason"),
        "reasoning_tokens": call.get("reasoning_tokens"),
    }
    row["bucket"] = bucket_of(row)
    return row


# ------------------------------------------------------------ Aggregation


def _rate(part: int, whole: int) -> Optional[float]:
    return round(part / whole, 4) if whole else None


def aggregate(rows: List[dict], models: List[str]) -> dict:
    """Per-model buckets plus the truth-task vitals.

    At a 50/50 truth balance a constant answerer scores 50%, so
    accuracy alone is not evidence of judgment: true_accuracy /
    false_accuracy (per-class recall) and answer_true_rate (bias
    detector) are first-class. Accuracy is also split by proof kind
    (explicit vs implicit) and by total-ops bin.
    """
    summary: Dict[str, dict] = {}
    for model in models:
        mine = [row for row in rows if row["model"] == model]
        counts = {bucket: 0 for bucket in BUCKETS}
        for row in mine:
            counts[row["bucket"]] += 1
        graded = [row for row in mine if row["bucket"] != "api-error"]
        answered = [row for row in graded if row["answer"] is not None]
        trues = [row for row in graded if row["truth"]]
        falses = [row for row in graded if not row["truth"]]
        by_kind = {}
        for kind in ("explicit", "implicit"):
            kin = [row for row in graded if row["status"].startswith(kind)]
            by_kind[kind] = _rate(
                sum(r["bucket"] == "correct" for r in kin), len(kin)
            )
        by_bin = {}
        for row in graded:
            by_bin.setdefault(row["ops_total"], []).append(row)
        lengths = [len(row["response"]) for row in mine if row["response"]]
        reasoned = [row["reasoning_tokens"] for row in mine if row.get("reasoning_tokens")]
        summary[model] = {
            "counts": counts,
            "graded": len(graded),
            "accuracy": _rate(counts["correct"], len(graded)),
            "true_accuracy": _rate(
                sum(r["bucket"] == "correct" for r in trues), len(trues)
            ),
            "false_accuracy": _rate(
                sum(r["bucket"] == "correct" for r in falses), len(falses)
            ),
            "answer_true_rate": _rate(
                sum(r["answer"] is True for r in answered), len(answered)
            ),
            "accuracy_by_proof_kind": by_kind,
            "accuracy_by_ops_total": {
                b: _rate(sum(r["bucket"] == "correct" for r in rs), len(rs))
                for b, rs in sorted(by_bin.items())
            },
            "median_response_len": int(statistics.median(lengths)) if lengths else None,
            "rows_with_reasoning": len(reasoned),
            "median_reasoning_tokens": int(statistics.median(reasoned)) if reasoned else 0,
        }
    return summary


def _pct(rate: Optional[float]) -> str:
    return f"{100 * rate:.1f}" if rate is not None else "-"


def summary_table(summary: dict) -> str:
    headers = [
        "model", *BUCKETS, "graded", "acc%", "true acc%", "false acc%",
        "ans-true%", "med rsn toks",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for model, stats in summary.items():
        cells = [
            model,
            *[str(stats["counts"][bucket]) for bucket in BUCKETS],
            str(stats["graded"]),
            _pct(stats["accuracy"]),
            _pct(stats["true_accuracy"]),
            _pct(stats["false_accuracy"]),
            _pct(stats["answer_true_rate"]),
            str(stats["median_reasoning_tokens"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------- Main


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(
        description="Benchmark LLM truth judgment of ETP implications."
    )
    cli.add_argument("--seed", type=int, default=0, help="sampling seed")
    cli.add_argument(
        "--per-bin",
        type=int,
        default=10,
        help="pairs per total-ops bin, split 50/50 true/false (default 10)",
    )
    cli.add_argument(
        "--bins", default="2:8", metavar="MIN:MAX",
        help="total-ops bins to cover (default 2:8; bin 1 would need a "
        "vacuous law and is structurally empty)",
    )
    cli.add_argument(
        "--form",
        choices=tuple(RENDERERS),
        default="story",
        help="presentation arm: themed story, literal description, or the "
        "rigid two-line prefix grammar (default story)",
    )
    cli.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated OpenRouter model slugs",
    )
    cli.add_argument(
        "--prompt-template",
        type=Path,
        default=None,
        metavar="PATH",
        help="override the chosen form's prove_*_prompt.md template",
    )
    cli.add_argument("--out-dir", type=Path, default=None, help="run directory")
    cli.add_argument("--concurrency", type=int, default=4)
    cli.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="default 16384 with --reasoning on, else 4096",
    )
    cli.add_argument("--temperature", type=float, default=0.0)
    cli.add_argument("--timeout", type=float, default=180.0)
    cli.add_argument(
        "--reasoning",
        choices=("on", "off"),
        default=None,
        help="standardize thinking: uniform prompt wrapper plus the native "
        "reasoning toggle where supported; omit for legacy behavior",
    )
    cli.add_argument(
        "--dry-run",
        action="store_true",
        help="no network calls: grade synthesized correct answers instead",
    )
    args = cli.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    regime = args.reasoning
    form = args.form
    bins = tuple(parse_bins(args.bins))
    max_tokens = args.max_tokens or (16384 if regime == "on" else 4096)
    template_path = args.prompt_template or PROMPT_PATHS[form]
    suffix = f"-think-{regime}" if regime else ""
    out_dir = args.out_dir or (
        EXPERIMENT_DIR / "runs" / f"run-truth{args.per_bin}-s{args.seed}-{form}{suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model_info: Dict[str, dict] = {}
    if not args.dry_run:
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set (use --dry-run to test offline)")
        model_info = validate_models(models, api_key, args.timeout)
    native_reasoning = {
        model: build_reasoning_payload(regime, model_info.get(model, {}))
        for model in models
    }

    equations, equations_sha = load_equations()
    matrix = ensure_matrix()
    if len(equations) != matrix.n:
        raise SystemExit(f"{len(equations)} equations but a {matrix.n}x{matrix.n} matrix")
    pairs = sample_truth_balanced(equations, matrix, args.per_bin, args.seed, bins)
    samples = [make_sample(equations, pair, form, template_path) for pair in pairs]
    assert all(sample["truth"] is not None for sample in samples)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "truth",
        "seed": args.seed,
        "per_bin": args.per_bin,
        "bins": args.bins,
        "truth_balance": "50/50 per bin",
        "form": form,
        "models": models,
        "dry_run": args.dry_run,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "reasoning_regime": regime,
        "regime_prefix": REGIME_PREFIX.get(regime, ""),
        "regime_suffix": REGIME_SUFFIX.get(regime, ""),
        "native_reasoning": native_reasoning,
        "equations_sha256": equations_sha,
        "outcomes_snapshot": matrix.meta.get("snapshot"),
        "outcomes_zip_sha256": matrix.meta.get("zip_sha256"),
        "outcomes_matrix_sha256": matrix.meta.get("matrix_sha256"),
        "truth_availability": {
            str(b): c for b, c in truth_availability(equations, matrix, bins).items()
        },
        "prompt_template": template_path.name,
        "prompt_template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    results_path = out_dir / "results.jsonl"
    rows, completed = load_completed(results_path)
    tasks = [
        (sample, model)
        for sample in samples
        for model in models
        if (sample["pair_id"], model) not in completed
    ]
    print(f"{len(samples)} pairs x {len(models)} models; {len(tasks)} calls to make "
          f"({len(rows)} already done)")

    if args.dry_run:
        def caller(model: str, prompt: str, sample: dict) -> dict:
            content = f"(reasoning omitted)\nANSWER: {sample['truth']}"
            return {"content": content, "error": None, "usage": None, "latency_s": 0.0}
    else:
        def caller(model: str, prompt: str, sample: dict) -> dict:
            return call_openrouter(
                model, prompt, api_key, max_tokens, args.timeout,
                temperature=args.temperature,
                reasoning=native_reasoning[model],
            )

    lock = threading.Lock()
    with results_path.open("a", encoding="utf-8") as fh:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(run_one, sample, model, caller, regime)
                for sample, model in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    rows.append(row)
                print(f"  {row['pair_id']:>12}  {row['model']:<36} {row['bucket']}")

    summary = aggregate(rows, models)
    table = summary_table(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(
        f"# Truth-judgment run: seed={args.seed}, per_bin={args.per_bin}, "
        f"bins={args.bins}, form={form}, reasoning={regime or 'legacy'}\n\n{table}\n",
        encoding="utf-8",
    )
    print()
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
