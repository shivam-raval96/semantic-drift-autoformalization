#!/usr/bin/env python3
"""Benchmark: LLM formalization of ETP implication stories via OpenRouter.

For each sampled (E, F) equation pair: render the question-story with
storyform, build the formalization prompt with checkform, send it to each
model through the OpenRouter chat-completions API, and grade the raw
response syntactically with checkform.grade. Sampling is seeded and the
story pipeline is pure, so a run is reproducible end to end; only the
model responses are nondeterministic.

Artifacts, written under --out-dir:
    run_meta.json   seed, n, models, equations-file sha256, CLI args
    samples.jsonl   one row per sampled pair (story, prompt, metadata)
    results.jsonl   one row per (pair, model) with response and verdict
    summary.json    per-model bucket counts and rates
    summary.md      the same summary as a Markdown table

Rerunning with the same --out-dir resumes: rows already graded are kept,
rows that failed with an API error are retried.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from checkform import build_prompt, grade
from storyform import Op, ParseError, Term, Var, parse_equation, render_story

EQUATIONS_URL = (
    "https://raw.githubusercontent.com/teorth/equational_theories/main/data/equations.txt"
)
EQUATIONS_PATH = Path(__file__).resolve().parent / "data" / "equations.txt"

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Four open-weight models and four lightweight closed models.
DEFAULT_MODELS = (
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3-32b",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5-mini",
)

# Uniform per-regime prompt wrappers, applied identically to every model
# on top of the untouched formalize_prompt.md output. "on" also enables
# native reasoning where the model supports OpenRouter's reasoning
# parameter; "off" disables it there. Models without the parameter get
# the wrapper alone.
REGIME_PREFIX = {
    "on": (
        "Work through the story step by step first — write out what "
        "expression each numbered intermediate stands for, one at a time — "
        "and only then finish with the two required lines.\n\n"
    ),
}
REGIME_SUFFIX = {
    "off": "\n\nRespond with only the two required lines, and no other text before them.",
}


def wrap_prompt(prompt: str, regime: Optional[str], model: str = "") -> str:
    text = REGIME_PREFIX.get(regime, "") + prompt + REGIME_SUFFIX.get(regime, "")
    # Qwen3's vendor-documented soft switch; some providers ignore the
    # OpenRouter reasoning toggle for it (observed: DeepInfra).
    if regime == "off" and model.startswith("qwen/qwen3"):
        text += "\n/no_think"
    return text

RETRYABLE_STATUSES = {429, 500, 502, 503, 529}
MAX_RETRIES = 5

BUCKETS = (
    "exact",
    "correct-swapped",
    "correct-dualized",
    "wrong",
    "unparseable",
    "api-error",
)


# ------------------------------------------------------------------- Data


def load_equations(path: Path = EQUATIONS_PATH, url: str = EQUATIONS_URL) -> Tuple[List[str], str]:
    """Return (equations, sha256) with equation N at index N-1.

    Downloads the ETP equation list to a local cache on first use; later
    runs read the cache, and its digest in run_meta.json pins which file
    a run used.
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
    else:
        data = path.read_bytes()
    equations = data.decode("utf-8").splitlines()
    if not equations:
        raise SystemExit(f"equation list {path} is empty")
    parse_equation(equations[0])  # fail fast if the format ever changes
    return equations, hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------- Sampling


def sample_pairs(equations: List[str], n: int, seed: int) -> List[dict]:
    """Deterministically sample n renderable (E, F) pairs.

    Draws ordered pairs from a seeded RNG, discarding duplicates, E = F,
    and pairs storyform cannot render (parse failures, more variables
    than the theme palette holds). Discards consume the same RNG stream,
    so the result depends only on (equations, n, seed).
    """
    rng = random.Random(seed)
    samples: List[dict] = []
    chosen = set()
    attempts = 0
    while len(samples) < n:
        attempts += 1
        if attempts > 100 * n:
            raise SystemExit(f"could not sample {n} renderable pairs in {attempts} draws")
        e_num = rng.randrange(1, len(equations) + 1)
        f_num = rng.randrange(1, len(equations) + 1)
        if e_num == f_num or (e_num, f_num) in chosen:
            continue
        chosen.add((e_num, f_num))
        e_text, f_text = equations[e_num - 1], equations[f_num - 1]
        try:
            story, metadata = render_story(e_text, f_text)
        except (ParseError, ValueError):
            continue
        metadata["label_e"] = f"E{e_num}"
        metadata["label_f"] = f"E{f_num}"
        prompt = build_prompt({"story": story, "metadata": metadata})
        samples.append(
            {
                "pair_id": f"E{e_num}-E{f_num}",
                "story": story,
                "metadata": metadata,
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
            }
        )
    return samples


def _term_ops(term: Term) -> int:
    if isinstance(term, Var):
        return 0
    return 1 + _term_ops(term.left) + _term_ops(term.right)


def _term_depth(term: Term) -> int:
    if isinstance(term, Var):
        return 0
    return 1 + max(_term_depth(term.left), _term_depth(term.right))


def _complexity(sample: dict) -> dict:
    e_lhs, e_rhs = parse_equation(sample["metadata"]["equation_e"])
    f_lhs, f_rhs = parse_equation(sample["metadata"]["equation_f"])
    ops_e = _term_ops(e_lhs) + _term_ops(e_rhs)
    ops_f = _term_ops(f_lhs) + _term_ops(f_rhs)
    return {
        "ops_e": ops_e,
        "ops_f": ops_f,
        "ops_total": ops_e + ops_f,
        "depth": max(map(_term_depth, (e_lhs, e_rhs, f_lhs, f_rhs))),
    }


def sample_pairs_stratified(
    equations: List[str], per_bin: int, seed: int, bins: Tuple[int, ...] = tuple(range(1, 9))
) -> List[dict]:
    """Deterministically sample per_bin renderable pairs for each total
    operation count in bins.

    Uniform sampling almost always lands on maximal 4+4-operation pairs
    (they dominate the ETP list), so complexity studies need this
    stratified mode: equations are grouped by operation count, and each
    bin draws a split (a, b) with a + b = bin, then one equation from
    each group. Same-seed determinism as sample_pairs.
    """
    rng = random.Random(seed)
    by_ops: Dict[int, List[int]] = {}
    for number, text in enumerate(equations, start=1):
        try:
            lhs, rhs = parse_equation(text)
        except ParseError:
            continue
        by_ops.setdefault(_term_ops(lhs) + _term_ops(rhs), []).append(number)

    samples: List[dict] = []
    chosen = set()
    for target in bins:
        splits = [
            (a, target - a)
            for a in range(0, 5)
            if 0 <= target - a <= 4 and by_ops.get(a) and by_ops.get(target - a)
        ]
        if not splits:
            raise SystemExit(f"no equations available for ops bin {target}")
        got = 0
        attempts = 0
        while got < per_bin:
            attempts += 1
            if attempts > 1000 * per_bin:
                raise SystemExit(f"could not fill ops bin {target}")
            a, b = splits[rng.randrange(len(splits))]
            e_num = by_ops[a][rng.randrange(len(by_ops[a]))]
            f_num = by_ops[b][rng.randrange(len(by_ops[b]))]
            if e_num == f_num or (e_num, f_num) in chosen:
                continue
            chosen.add((e_num, f_num))
            try:
                story, metadata = render_story(equations[e_num - 1], equations[f_num - 1])
            except (ParseError, ValueError):
                continue
            metadata["label_e"] = f"E{e_num}"
            metadata["label_f"] = f"E{f_num}"
            prompt = build_prompt({"story": story, "metadata": metadata})
            sample = {
                "pair_id": f"E{e_num}-E{f_num}",
                "story": story,
                "metadata": metadata,
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
            }
            sample.update(_complexity(sample))
            samples.append(sample)
            got += 1
    return samples


# ------------------------------------------------------------- OpenRouter


def _post_json(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_openrouter(
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    reasoning: Optional[dict] = None,
) -> dict:
    """One chat completion; never raises. Retries 429/5xx with backoff."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if reasoning is not None:
        payload["reasoning"] = reasoning
    error = None
    for attempt in range(MAX_RETRIES):
        start = time.monotonic()
        try:
            body = _post_json(OPENROUTER_CHAT_URL, payload, api_key, timeout)
            latency = time.monotonic() - start
            if "error" in body:  # OpenRouter can return errors with HTTP 200
                error = f"api error: {body['error']}"
                code = body["error"].get("code") if isinstance(body["error"], dict) else None
                if code not in RETRYABLE_STATUSES:
                    break
            else:
                choice = body["choices"][0]
                usage = body.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                return {
                    "content": choice["message"]["content"],
                    "error": None,
                    "usage": usage,
                    "latency_s": round(latency, 2),
                    "routed_model": body.get("model"),
                    "provider": body.get("provider"),
                    "finish_reason": choice.get("finish_reason"),
                    "reasoning_tokens": details.get("reasoning_tokens"),
                }
        except urllib.error.HTTPError as http_error:
            detail = http_error.read().decode("utf-8", errors="replace")[:500]
            error = f"HTTP {http_error.code}: {detail}"
            if http_error.code not in RETRYABLE_STATUSES:
                break
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        time.sleep(2**attempt)
    return {"content": None, "error": error, "usage": None, "latency_s": None}


def validate_models(models: List[str], api_key: str, timeout: float) -> Dict[str, set]:
    """Fail fast if a requested model slug is unknown to OpenRouter.

    Returns each requested model's supported_parameters set, so the
    caller knows which models accept the native reasoning toggle.
    """
    request = urllib.request.Request(
        OPENROUTER_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        entries = {entry["id"]: entry for entry in json.loads(response.read())["data"]}
    missing = [m for m in models if m not in entries]
    if missing:
        lines = []
        for model in missing:
            near = difflib.get_close_matches(model, entries, n=3, cutoff=0.4)
            hint = f" (close matches: {', '.join(near)})" if near else ""
            lines.append(f"  {model}{hint}")
        raise SystemExit("unknown OpenRouter model(s):\n" + "\n".join(lines))
    return {m: set(entries[m].get("supported_parameters") or ()) for m in models}


# ---------------------------------------------------------------- Dry run


def _prefix(term: Term) -> str:
    if isinstance(term, Var):
        return term.name
    return f"op({_prefix(term.left)}, {_prefix(term.right)})"


def synthesize_response(metadata: dict) -> str:
    """Fabricate a correct answer from the record's own canonical forms.

    The canonical strings are storyform infix syntax except for the
    digits in v1, v2, ...; those become letters (grading renames
    variables anyway). This exercises prompt building and grading
    without any network; a dry run must therefore grade 100% exact.
    """
    lines = []
    for label, key in (("ASSUME", "canonical_e"), ("ASK", "canonical_f")):
        text = re.sub(r"v(\d+)", lambda m: "v" + "abcdef"[int(m.group(1)) - 1], metadata[key])
        lhs, rhs = parse_equation(text)
        lines.append(f"{label}: {_prefix(lhs)} = {_prefix(rhs)}")
    return "\n".join(lines)


# ---------------------------------------------------------------- Grading


def bucket_of(row: dict) -> str:
    """Map a result row to one of BUCKETS."""
    if row["response"] is None:
        return "api-error"
    verdict = row["verdict"]
    if verdict["status"] != "correct":
        return verdict["status"]  # "wrong" or "unparseable"
    transform = verdict["transform"]
    if transform["dual"]:
        return "correct-dualized"
    if transform["swap_e"] or transform["swap_f"]:
        return "correct-swapped"
    return "exact"


def run_one(sample: dict, model: str, caller, regime: Optional[str]) -> dict:
    sent_prompt = wrap_prompt(sample["prompt"], regime, model)
    call = caller(model, sent_prompt, sample)
    response = call["content"]
    verdict = grade(response, sample["metadata"]) if response is not None else None
    row = {
        "pair_id": sample["pair_id"],
        "label_e": sample["metadata"]["label_e"],
        "label_f": sample["metadata"]["label_f"],
        "theme": sample["metadata"]["theme"],
        "model": model,
        "regime": regime,
        "ops_e": sample.get("ops_e"),
        "ops_f": sample.get("ops_f"),
        "ops_total": sample.get("ops_total"),
        "depth": sample.get("depth"),
        "prompt_hash": sample["prompt_hash"],
        "sent_prompt_hash": hashlib.sha256(sent_prompt.encode("utf-8")).hexdigest()[:12],
        "response": response,
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


def aggregate(rows: List[dict], models: List[str]) -> dict:
    summary: Dict[str, dict] = {}
    for model in models:
        mine = [row for row in rows if row["model"] == model]
        counts = {bucket: 0 for bucket in BUCKETS}
        for row in mine:
            counts[row["bucket"]] += 1
        graded = sum(counts.values()) - counts["api-error"]
        correct = counts["exact"] + counts["correct-swapped"] + counts["correct-dualized"]
        lengths = [len(row["response"]) for row in mine if row["response"]]
        reasoned = [
            row["reasoning_tokens"] for row in mine if row.get("reasoning_tokens")
        ]
        summary[model] = {
            "counts": counts,
            "graded": graded,
            "correct_rate": round(correct / graded, 4) if graded else None,
            "exact_rate": round(counts["exact"] / graded, 4) if graded else None,
            "median_response_len": int(statistics.median(lengths)) if lengths else None,
            "rows_with_reasoning": len(reasoned),
            "median_reasoning_tokens": int(statistics.median(reasoned)) if reasoned else 0,
        }
    return summary


def summary_table(summary: dict) -> str:
    headers = ["model", *BUCKETS, "graded", "correct%", "rsn rows", "med rsn toks"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for model, stats in summary.items():
        rate = stats["correct_rate"]
        cells = [
            model,
            *[str(stats["counts"][bucket]) for bucket in BUCKETS],
            str(stats["graded"]),
            f"{100 * rate:.1f}" if rate is not None else "-",
            str(stats["rows_with_reasoning"]),
            str(stats["median_reasoning_tokens"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compliance_notes(rows: List[dict], regime: Optional[str]) -> List[str]:
    """Regime violations: native reasoning where none was wanted, or none
    where it was required (prompt-induced CoT still shows in length)."""
    notes = []
    if regime == "off":
        # <= 16 tokens is an empty think-block artifact, not actual reasoning
        offenders = sorted(
            {row["model"] for row in rows if (row.get("reasoning_tokens") or 0) > 16}
        )
        if offenders:
            notes.append(
                "off-regime rows with native reasoning tokens: " + ", ".join(offenders)
            )
    return notes


# ------------------------------------------------------------------- Main


def load_completed(results_path: Path) -> Tuple[List[dict], set]:
    """Rows already graded in a previous run; api-error rows are retried."""
    rows: List[dict] = []
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    kept = [row for row in rows if row["bucket"] != "api-error"]
    return kept, {(row["pair_id"], row["model"]) for row in kept}


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(
        description="Benchmark LLM formalization of ETP stories via OpenRouter."
    )
    cli.add_argument("--n", type=int, default=30, help="number of (E, F) pairs")
    cli.add_argument("--seed", type=int, default=0, help="sampling seed")
    cli.add_argument(
        "--stratify-ops",
        type=int,
        default=None,
        metavar="PER_BIN",
        help="sample PER_BIN pairs per total-operation count 1..8 "
        "instead of --n uniform pairs (complexity studies)",
    )
    cli.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated OpenRouter model slugs",
    )
    cli.add_argument("--out-dir", type=Path, default=None, help="run directory")
    cli.add_argument("--concurrency", type=int, default=4)
    cli.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="default 16384 with --reasoning on, else 4096",
    )
    cli.add_argument("--timeout", type=float, default=120.0)
    cli.add_argument("--equations-path", type=Path, default=EQUATIONS_PATH)
    cli.add_argument(
        "--reasoning",
        choices=("on", "off"),
        default=None,
        help="standardize thinking: uniform prompt wrapper for all models plus "
        "the native reasoning toggle where supported; omit for legacy behavior",
    )
    cli.add_argument(
        "--dry-run",
        action="store_true",
        help="no network calls: grade synthesized correct answers instead",
    )
    args = cli.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    regime = args.reasoning
    max_tokens = args.max_tokens or (16384 if regime == "on" else 4096)
    suffix = f"-think-{regime}" if regime else ""
    stem = (
        f"run-strat{args.stratify_ops}-s{args.seed}"
        if args.stratify_ops
        else f"run-s{args.seed}-n{args.n}"
    )
    out_dir = args.out_dir or Path(f"results/{stem}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    supports: Dict[str, set] = {}
    if not args.dry_run:
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set (use --dry-run to test offline)")
        supports = validate_models(models, api_key, args.timeout)
    # Native toggle only where the model supports it; wrapper covers the rest.
    def reasoning_payload(model: str) -> Optional[dict]:
        if not regime or "reasoning" not in supports.get(model, set()):
            return None
        if regime == "on":
            return {"enabled": True}
        if model.startswith("openai/gpt-5"):
            # reasoning is mandatory for this endpoint; minimal is the floor
            return {"effort": "minimal", "exclude": True}
        # exclude:true is belt-and-braces: some providers ignore the bare
        # toggle, and hiding the reasoning stream keeps content clean.
        return {"enabled": False, "exclude": True}

    native_reasoning = {m: reasoning_payload(m) for m in models}

    equations, equations_sha = load_equations(args.equations_path)
    if args.stratify_ops:
        samples = sample_pairs_stratified(equations, args.stratify_ops, args.seed)
    else:
        samples = sample_pairs(equations, args.n, args.seed)

    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed,
                "n": args.n,
                "stratify_ops": args.stratify_ops,
                "models": models,
                "dry_run": args.dry_run,
                "max_tokens": max_tokens,
                "reasoning_regime": regime,
                "regime_prefix": REGIME_PREFIX.get(regime, ""),
                "regime_suffix": REGIME_SUFFIX.get(regime, ""),
                "native_reasoning": {m: v for m, v in native_reasoning.items()},
                "equations_sha256": equations_sha,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
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
            return {
                "content": synthesize_response(sample["metadata"]),
                "error": None,
                "usage": None,
                "latency_s": 0.0,
            }
    else:
        def caller(model: str, prompt: str, sample: dict) -> dict:
            return call_openrouter(
                model, prompt, api_key, max_tokens, args.timeout,
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
                print(f"  {row['pair_id']:>14}  {row['model']:<32} {row['bucket']}")

    summary = aggregate(rows, models)
    table = summary_table(summary)
    notes = compliance_notes(rows, regime)
    notes_text = "".join(f"\n**Compliance:** {note}\n" for note in notes)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(
        f"# Benchmark run: seed={args.seed}, n={args.n}, "
        f"reasoning={regime or 'legacy'}\n\n{table}\n{notes_text}",
        encoding="utf-8",
    )
    print()
    print(table)
    for note in notes:
        print(f"compliance: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
