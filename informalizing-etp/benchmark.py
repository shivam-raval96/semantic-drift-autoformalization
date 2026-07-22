#!/usr/bin/env python3
"""Benchmark: LLM formalization of ETP implication stories via OpenRouter.

For each sampled (E, F) equation pair: render the question in the chosen
form (--form story: themed storyform narrative with formalize_prompt.md;
--form literal: direct literalform description with literal_prompt.md;
--form two-stage: the story, first abstracted by the model into a literal
description with abstract_prompt.md, whose output is then formalized with
literal_prompt.md in a second call), build the formalization prompt with
checkform, send it to each model through the OpenRouter chat-completions
API, and grade the raw response syntactically with checkform.grade.
Sampling is seeded and both render pipelines are pure, so a run is
reproducible end to end; only the model responses are nondeterministic.
The RNG stream never depends on the form, so runs of any form with the
same seed cover the same pair set.

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

from checkform import PROMPT_PATH as STORY_PROMPT_PATH
from checkform import build_prompt, grade
from filter_vacuous import is_vacuous
from genform import parse_bins
from literalform import render_description
from storyform import Op, ParseError, Term, Var, parse_equation, render_story

LITERAL_PROMPT_PATH = Path(__file__).resolve().parent / "literal_prompt.md"
ABSTRACT_PROMPT_PATH = Path(__file__).resolve().parent / "abstract_prompt.md"

# Each form is a (renderer, prompt template) arm over the same record
# schema; checkform grades all of them unchanged. The two-stage arm
# renders the story and prompts for its literalform-style abstraction
# (stage 1); the stage-2 prompt is built at run time from
# LITERAL_PROMPT_PATH and the stage-1 response (see run_two_stage).
FORMS = {
    "story": (render_story, STORY_PROMPT_PATH),
    "literal": (render_description, LITERAL_PROMPT_PATH),
    "two-stage": (render_story, ABSTRACT_PROMPT_PATH),
}

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
# on top of the untouched prompt-template output. "on" also enables
# native reasoning where the model supports OpenRouter's reasoning
# parameter; "off" disables it there. Models without the parameter get
# the wrapper alone. Wordings are form-specific and kept byte-identical
# to earlier runs (story: experiments 01-02, literal: experiments 03-04).
# The literal wording predates literalform's named intermediates, but
# still fits: "each application of the operation" is exactly one
# definition step. "abstract" is the two-stage arm's stage-1 wrapper —
# its output is a description, not the two lines, so it gets its own
# wording; stage 2 reuses the literal wrapper unchanged.
_TWO_LINES_SUFFIX = (
    "\n\nRespond with only the two required lines, and no other text before them."
)
REGIME_PREFIX = {
    "on": {
        "story": (
            "Work through the story step by step first — write out what "
            "expression each numbered intermediate stands for, one at a time — "
            "and only then finish with the two required lines.\n\n"
        ),
        "literal": (
            "Work through the description step by step first — write out what "
            "expression each application of the operation stands for, one at "
            "a time — and only then finish with the two required lines.\n\n"
        ),
        "abstract": (
            "Work through the story step by step first — write out what "
            "expression each numbered intermediate stands for, one at a time — "
            "and only then finish with the complete rewritten description.\n\n"
        ),
    },
}
REGIME_SUFFIX = {
    "off": {
        "story": _TWO_LINES_SUFFIX,
        "literal": _TWO_LINES_SUFFIX,
        "abstract": (
            "\n\nRespond with only the rewritten description, and no other "
            "text before it."
        ),
    },
}


def regime_prefix(regime: Optional[str], form: str) -> str:
    return REGIME_PREFIX.get(regime, {}).get(form, "")


def regime_suffix(regime: Optional[str], form: str) -> str:
    return REGIME_SUFFIX.get(regime, {}).get(form, "")


def wrap_prompt(
    prompt: str, regime: Optional[str], model: str = "", form: str = "story"
) -> str:
    text = regime_prefix(regime, form) + prompt + regime_suffix(regime, form)
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


def make_sample(
    equations: List[str],
    e_num: int,
    f_num: int,
    form: str,
    template_path: Optional[Path] = None,
    label_prefix: str = "E",
) -> Optional[dict]:
    """Render one (E, F) pair in the given form, or None if unrenderable.

    label_prefix names the equations after their line numbers; the
    default "E" matches ETP numbering. Synthetic lists should pass a
    distinct prefix so their labels cannot be misread as ETP numbers.
    """
    render, template = FORMS[form]
    if template_path is not None:
        template = template_path
    try:
        story, metadata = render(equations[e_num - 1], equations[f_num - 1])
    except (ParseError, ValueError):
        return None
    metadata["label_e"] = f"{label_prefix}{e_num}"
    metadata["label_f"] = f"{label_prefix}{f_num}"
    prompt = build_prompt({"story": story, "metadata": metadata}, template_path=template)
    return {
        "pair_id": f"{label_prefix}{e_num}-{label_prefix}{f_num}",
        "form": form,
        "story": story,
        "metadata": metadata,
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
    }


def sample_pairs(
    equations: List[str],
    n: int,
    seed: int,
    form: str = "story",
    template_path: Optional[Path] = None,
    label_prefix: str = "E",
) -> List[dict]:
    """Deterministically sample n renderable (E, F) pairs.

    Draws ordered pairs from a seeded RNG, discarding duplicates, E = F,
    and pairs the renderer cannot handle (parse failures, more variables
    than the theme palette holds). Discards consume the same RNG stream,
    so the result depends only on (equations, n, seed, form) — and both
    renderers reject exactly the same pairs, so story and literal runs
    with one seed cover the identical pair set.
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
        sample = make_sample(equations, e_num, f_num, form, template_path, label_prefix)
        if sample is not None:
            samples.append(sample)
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
    equations: List[str],
    per_bin: int,
    seed: int,
    bins: Tuple[int, ...] = tuple(range(1, 9)),
    form: str = "story",
    template_path: Optional[Path] = None,
    label_prefix: str = "E",
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
            sample = make_sample(
                equations, e_num, f_num, form, template_path, label_prefix
            )
            if sample is None:
                continue
            sample.update(_complexity(sample))
            samples.append(sample)
            got += 1
    return samples


def sample_pairs_balanced(
    equations: List[str],
    per_bin: int,
    seed: int,
    bins: Tuple[int, ...] = tuple(range(1, 11)),
    form: str = "story",
    template_path: Optional[Path] = None,
    label_prefix: str = "E",
) -> List[dict]:
    """Deterministically sample per_bin renderable pairs per
    per-equation operation count, both laws drawn from the same bin.

    sample_pairs_stratified bins by the pair's summed operation count,
    so one bin mixes very different splits — experiment 07 found the
    mix distorts the complexity axis (a bin-4 pair can be a vacuous law
    against a 4-op partner). Here E and F each carry exactly the bin's
    op count: bins are homogeneous, ops_total = 2 * bin, and no bin
    contains a vacuous law. The ETP list stops at 4 ops per equation;
    bins above 4 need a synthetic list (--equations-path, genform.py).
    Same-seed determinism as sample_pairs.
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
        pool = by_ops.get(target, [])
        if len(pool) < 2:
            raise SystemExit(f"no equations available for eq-ops bin {target}")
        got = 0
        attempts = 0
        while got < per_bin:
            attempts += 1
            if attempts > 1000 * per_bin:
                raise SystemExit(f"could not fill eq-ops bin {target}")
            e_num = pool[rng.randrange(len(pool))]
            f_num = pool[rng.randrange(len(pool))]
            if e_num == f_num or (e_num, f_num) in chosen:
                continue
            chosen.add((e_num, f_num))
            sample = make_sample(
                equations, e_num, f_num, form, template_path, label_prefix
            )
            if sample is None:
                continue
            sample.update(_complexity(sample))
            samples.append(sample)
            got += 1
    return samples


def drop_vacuous(samples: List[dict]) -> List[dict]:
    """Filter out pairs containing a zero-op law (E1 x = x, E2 x = y).

    Applied after sampling, so the RNG stream is untouched: the surviving
    pairs are exactly the unfiltered draw minus the vacuous ones, in the
    same order (see the vacuous-law convention in experiments/README.md).
    """
    return [sample for sample in samples if not is_vacuous(sample)]


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


def validate_models(models: List[str], api_key: str, timeout: float) -> Dict[str, dict]:
    """Fail fast if a requested model slug is unknown to OpenRouter.

    Returns each requested model's metadata, including its supported
    parameters and reasoning capabilities.
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
    return {m: entries[m] for m in models}


def build_reasoning_payload(regime: Optional[str], model_info: dict) -> Optional[dict]:
    """Translate the benchmark regime into the model's native API setting."""
    supported = set(model_info.get("supported_parameters") or ())
    if not regime or "reasoning" not in supported:
        return None
    if regime == "on":
        return {"enabled": True}

    reasoning = model_info.get("reasoning") or {}
    efforts = reasoning.get("supported_efforts") or ()
    if "none" in efforts:
        return {"effort": "none", "exclude": True}
    if not reasoning.get("mandatory", False):
        return {"enabled": False, "exclude": True}

    # Some endpoints cannot disable reasoning. Preserve the historical
    # behavior by selecting their lowest advertised effort.
    for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
        if effort in efforts:
            return {"effort": effort, "exclude": True}
    return {"enabled": True, "exclude": True}


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


def _row_base(sample: dict, model: str, regime: Optional[str], form: str) -> dict:
    """The row fields identifying the (pair, model) task, shared by both arms."""
    return {
        "pair_id": sample["pair_id"],
        "form": form,
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
    }


def _hash12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def run_one(sample: dict, model: str, caller, regime: Optional[str]) -> dict:
    form = sample.get("form", "story")
    if form == "two-stage":
        return run_two_stage(sample, model, caller, regime)
    sent_prompt = wrap_prompt(sample["prompt"], regime, model, form)
    call = caller(model, sent_prompt, sample)
    response = call["content"]
    verdict = grade(response, sample["metadata"]) if response is not None else None
    row = _row_base(sample, model, regime, form)
    row.update(
        sent_prompt_hash=_hash12(sent_prompt),
        response=response,
        verdict=verdict,
        api_error=call["error"],
        usage=call["usage"],
        latency_s=call["latency_s"],
        routed_model=call.get("routed_model"),
        provider=call.get("provider"),
        finish_reason=call.get("finish_reason"),
        reasoning_tokens=call.get("reasoning_tokens"),
    )
    row["bucket"] = bucket_of(row)
    return row


def run_two_stage(sample: dict, model: str, caller, regime: Optional[str]) -> dict:
    """Two model calls: story -> literal description -> the two lines.

    Stage 1 sends the sample's prompt (the story under abstract_prompt.md)
    and stage 2 feeds its raw response — verbatim, with no extraction or
    validation, since the pipeline's end-to-end fidelity is the
    measurement — into literal_prompt.md. Top-level call fields describe
    stage 2 (the graded call) so grading, aggregation, resume, and charts
    are unchanged; stage-1 bookkeeping lives in the stage1_* fields.
    """
    stage1_sent = wrap_prompt(sample["prompt"], regime, model, "abstract")
    call1 = caller(model, stage1_sent, sample, stage=1)
    stage1_response = call1["content"]

    call2 = stage2_sent = None
    if stage1_response is not None:
        stage2_prompt = build_prompt(
            {"story": stage1_response}, template_path=LITERAL_PROMPT_PATH
        )
        stage2_sent = wrap_prompt(stage2_prompt, regime, model, "literal")
        call2 = caller(model, stage2_sent, sample, stage=2)

    response = call2["content"] if call2 else None
    verdict = grade(response, sample["metadata"]) if response is not None else None
    errors = [
        f"stage {n}: {call['error']}"
        for n, call in ((1, call1), (2, call2))
        if call and call["error"]
    ]

    row = _row_base(sample, model, regime, "two-stage")
    final = call2 or {"usage": None, "latency_s": None}
    row.update(
        sent_prompt_hash=_hash12(stage2_sent) if stage2_sent is not None else None,
        response=response,
        verdict=verdict,
        api_error="; ".join(errors) or None,
        usage=final.get("usage"),
        latency_s=final.get("latency_s"),
        routed_model=final.get("routed_model"),
        provider=final.get("provider"),
        finish_reason=final.get("finish_reason"),
        reasoning_tokens=final.get("reasoning_tokens"),
        stage1_sent_prompt_hash=_hash12(stage1_sent),
        stage1_response=stage1_response,
        stage1_usage=call1["usage"],
        stage1_latency_s=call1["latency_s"],
        stage1_routed_model=call1.get("routed_model"),
        stage1_provider=call1.get("provider"),
        stage1_finish_reason=call1.get("finish_reason"),
        stage1_reasoning_tokens=call1.get("reasoning_tokens"),
    )
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
            {
                row["model"]
                for row in rows
                if max(
                    row.get("reasoning_tokens") or 0,
                    row.get("stage1_reasoning_tokens") or 0,
                )
                > 16
            }
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
        "--stratify-eq-ops",
        type=int,
        default=None,
        metavar="PER_BIN",
        help="sample PER_BIN pairs per per-equation operation count, "
        "both laws drawn from the same bin (see --eq-bins); balanced "
        "complexity studies, needed past the ETP cap (genform.py)",
    )
    cli.add_argument(
        "--eq-bins",
        default="1:10",
        metavar="MIN:MAX",
        help="per-equation operation counts covered by --stratify-eq-ops "
        "(default 1:10)",
    )
    cli.add_argument(
        "--label-prefix",
        default="E",
        metavar="PREFIX",
        help="prefix for equation labels and pair ids (default E, the "
        "ETP numbering; use a distinct prefix for synthetic lists so "
        "labels cannot be misread as ETP numbers)",
    )
    cli.add_argument(
        "--exclude-vacuous",
        action="store_true",
        help="drop sampled pairs containing a zero-op law (E1 x = x, "
        "E2 x = y); applied after the draw, so the surviving pair ids "
        "match the unfiltered draw",
    )
    cli.add_argument(
        "--form",
        choices=tuple(FORMS),
        default="story",
        help="rendering arm: themed question-story (storyform), direct "
        "literal description (literalform), or two-stage (story abstracted "
        "into a literal description by the model, then formalized in a "
        "second call); default story",
    )
    cli.add_argument(
        "--prompt-template",
        type=Path,
        default=None,
        metavar="PATH",
        help="override the chosen form's prompt template (for two-stage, "
        "the stage-1 template); recorded in run_meta.json",
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

    if args.stratify_ops and args.stratify_eq_ops:
        raise SystemExit("--stratify-ops and --stratify-eq-ops are mutually exclusive")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    regime = args.reasoning
    form = args.form
    max_tokens = args.max_tokens or (16384 if regime == "on" else 4096)
    form_tag = f"-{form}" if form != "story" else ""
    suffix = f"-think-{regime}" if regime else ""
    if args.stratify_eq_ops:
        stem = f"run-eqstrat{args.stratify_eq_ops}-s{args.seed}"
    elif args.stratify_ops:
        stem = f"run-strat{args.stratify_ops}-s{args.seed}"
    else:
        stem = f"run-s{args.seed}-n{args.n}"
    out_dir = args.out_dir or Path(f"results/{stem}{form_tag}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model_info: Dict[str, dict] = {}
    if not args.dry_run:
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set (use --dry-run to test offline)")
        model_info = validate_models(models, api_key, args.timeout)
    # Native toggle only where the model supports it; wrapper covers the rest.
    native_reasoning = {
        model: build_reasoning_payload(regime, model_info.get(model, {}))
        for model in models
    }

    if args.equations_path != EQUATIONS_PATH and not args.equations_path.exists():
        # Without this, load_equations would silently download the ETP
        # list into the missing path.
        raise SystemExit(f"equations file not found: {args.equations_path}")
    equations, equations_sha = load_equations(args.equations_path)
    if args.stratify_eq_ops:
        samples = sample_pairs_balanced(
            equations,
            args.stratify_eq_ops,
            args.seed,
            bins=tuple(parse_bins(args.eq_bins)),
            form=form,
            template_path=args.prompt_template,
            label_prefix=args.label_prefix,
        )
    elif args.stratify_ops:
        samples = sample_pairs_stratified(
            equations,
            args.stratify_ops,
            args.seed,
            form=form,
            template_path=args.prompt_template,
            label_prefix=args.label_prefix,
        )
    else:
        samples = sample_pairs(
            equations, args.n, args.seed, form=form,
            template_path=args.prompt_template,
            label_prefix=args.label_prefix,
        )
    if args.exclude_vacuous:
        samples = drop_vacuous(samples)

    # Two-stage wraps its stage-1 prompt with the "abstract" wording and
    # its stage-2 prompt with the literal wording; record both.
    wrap_form = "abstract" if form == "two-stage" else form
    template_path = args.prompt_template or FORMS[form][1]
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n": args.n,
        "stratify_ops": args.stratify_ops,
        "stratify_eq_ops": args.stratify_eq_ops,
        "eq_bins": args.eq_bins if args.stratify_eq_ops else None,
        "label_prefix": args.label_prefix,
        "exclude_vacuous": args.exclude_vacuous,
        "form": form,
        "models": models,
        "dry_run": args.dry_run,
        "max_tokens": max_tokens,
        "reasoning_regime": regime,
        "regime_prefix": regime_prefix(regime, wrap_form),
        "regime_suffix": regime_suffix(regime, wrap_form),
        "native_reasoning": {m: v for m, v in native_reasoning.items()},
        "equations_sha256": equations_sha,
        "prompt_template": template_path.name,
        "prompt_template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
    }
    if form == "two-stage":
        meta["stage2_template"] = LITERAL_PROMPT_PATH.name
        meta["stage2_template_sha256"] = hashlib.sha256(
            LITERAL_PROMPT_PATH.read_bytes()
        ).hexdigest()
        meta["stage2_regime_prefix"] = regime_prefix(regime, "literal")
        meta["stage2_regime_suffix"] = regime_suffix(regime, "literal")
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
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
        # Stage 1 of the two-stage arm "answers" with the deterministic
        # literalform rendering, so the dry run exercises the real stage-2
        # prompt build on stage-1 output and must still grade 100% exact.
        def caller(model: str, prompt: str, sample: dict, stage: int = 2) -> dict:
            metadata = sample["metadata"]
            if stage == 1:
                content = render_description(
                    metadata["equation_e"], metadata["equation_f"]
                )[0]
            else:
                content = synthesize_response(metadata)
            return {
                "content": content,
                "error": None,
                "usage": None,
                "latency_s": 0.0,
            }
    else:
        def caller(model: str, prompt: str, sample: dict, stage: int = 2) -> dict:
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
        f"# Benchmark run: seed={args.seed}, n={args.n}, form={form}, "
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
