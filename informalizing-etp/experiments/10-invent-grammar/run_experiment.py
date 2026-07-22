"""Experiment 10 runner: invent-your-own-grammar mini-experiment.

Prompts each model to invent its own rigid notation for Storyform-style
stories and encode one target story in it, varying the number K of example
stories shown as context (K = 1, 2, 3). Qualitative: responses are collected
and split into GRAMMAR / ENCODING sections; nothing is graded.

Fixed materials, all drawn seeded from the ETP list with vacuous laws
excluded: two target pairs (one low-complexity, ops_total 2-3; one
high-complexity, ops_total 7-8), both rendered in the signal theme, and
three mid-complexity example pairs (ops_total 4-6) in three other themes.
Example sets are nested: K=2 shows K=1's story plus one more.

Run from the repo's informalizing-etp/ directory:

    set -a; source .env; set +a
    python3 experiments/10-invent-grammar/run_experiment.py

--dry-run writes run_meta.json, samples.jsonl, and the six prompts under
prompts/ without any network calls.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark import _complexity, _term_ops, call_openrouter, load_equations, validate_models  # noqa: E402
from storyform import ParseError, parse_equation, render_story  # noqa: E402

PROMPT_PATH = ROOT / "invent_prompt.md"
DEFAULT_OUT_DIR = ROOT / "experiments" / "10-invent-grammar" / "runs" / "run-mini"
DEFAULT_MODELS = ("openai/gpt-5.5", "google/gemini-2.5-flash")
K_VALUES = (1, 2, 3)

# Targets share one theme the examples never use, so at every K the model
# must carry its notation to an unseen setting; examples span three themes
# to push toward a general notation rather than one tied to a setting.
TARGET_THEME = "signal"
EXAMPLE_THEMES = ("paint", "tea", "graft")

TARGET_BINS = {"low": (2, 3), "high": (7, 8)}
EXAMPLE_BIN = (4, 5, 6)

_SECTION_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?\**[ \t]*(GRAMMAR|ENCODING)\b\**[ \t]*:?\**[ \t]*$",
    re.MULTILINE,
)


def pick_materials(equations: List[str], seed: int) -> Tuple[dict, List[dict]]:
    """Seeded draw of 2 targets + 3 examples, all distinct and non-vacuous.

    Draws consume one RNG stream and fill slots in a fixed priority order
    (low target, high target, examples in order), so the selection depends
    only on (equations, seed). A pair must render in its assigned theme to
    be accepted.
    """
    rng = random.Random(seed)
    targets: dict = {"low": None, "high": None}
    examples: List[dict] = []
    chosen = set()
    attempts = 0
    while targets["low"] is None or targets["high"] is None or len(examples) < 3:
        attempts += 1
        if attempts > 200_000:
            raise SystemExit("could not fill all slots; check bins/seed")
        e_num = rng.randrange(1, len(equations) + 1)
        f_num = rng.randrange(1, len(equations) + 1)
        if e_num == f_num or (e_num, f_num) in chosen:
            continue
        chosen.add((e_num, f_num))
        try:
            e_lhs, e_rhs = parse_equation(equations[e_num - 1])
            f_lhs, f_rhs = parse_equation(equations[f_num - 1])
        except ParseError:
            continue
        ops_e = _term_ops(e_lhs) + _term_ops(e_rhs)
        ops_f = _term_ops(f_lhs) + _term_ops(f_rhs)
        if ops_e == 0 or ops_f == 0:  # vacuous law (E1/E2) in the pair
            continue
        total = ops_e + ops_f

        slot = None
        if targets["low"] is None and total in TARGET_BINS["low"]:
            slot = ("target", "low", TARGET_THEME)
        elif targets["high"] is None and total in TARGET_BINS["high"]:
            slot = ("target", "high", TARGET_THEME)
        elif len(examples) < 3 and total in EXAMPLE_BIN:
            slot = ("example", len(examples), EXAMPLE_THEMES[len(examples)])
        if slot is None:
            continue

        kind, key, theme = slot
        try:
            story, metadata = render_story(equations[e_num - 1], equations[f_num - 1], theme)
        except (ParseError, ValueError):
            continue
        metadata["label_e"] = f"E{e_num}"
        metadata["label_f"] = f"E{f_num}"
        record = {
            "pair_id": f"E{e_num}-E{f_num}",
            "story": story,
            "metadata": metadata,
            **_complexity({"metadata": metadata}),
        }
        if kind == "target":
            record["target"] = key
            targets[key] = record
        else:
            examples.append(record)
    return targets, examples


def build_prompt(template: str, examples: List[dict], k: int, target_story: str) -> str:
    blocks = [
        f"### Example story {i}\n\n{record['story']}"
        for i, record in enumerate(examples[:k], 1)
    ]
    return template.replace("{examples}", "\n\n".join(blocks)).replace("{story}", target_story)


def split_sections(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort split of a response into its GRAMMAR and ENCODING parts.

    Uses the last ENCODING marker and the last GRAMMAR marker before it, so
    preamble chatter that mentions the markers early cannot shadow the real
    sections. Returns (None, None) when the markers are absent; the full
    response is always kept alongside.
    """
    if not text:
        return None, None
    matches = list(_SECTION_RE.finditer(text))
    encoding_idxs = [i for i, m in enumerate(matches) if m.group(1) == "ENCODING"]
    if not encoding_idxs:
        return None, None
    enc = encoding_idxs[-1]
    grammar_idxs = [i for i, m in enumerate(matches[:enc]) if m.group(1) == "GRAMMAR"]
    encoding = text[matches[enc].end():].strip()
    grammar = None
    if grammar_idxs:
        gra = grammar_idxs[-1]
        grammar = text[matches[gra].end():matches[enc].start()].strip()
    return grammar, encoding


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true", help="write prompts only, no network")
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    template = PROMPT_PATH.read_text(encoding="utf-8")
    equations, equations_sha = load_equations()
    targets, examples = pick_materials(equations, args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    def brief(record: dict) -> dict:
        return {
            "pair_id": record["pair_id"],
            "equation_e": record["metadata"]["equation_e"],
            "equation_f": record["metadata"]["equation_f"],
            "theme": record["metadata"]["theme"],
            "ops_total": record["ops_total"],
            "depth": record["depth"],
        }

    run_meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "models": models,
        "k_values": list(K_VALUES),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "reasoning": "native-default",
        "prompt_template": PROMPT_PATH.name,
        "prompt_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "equations_sha256": equations_sha,
        "targets": [brief(targets[key]) for key in ("low", "high")],
        "examples": [brief(record) for record in examples],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for key in ("low", "high"):
            handle.write(json.dumps(targets[key]) + "\n")
        for record in examples:
            handle.write(json.dumps({**record, "role": "example"}) + "\n")

    tasks = []  # (target_key, k, prompt, prompt_hash)
    for key in ("low", "high"):
        for k in K_VALUES:
            prompt = build_prompt(template, examples, k, targets[key]["story"])
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            path = prompts_dir / f"{targets[key]['pair_id']}-k{k}.md"
            path.write_text(prompt, encoding="utf-8")
            tasks.append((key, k, prompt, prompt_hash))

    for key in ("low", "high"):
        record = targets[key]
        print(f"target {key}: {record['pair_id']} ops_total={record['ops_total']} depth={record['depth']}")
    for i, record in enumerate(examples, 1):
        print(f"example {i}: {record['pair_id']} ops_total={record['ops_total']} theme={record['metadata']['theme']}")

    if args.dry_run:
        print(f"dry run: wrote {len(tasks)} prompts under {prompts_dir}")
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (use --dry-run to test offline)")
    validate_models(models, api_key, args.timeout)

    def run_one(model: str, key: str, k: int, prompt: str, prompt_hash: str) -> dict:
        record = targets[key]
        call = call_openrouter(model, prompt, api_key, args.max_tokens, args.timeout)
        grammar, encoding = split_sections(call["content"])
        return {
            "pair_id": record["pair_id"],
            "target": key,
            "k": k,
            "model": model,
            "example_pair_ids": [r["pair_id"] for r in examples[:k]],
            "prompt_hash": prompt_hash,
            "ops_e": record["ops_e"],
            "ops_f": record["ops_f"],
            "ops_total": record["ops_total"],
            "depth": record["depth"],
            "response": call["content"],
            "grammar": grammar,
            "encoding": encoding,
            "api_error": call["error"],
            "usage": call["usage"],
            "latency_s": call["latency_s"],
            "routed_model": call.get("routed_model"),
            "provider": call.get("provider"),
            "finish_reason": call.get("finish_reason"),
            "reasoning_tokens": call.get("reasoning_tokens"),
        }

    calls = [(model, key, k, prompt, prompt_hash) for model in models for (key, k, prompt, prompt_hash) in tasks]
    rows: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, *call): call for call in calls}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            status = row["api_error"] or (
                "ok" if row["grammar"] and row["encoding"] else "ok (sections not split)"
            )
            print(f"{row['model']}  {row['pair_id']}  k={row['k']}  {row['latency_s']}s  {status}")

    rows.sort(key=lambda r: (models.index(r["model"]), r["target"] == "high", r["k"]))
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    errors = [r for r in rows if r["api_error"]]
    unsplit = [r for r in rows if not r["api_error"] and not (r["grammar"] and r["encoding"])]
    cost = sum((r["usage"] or {}).get("cost", 0) for r in rows)
    print(f"\n{len(rows)} rows -> {out_dir / 'results.jsonl'}")
    print(f"api errors: {len(errors)}, section-split failures: {len(unsplit)}, cost: ${cost:.4f}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
