#!/usr/bin/env python3
"""Run the LADR Lean-statement pilot A/B generation.

Each theorem is processed in interleaved order:
  1. statement_only
  2. statement_plus_proof

The model is asked for exactly one Lean 4 declaration ending in `:= by sorry`.
This script records prompts and metadata; it does not typecheck Lean output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "LADR_all_material" / "LADR_pilot_27.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT / "LADR_all_material" / "generated" / "lean_statement_pilot_ab.jsonl"
)

CONDITIONS = ("statement_only", "statement_plus_proof")
PROMPT_VERSION = "ladr_statement_pilot_ab_v3"
SYSTEM_PROMPT = (
    "Return only one Lean 4 theorem declaration. Do not use Markdown, comments, "
    "explanations, or proof attempts."
)

PROMPT_TEMPLATE = """\
Formalize the LADR theorem below as one Lean 4 theorem statement.

Requirements:
- Name the theorem `{name}`.
- It should typecheck with `import Mathlib`.
- End with `:= by sorry`.
- Return only the theorem declaration.
{proof_rule}

Dataset: {dataset}
Theorem dataset name: {name}
Natural-language theorem:
{nl_statement}
{proof_block}"""


def die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_jobs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()

    done: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") != "ok":
                continue
            name = record.get("theorem_dataset_name") or record.get("name")
            condition = record.get("condition")
            if name and condition:
                done.add((name, condition))
    return done


def render_prompt(row: dict[str, Any], condition: str) -> str:
    name = row.get("name")
    nl_statement = row.get("nl_statement")
    if not name or not nl_statement:
        raise ValueError("input row must include name and nl_statement")

    proof_rule = ""
    proof_block = ""
    if condition == "statement_plus_proof":
        informal_proof = (row.get("informal_proof") or "").strip()
        if not informal_proof:
            raise ValueError(f"{name}: statement_plus_proof requires informal_proof")
        proof_rule = (
            "- Use the informal proof only to disambiguate the theorem statement; "
            "do not formalize the proof."
        )
        proof_block = f"\nInformal proof:\n{informal_proof}\n"
    elif condition != "statement_only":
        raise ValueError(f"unknown condition: {condition}")

    return PROMPT_TEMPLATE.format(
        dataset=row.get("domain") or "LADR",
        name=name,
        nl_statement=nl_statement,
        proof_rule=proof_rule,
        proof_block=proof_block,
    )


def iter_jobs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    return [(row, condition) for row in rows for condition in CONDITIONS]


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def call_openai(
    client: Any,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
) -> tuple[str, dict[str, Any] | None]:
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.responses.create(**kwargs)
    usage = getattr(response, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return extract_response_text(response), usage_dict


def validate_output(text: str, expected_name: str | None) -> dict[str, Any]:
    declaration_names = re.findall(r"(?m)^\s*(?:theorem|lemma)\s+([^\s:]+)", text)
    has_sorry_stub = ":= by sorry" in text
    actual_name = declaration_names[0] if len(declaration_names) == 1 else None
    name_matches = bool(expected_name) and actual_name == expected_name
    notes: list[str] = []
    if not has_sorry_stub:
        notes.append("missing ':= by sorry'")
    if len(declaration_names) != 1:
        notes.append(
            f"expected one theorem/lemma declaration, found {len(declaration_names)}"
        )
    elif not name_matches:
        notes.append(f"declaration name {actual_name!r} does not match {expected_name!r}")

    return {
        "has_sorry_stub": has_sorry_stub,
        "theorem_or_lemma_declaration_count": len(declaration_names),
        "expected_declaration_name": expected_name,
        "actual_declaration_name": actual_name,
        "declaration_name_matches_input": name_matches,
        "passed_basic_checks": has_sorry_stub
        and len(declaration_names) == 1
        and name_matches,
        "notes": notes,
        "lean_typechecked": False,
    }


def base_record(
    *,
    row: dict[str, Any],
    condition: str,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "created_at": iso_now(),
        "prompt_version": PROMPT_VERSION,
        "condition": condition,
        "theorem_dataset_name": row.get("name"),
        "dataset": row.get("domain") or "LADR",
        "input_row": row,
        "system_prompt": SYSTEM_PROMPT,
        "prompt": prompt,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "output_text": None,
        "validation": None,
        "status": "pending",
        "error": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interleaved LADR Lean 4 statement-only A/B pilot data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-count", type=int, default=4)
    return parser.parse_args()


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env", override=True)


def make_client() -> Any:
    if OpenAI is None:
        die(
            "Missing dependency: openai. Install dependencies with "
            "`pip install -r requirements.txt`."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        extra = ""
        if load_dotenv is None:
            extra = " python-dotenv is also missing, so .env files cannot be loaded."
        die(
            "OPENAI_API_KEY is not set. Export it in the environment or add it to "
            f"{REPO_ROOT / '.env'}." + extra
        )
    return OpenAI()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    if not input_path.exists():
        die(f"Input file not found: {input_path}")

    load_environment()
    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    jobs = iter_jobs(rows)
    if args.dry_run:
        print(f"Dry run: showing {min(args.dry_run_count, len(jobs))} rendered prompts")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        for index, (row, condition) in enumerate(jobs[: args.dry_run_count], start=1):
            prompt = render_prompt(row, condition)
            print(f"\n--- prompt {index}: {row.get('name')} / {condition} ---")
            print("SYSTEM:")
            print(SYSTEM_PROMPT)
            print("USER:")
            print(prompt)
        return

    client = make_client()
    done = completed_jobs(output_path)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Skipping {len(done)} completed records")

    for index, (row, condition) in enumerate(jobs, start=1):
        name = row.get("name")
        if (name, condition) in done:
            print(f"[{index}/{len(jobs)}] skip {name} / {condition}")
            continue

        prompt = ""
        try:
            prompt = render_prompt(row, condition)
            record = base_record(row=row, condition=condition, prompt=prompt, args=args)
            print(f"[{index}/{len(jobs)}] {name} / {condition}")
            output_text, usage = call_openai(
                client,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            record["output_text"] = output_text
            record["usage"] = usage
            record["validation"] = validate_output(output_text, row.get("name"))
            record["status"] = "ok"
            done.add((name, condition))
        except Exception as exc:  # noqa: BLE001 - keep the pilot moving per row.
            record = base_record(row=row, condition=condition, prompt=prompt, args=args)
            record["status"] = "error"
            record["error"] = str(exc)
            print(f"  error: {exc}", file=sys.stderr)

        append_jsonl(output_path, record)
        if args.sleep > 0:
            time.sleep(args.sleep)

    print("Done.")


if __name__ == "__main__":
    main()
