#!/usr/bin/env python3
"""Typecheck generated Lean declarations with the local mathlib Lake project."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "LADR_all_material" / "generated" / "lean_statement_pilot_ab.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT / "LADR_all_material" / "generated" / "lean_statement_pilot_ab_checked.jsonl"
)
DEFAULT_LEAN_PROJECT = REPO_ROOT / "lean_checker"
LEAN_PREAMBLE = """\
import Mathlib

set_option linter.style.header false

"""


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                die(f"{path}:{line_no}: invalid JSON: {exc}")
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def output_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_key(record: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (
        record.get("theorem_dataset_name") or record.get("name"),
        record.get("condition"),
        record.get("output_sha256"),
    )


def completed_checks(path: Path) -> set[tuple[str | None, str | None, str | None]]:
    if not path.exists():
        return set()
    done: set[tuple[str | None, str | None, str | None]] = set()
    for record in load_jsonl(path):
        done.add(check_key(record))
    return done


def lean_input(output_text: str) -> str:
    return LEAN_PREAMBLE + output_text.strip() + "\n"


def parse_lean_json(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    messages: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            raw_lines.append(line)
            continue
        if isinstance(parsed, dict):
            messages.append(parsed)
        else:
            raw_lines.append(line)
    return messages, raw_lines


def run_lean(code: str, *, lean_project: Path, timeout: float) -> dict[str, Any]:
    cmd = ["lake", "env", "lean", "--stdin", "--json"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=lean_project,
            input=code,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "lean_typechecked": False,
            "check_status": "timeout",
            "returncode": None,
            "timeout_seconds": timeout,
            "messages": [],
            "stdout_raw": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    messages, raw_lines = parse_lean_json(proc.stdout)
    return {
        "lean_typechecked": proc.returncode == 0,
        "check_status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "messages": messages,
        "stdout_raw": "\n".join(raw_lines),
        "stderr": proc.stderr,
    }


def source_job(record: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    if record.get("status") != "ok":
        return None
    validation = record.get("validation") or {}
    if not validation.get("passed_basic_checks"):
        return None
    output_text = (record.get("output_text") or "").strip()
    if not output_text:
        return None
    return {
        "source_index": source_index,
        "theorem_dataset_name": record.get("theorem_dataset_name") or record.get("name"),
        "condition": record.get("condition"),
        "model": record.get("model"),
        "prompt_version": record.get("prompt_version"),
        "output_sha256": output_hash(output_text),
        "output_text": output_text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Typecheck generated Lean JSONL records with lake env lean."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lean-project", type=Path, default=DEFAULT_LEAN_PROJECT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the checked output file instead of resuming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    lean_project = (
        args.lean_project if args.lean_project.is_absolute() else REPO_ROOT / args.lean_project
    )

    if not input_path.exists():
        die(f"Input file not found: {input_path}")
    if not lean_project.exists():
        die(f"Lean checker project not found: {lean_project}")

    if args.force and output_path.exists():
        output_path.unlink()

    source_rows = load_jsonl(input_path)
    jobs = [
        job
        for index, record in enumerate(source_rows, start=1)
        if (job := source_job(record, index)) is not None
    ]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    done = completed_checks(output_path)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Lean project: {lean_project}")
    print(f"Candidate valid records: {len(jobs)}")
    print(f"Skipping {len(done)} completed checks")

    summary: Counter[str] = Counter()
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)

    for index, job in enumerate(jobs, start=1):
        key = check_key(job)
        label = f"{job['theorem_dataset_name']} / {job['condition']}"
        if key in done:
            print(f"[{index}/{len(jobs)}] skip {label}")
            continue

        print(f"[{index}/{len(jobs)}] check {label}")
        result = run_lean(
            lean_input(job["output_text"]),
            lean_project=lean_project,
            timeout=args.timeout,
        )
        record = {
            "checked_at": iso_now(),
            "source_index": job["source_index"],
            "theorem_dataset_name": job["theorem_dataset_name"],
            "condition": job["condition"],
            "model": job["model"],
            "prompt_version": job["prompt_version"],
            "output_sha256": job["output_sha256"],
            "lean_preamble": LEAN_PREAMBLE.strip(),
            **result,
        }
        append_jsonl(output_path, record)
        done.add(key)

        status = record["check_status"]
        summary[status] += 1
        by_condition[str(job["condition"])][status] += 1
        if status != "ok":
            first_error = next(
                (
                    msg.get("data")
                    for msg in record["messages"]
                    if msg.get("severity") == "error"
                ),
                record["stderr"] or record["stdout_raw"],
            )
            if first_error:
                print(f"  error: {str(first_error).splitlines()[0]}")

    print("Done.")
    print("New checks:", dict(summary))
    for condition in sorted(by_condition):
        print(f"{condition}: {dict(by_condition[condition])}")


if __name__ == "__main__":
    main()
