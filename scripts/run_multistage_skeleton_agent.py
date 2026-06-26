#!/usr/bin/env python3
"""Run the LADR multistage skeleton autoformalization experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
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
    REPO_ROOT / "LADR_all_material" / "generated" / "lean_statement_multistage_skeleton.jsonl"
)
DEFAULT_LEAN_PROJECT = REPO_ROOT / "lean_checker"

CONDITION = "multistage_skeleton"
STAGE1 = "stage1_skeleton"
STAGE2 = "stage2_final"
PROMPT_VERSION = "ladr_multistage_skeleton_agent_v1"

SYSTEM_PROMPTS = {
    STAGE1: (
        "Return only one Lean 4 theorem declaration. Do not use Markdown or "
        "explanations outside the Lean code. Lean comments inside the theorem "
        "proof are allowed."
    ),
    STAGE2: (
        "Return only one Lean 4 theorem declaration. Do not use Markdown, "
        "comments, explanations, or proof skeletons."
    ),
}

LEAN_PREAMBLE = """\
import Mathlib

set_option linter.style.header false

"""

STAGE1_INITIAL_PROMPT = """\
Given the theorem-informal proof pair below, formalize the informal proof into
a Lean 4 proof skeleton that follows the mathematical logic of the proof.

You do not need to prove any step. Use `sorry` for each unfinished proof step.
The skeleton should contain meaningful Lean proof structure such as `have`,
`suffices`, or `calc`; comments are allowed but should not be the only
structure.

Requirements:
- Name the theorem `{name}` exactly.
- Return exactly one Lean `theorem` declaration and no extra declarations.
- The code must typecheck with `import Mathlib`.
- Use a `:= by` proof block with structured `have`, `suffices`, or `calc` steps that reflect the informal proof.
- Use `sorry` holes where proof details are not completed. Full proof completion is not required.
- Do not introduce axioms, constants, namespaces, imports, definitions, or lemmas.
- Return only Lean code.

Dataset: {dataset}
Theorem dataset name: {name}
Natural-language theorem:
{nl_statement}

Informal proof:
{informal_proof}
"""

STAGE2_INITIAL_PROMPT = """\
You are given a natural-language theorem and a Lean 4 sketch proof. Use the
sketch proof only as context to infer the intended theorem statement.

Return one clean Lean 4 theorem statement, filled by `sorry`. You do not have
to prove it.

Requirements:
- Name the theorem `{name}` exactly.
- Preserve the same theorem intent as the natural-language theorem and Lean 4 sketch proof.
- Return exactly one Lean `theorem` declaration and no extra declarations.
- The code must typecheck with `import Mathlib`.
- End the declaration with exactly `:= by sorry`.
- Do not include `have`, `suffices`, `calc`, proof-plan comments, or any proof skeleton.
- Do not introduce axioms, constants, namespaces, imports, definitions, or lemmas.
- Return only Lean code.

Dataset: {dataset}
Theorem dataset name: {name}
Natural-language theorem:
{nl_statement}

Lean 4 sketch proof:
{stage1_skeleton}
"""

RETRY_PROMPT = """\
The previous Lean code did not pass validation or Lean typechecking.
Use the feedback below to revise the same answer.

Keep the theorem name and original output format from the initial request.
Return only the revised Lean code.

Format validation feedback:
{validation_feedback}

Lean 4 compiler feedback:
{lean_feedback}
"""


def die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def clean_model_text(text: str) -> str:
    text = text.strip()
    fence = re.fullmatch(r"```(?:lean|lean4)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def strip_lean_comments(text: str) -> str:
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    return re.sub(r"--.*?$", "", text, flags=re.MULTILINE)


def theorem_names(text: str) -> list[str]:
    scan_text = strip_lean_comments(text)
    return re.findall(r"(?m)^\s*theorem\s+([^\s:]+)", scan_text)


def theorem_or_lemma_names(text: str) -> list[str]:
    scan_text = strip_lean_comments(text)
    return re.findall(r"(?m)^\s*(?:theorem|lemma)\s+([^\s:]+)", scan_text)


def contains_forbidden_declaration(text: str) -> bool:
    scan_text = strip_lean_comments(text)
    return bool(
        re.search(
            r"(?m)^\s*(?:"
            r"axiom|constant|def|abbrev|instance|example|class|structure|inductive|"
            r"namespace|section|import|open|variable|variables|universe|universes|"
            r"set_option|local|attribute|notation|opaque|mutual|macro|syntax|"
            r"#check|#eval|#print"
            r")\b",
            scan_text,
        )
    )


def has_structured_skeleton_marker(text: str) -> bool:
    return bool(re.search(r"\b(?:have|suffices|calc)\b", text))


def has_skeleton_marker(text: str) -> bool:
    return bool(re.search(r"\b(?:have|suffices|calc)\b", text) or "--" in text or "/-" in text)


def validate_stage_output(text: str, expected_name: str | None, stage: str) -> dict[str, Any]:
    theorem_declarations = theorem_names(text)
    theorem_or_lemma_declarations = theorem_or_lemma_names(text)
    actual_name = theorem_declarations[0] if len(theorem_declarations) == 1 else None
    name_matches = bool(expected_name) and actual_name == expected_name
    has_sorry = bool(re.search(r"\bsorry\b", text))
    has_by_block = ":= by" in text
    has_final_stub = ":= by sorry" in text
    ends_with_final_stub = text.strip().endswith(":= by sorry")
    has_forbidden_decl = contains_forbidden_declaration(text)
    structured_skeleton_marker = has_structured_skeleton_marker(text)
    skeleton_marker = has_skeleton_marker(text)

    notes: list[str] = []
    if len(theorem_declarations) != 1:
        notes.append(f"expected exactly one theorem declaration, found {len(theorem_declarations)}")
    if len(theorem_or_lemma_declarations) != 1:
        notes.append(
            "expected exactly one theorem/lemma-style declaration, "
            f"found {len(theorem_or_lemma_declarations)}"
        )
    if expected_name and actual_name != expected_name:
        notes.append(f"theorem name {actual_name!r} does not match {expected_name!r}")
    if has_forbidden_decl:
        notes.append("output contains a forbidden top-level declaration or command")

    if stage == STAGE1:
        if not has_by_block:
            notes.append("stage 1 skeleton must use a ':= by' proof block")
        if not has_sorry:
            notes.append("stage 1 skeleton must contain at least one sorry hole")
        if not structured_skeleton_marker:
            notes.append("stage 1 skeleton must include structured have/suffices/calc steps")
        passed = (
            len(theorem_declarations) == 1
            and len(theorem_or_lemma_declarations) == 1
            and name_matches
            and has_by_block
            and has_sorry
            and structured_skeleton_marker
            and not has_forbidden_decl
        )
    elif stage == STAGE2:
        if not has_final_stub:
            notes.append("stage 2 final theorem must contain exactly ':= by sorry'")
        if has_final_stub and not ends_with_final_stub:
            notes.append("stage 2 final theorem must end with exactly ':= by sorry'")
        if skeleton_marker:
            notes.append("stage 2 final theorem must not include proof skeleton steps or comments")
        passed = (
            len(theorem_declarations) == 1
            and len(theorem_or_lemma_declarations) == 1
            and name_matches
            and has_final_stub
            and ends_with_final_stub
            and not skeleton_marker
            and not has_forbidden_decl
        )
    else:
        raise ValueError(f"unknown stage: {stage}")

    return {
        "stage": stage,
        "theorem_declaration_count": len(theorem_declarations),
        "theorem_or_lemma_declaration_count": len(theorem_or_lemma_declarations),
        "expected_name": expected_name,
        "actual_name": actual_name,
        "name_matches": name_matches,
        "has_by_block": has_by_block,
        "has_sorry": has_sorry,
        "has_final_stub": has_final_stub,
        "ends_with_final_stub": ends_with_final_stub,
        "has_structured_skeleton_marker": structured_skeleton_marker,
        "has_skeleton_marker": skeleton_marker,
        "has_forbidden_declaration": has_forbidden_decl,
        "passed_basic_checks": passed,
        "notes": notes,
    }


def validation_feedback(validation: dict[str, Any] | None) -> str:
    if validation is None:
        return "- no validation was run"
    notes = validation.get("notes") or []
    if not notes:
        return "- basic validation passed"
    return "\n".join(f"- {note}" for note in notes)


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


def first_lean_error(messages: list[dict[str, Any]], stderr: str, stdout_raw: str) -> str:
    for message in messages:
        if message.get("severity") == "error":
            return safe_text(message.get("data"))
    return stderr or stdout_raw


def lean_feedback(lean_result: dict[str, Any] | None) -> str:
    if lean_result is None:
        return "Lean was not run."

    message_texts = [
        safe_text(message.get("data")).strip()
        for message in lean_result.get("messages") or []
        if safe_text(message.get("data")).strip()
    ]
    if message_texts:
        return "\n\n".join(message_texts)

    fallback = safe_text(
        lean_result.get("error_text")
        or lean_result.get("stderr")
        or lean_result.get("stdout_raw")
        or ""
    ).strip()
    if fallback:
        return fallback
    return "No Lean errors, but the output failed format validation."


def lean_input(output_text: str) -> str:
    return LEAN_PREAMBLE + output_text.strip() + "\n"


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
        stdout_raw = safe_text(exc.stdout)
        stderr = safe_text(exc.stderr)
        return {
            "lean_typechecked": False,
            "check_status": "timeout",
            "command": cmd,
            "returncode": None,
            "timeout_seconds": timeout,
            "messages": [],
            "stdout_raw": stdout_raw,
            "stderr": stderr,
            "error_text": f"Lean timed out after {timeout} seconds.",
        }

    stdout = safe_text(proc.stdout)
    stderr = safe_text(proc.stderr)
    messages, raw_lines = parse_lean_json(stdout)
    stdout_raw = "\n".join(raw_lines)
    return {
        "lean_typechecked": proc.returncode == 0,
        "check_status": "ok" if proc.returncode == 0 else "error",
        "command": cmd,
        "returncode": proc.returncode,
        "messages": messages,
        "stdout_raw": stdout_raw,
        "stderr": stderr,
        "error_text": first_lean_error(messages, stderr, stdout_raw),
    }


def render_stage1_initial_prompt(row: dict[str, Any]) -> str:
    name = row.get("name")
    nl_statement = row.get("nl_statement")
    informal_proof = (row.get("informal_proof") or "").strip()
    if not name or not nl_statement:
        raise ValueError("input row must include name and nl_statement")
    if not informal_proof:
        raise ValueError(f"{name}: multistage_skeleton requires informal_proof")
    return STAGE1_INITIAL_PROMPT.format(
        dataset=row.get("domain") or "LADR",
        name=name,
        nl_statement=nl_statement,
        informal_proof=informal_proof,
    )


def render_stage2_initial_prompt(row: dict[str, Any], stage1_skeleton: str) -> str:
    name = row.get("name")
    nl_statement = row.get("nl_statement")
    if not name or not nl_statement:
        raise ValueError("input row must include name and nl_statement")
    return STAGE2_INITIAL_PROMPT.format(
        dataset=row.get("domain") or "LADR",
        name=name,
        nl_statement=nl_statement,
        stage1_skeleton=stage1_skeleton.strip(),
    )


def render_retry_prompt(
    *,
    validation: dict[str, Any] | None,
    lean_result: dict[str, Any] | None,
) -> str:
    return RETRY_PROMPT.format(
        validation_feedback=validation_feedback(validation),
        lean_feedback=lean_feedback(lean_result)[:6000],
    )


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


def usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "to_dict"):
        return usage.to_dict()
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def call_openai(
    client: Any,
    *,
    stage: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
) -> tuple[str, dict[str, Any] | None]:
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPTS[stage],
        "input": messages,
        "max_output_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.responses.create(**kwargs)
    return clean_model_text(extract_response_text(response)), usage_to_dict(getattr(response, "usage", None))


def run_stage(
    client: Any,
    row: dict[str, Any],
    *,
    stage: str,
    initial_prompt: str,
    max_iters: int,
    args: argparse.Namespace,
    lean_project: Path,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    messages: list[dict[str, str]] = [{"role": "user", "content": initial_prompt}]
    final_code = ""
    final_validation: dict[str, Any] | None = None
    final_lean: dict[str, Any] | None = None
    stage_error: str | None = None
    name = row["name"]

    for attempt_no in range(1, max_iters + 1):
        prompt = messages[-1]["content"]
        input_message_count = len(messages)
        attempt_record: dict[str, Any] = {
            "stage": stage,
            "attempt": attempt_no,
            "prompt": prompt,
            "input_message_count": input_message_count,
            "output_text": "",
            "output_sha256": output_hash(""),
            "usage": None,
            "validation": None,
            "lean": None,
        }

        try:
            output_text, usage = call_openai(
                client,
                stage=stage,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - record partial stage attempts.
            stage_error = str(exc)
            attempt_record["status"] = "api_error"
            attempt_record["error"] = stage_error
            attempts.append(attempt_record)
            break

        validation = validate_stage_output(output_text, name, stage)
        lean_result = run_lean(lean_input(output_text), lean_project=lean_project, timeout=args.timeout)
        passed = validation["passed_basic_checks"] and lean_result["lean_typechecked"]

        attempt_record.update(
            {
                "status": "ok" if passed else "failed",
                "output_text": output_text,
                "output_sha256": output_hash(output_text),
                "usage": usage,
                "validation": validation,
                "lean": lean_result,
            }
        )
        attempts.append(attempt_record)

        final_code = output_text
        final_validation = validation
        final_lean = lean_result
        messages.append({"role": "assistant", "content": output_text})

        if passed:
            break

        if attempt_no < max_iters:
            messages.append(
                {
                    "role": "user",
                    "content": render_retry_prompt(
                        validation=validation,
                        lean_result=lean_result,
                    ),
                }
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

    passed_stage = bool(
        final_validation
        and final_lean
        and final_validation["passed_basic_checks"]
        and final_lean["lean_typechecked"]
    )
    if stage_error:
        status = "error"
    else:
        status = "ok" if passed_stage else "failed"

    return {
        "stage": stage,
        "status": status,
        "error": stage_error,
        "max_iters": max_iters,
        "attempt_count": len(attempts),
        "final_output_text": final_code,
        "final_output_sha256": output_hash(final_code) if final_code else None,
        "final_validation": final_validation,
        "final_lean": final_lean,
        "lean_typechecked": bool(final_lean and final_lean["lean_typechecked"]),
        "message_history": messages,
        "attempts": attempts,
    }


def skipped_stage(stage: str, reason: str, max_iters: int) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "skipped",
        "skip_reason": reason,
        "max_iters": max_iters,
        "attempt_count": 0,
        "final_output_text": "",
        "final_output_sha256": None,
        "final_validation": None,
        "final_lean": None,
        "lean_typechecked": False,
        "message_history": [],
        "attempts": [],
    }


def run_multistage_job(
    client: Any,
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    lean_project: Path,
) -> dict[str, Any]:
    stage1_prompt = render_stage1_initial_prompt(row)
    stage1_record = run_stage(
        client,
        row,
        stage=STAGE1,
        initial_prompt=stage1_prompt,
        max_iters=args.stage1_max_iters,
        args=args,
        lean_project=lean_project,
    )

    if stage1_record["status"] == "ok":
        stage2_prompt = render_stage2_initial_prompt(row, stage1_record["final_output_text"])
        stage2_record = run_stage(
            client,
            row,
            stage=STAGE2,
            initial_prompt=stage2_prompt,
            max_iters=args.stage2_max_iters,
            args=args,
            lean_project=lean_project,
        )
    else:
        stage2_record = skipped_stage(
            STAGE2,
            f"{STAGE1} did not complete successfully",
            args.stage2_max_iters,
        )

    stage_statuses = {
        STAGE1: stage1_record["status"],
        STAGE2: stage2_record["status"],
    }
    if stage1_record["status"] == "error" or stage2_record["status"] == "error":
        status = "error"
    elif stage1_record["status"] == "ok" and stage2_record["status"] == "ok":
        status = "ok"
    else:
        status = "failed"

    final_output = stage2_record["final_output_text"] if stage2_record["status"] == "ok" else ""
    skeleton_output = stage1_record["final_output_text"]
    return {
        "created_at": iso_now(),
        "prompt_version": PROMPT_VERSION,
        "condition": CONDITION,
        "theorem_dataset_name": row.get("name"),
        "dataset": row.get("domain") or "LADR",
        "input_row": row,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stage1_max_iters": args.stage1_max_iters,
        "stage2_max_iters": args.stage2_max_iters,
        "timeout": args.timeout,
        "status": status,
        "stage_statuses": stage_statuses,
        "lean_typechecked": bool(
            stage2_record.get("final_lean")
            and stage2_record["final_lean"].get("lean_typechecked")
        ),
        "skeleton_output_text": skeleton_output,
        "skeleton_output_sha256": output_hash(skeleton_output) if skeleton_output else None,
        "final_output_text": final_output,
        "final_output_sha256": output_hash(final_output) if final_output else None,
        "attempt_count": stage1_record["attempt_count"] + stage2_record["attempt_count"],
        "stages": {
            STAGE1: stage1_record,
            STAGE2: stage2_record,
        },
    }


def completed_theorem_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for record in load_jsonl(path):
        if record.get("condition") != CONDITION:
            continue
        if record.get("status") not in {"ok", "failed"}:
            continue
        name = record.get("theorem_dataset_name") or record.get("name")
        if name:
            done.add(str(name))
    return done


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env", override=True)


def make_client() -> Any:
    if OpenAI is None:
        die("Missing dependency: openai. Install dependencies with `pip install -r requirements.txt`.")
    if not os.environ.get("OPENAI_API_KEY"):
        extra = ""
        if load_dotenv is None:
            extra = " python-dotenv is also missing, so .env files cannot be loaded."
        die("OPENAI_API_KEY is not set. Export it or add it to repo `.env`." + extra)
    return OpenAI()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LADR multistage skeleton Lean generation experiment."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lean-project", type=Path, default=DEFAULT_LEAN_PROJECT)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--stage1-max-iters", type=int, default=3)
    parser.add_argument("--stage2-max-iters", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.stage1_max_iters < 1:
        die("--stage1-max-iters must be at least 1")
    if args.stage2_max_iters < 1:
        die("--stage2-max-iters must be at least 1")
    if args.max_tokens < 1:
        die("--max-tokens must be at least 1")
    if args.timeout <= 0:
        die("--timeout must be positive")
    if args.sleep < 0:
        die("--sleep must be non-negative")
    if args.limit is not None and args.limit < 0:
        die("--limit must be non-negative")


def dry_run(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No input rows selected.")
        return

    row = rows[0]
    print(f"\n--- dry run Stage 1 initial prompt: {row.get('name')} ---")
    print(render_stage1_initial_prompt(row))
    placeholder_skeleton = f"""\
theorem {row["name"]} : True := by
  -- A compiling Lean sketch proof would appear here.
  sorry"""
    print(f"\n--- dry run representative Stage 2 prompt template: {row.get('name')} ---")
    print(render_stage2_initial_prompt(row, placeholder_skeleton))


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    lean_project = resolve_path(args.lean_project)

    if not input_path.exists():
        die(f"Input file not found: {input_path}")
    if not lean_project.exists():
        die(f"Lean checker project not found: {lean_project}")

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Lean project: {lean_project}")
    print(f"Condition: {CONDITION}")
    print(f"Jobs: {len(rows)}")
    print(f"Stage 1 max attempts per job: {args.stage1_max_iters}")
    print(f"Stage 2 max attempts per job: {args.stage2_max_iters}")

    if args.dry_run:
        dry_run(rows)
        return

    if args.force and output_path.exists():
        output_path.unlink()

    load_environment()
    client = make_client()
    done = completed_theorem_names(output_path)
    print(f"Skipping {len(done)} completed theorem(s)")

    summary: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        name = row.get("name")
        if name in done:
            print(f"[{index}/{len(rows)}] skip {name}")
            continue

        print(f"[{index}/{len(rows)}] agent {name}")
        try:
            record = run_multistage_job(client, row, args=args, lean_project=lean_project)
        except Exception as exc:  # noqa: BLE001 - preserve a JSONL record for hard failures.
            record = {
                "created_at": iso_now(),
                "prompt_version": PROMPT_VERSION,
                "condition": CONDITION,
                "theorem_dataset_name": row.get("name"),
                "dataset": row.get("domain") or "LADR",
                "input_row": row,
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stage1_max_iters": args.stage1_max_iters,
                "stage2_max_iters": args.stage2_max_iters,
                "timeout": args.timeout,
                "status": "error",
                "stage_statuses": {},
                "lean_typechecked": False,
                "attempt_count": 0,
                "stages": {},
                "error": str(exc),
            }
            print(f"  error: {exc}", file=sys.stderr)

        append_jsonl(output_path, record)
        status = record["status"]
        summary[status] += 1
        print(f"  {status} after {record.get('attempt_count')} attempt(s)")
        if args.sleep > 0:
            time.sleep(args.sleep)

    print("Done.")
    print("New jobs:", dict(summary))


if __name__ == "__main__":
    main()
