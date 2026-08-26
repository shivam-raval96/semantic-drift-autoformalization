#!/usr/bin/env python3
"""Build eval_v1: render the four SAIR evaluation subsets through the
repo pipeline into story / literal / reference-RG rows.

Per problem: parse both equations, drop vacuous problems (a zero-op law
— excluded everywhere by standing rule), render the themed story
(deterministic hash-selected theme, recorded) and the literal
description, serialize the reference rigid grammar, and verify
everything mechanically:

  - the reference RG grades "correct" with the identity transform
    against the problem's canonical forms (checkform.grade);
  - the story back-parses to the source laws (backparse);
  - the literal description back-parses to the source laws
    (backparse_literal).

Any verification failure is a bug: stop and report, do not drop.
TRUE/FALSE answers are carried as metadata only (R7). eq1_id/eq2_id are
provenance only — all identity is keyed on the equation strings.

Output: ft-experiments/eval_v1/eval_{tier}.jsonl + manifest.json.
eval_v1 is frozen once signed off; any change is eval_v2.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ftlib import (
    REPO,
    canonical,
    equation_ops,
    file_sha256,
    is_vacuous_pair,
    law_hash,
    pair_hash,
    parse_equation,
    rg_text,
    term_depth,
    variables_in_order,
    verify_rg_round_trip,
)

from backparse import backparse  # noqa: E402  (repo module, via ftlib's path)
from literalform import backparse_literal, render_description  # noqa: E402
from storyform import render_story  # noqa: E402

from config import PATHS  # noqa: E402  (stage config; paths from root registry)

HERE = Path(__file__).resolve().parent
SAIR_DIR = PATHS["sair"]
OUT_DIR = PATHS["eval_v1"]

TIERS = {
    "evaluation_normal": "normal",
    "evaluation_hard": "hard",
    "evaluation_extra_hard": "extra_hard",
    "evaluation_order5": "order5",
}


def check_backparse(story: str, e, f) -> None:
    got = backparse(story)
    if canonical(*got["habit_law"]) != canonical(*e) or canonical(
        *got["question_law"]
    ) != canonical(*f):
        raise AssertionError("story back-parse does not recover the source laws")


def check_backparse_literal(description: str, e, f) -> None:
    got = backparse_literal(description)
    if canonical(*got["habit_law"]) != canonical(*e) or canonical(
        *got["question_law"]
    ) != canonical(*f):
        raise AssertionError("literal back-parse does not recover the source laws")


def build_row(raw: dict, tier: str) -> dict:
    e = parse_equation(raw["equation1"])
    f = parse_equation(raw["equation2"])

    story, meta = render_story(raw["equation1"], raw["equation2"])
    literal, _ = render_description(raw["equation1"], raw["equation2"])
    reference_rg = rg_text(e, f)

    verify_rg_round_trip(reference_rg, e, f)
    check_backparse(story, e, f)
    check_backparse_literal(literal, e, f)

    return {
        "problem_id": raw["id"],
        "tier": tier,
        "sair_subset": f"evaluation_{tier}" if tier != "order5" else "evaluation_order5",
        "sair_index": raw["index"],
        "sair_difficulty": raw["difficulty"],
        "eq1_id": raw["eq1_id"],
        "eq2_id": raw["eq2_id"],
        "equation_e": raw["equation1"],
        "equation_f": raw["equation2"],
        "canonical_e": meta["canonical_e"],
        "canonical_f": meta["canonical_f"],
        "theme": meta["theme"],
        "palette_e": meta["palette_e"],
        "palette_f": meta["palette_f"],
        "story": story,
        "literal": literal,
        "reference_rg": reference_rg,
        "answer": raw["answer"],
        "ops_e": equation_ops(e),
        "ops_f": equation_ops(f),
        "ops_total": equation_ops(e) + equation_ops(f),
        "max_depth": max(term_depth(t) for t in (*e, *f)),
        "n_vars_e": len(variables_in_order(*e)),
        "n_vars_f": len(variables_in_order(*f)),
        "pair_hash": pair_hash(e, f),
        "law_hash_e": law_hash(*e),
        "law_hash_f": law_hash(*f),
        "split": "eval",
        "eval_version": "v1",
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "eval_version": "v1",
        "source_dataset": "SAIRfoundation/equational-theories-selected-problems",
        "generator": "ft-experiments/build_eval.py",
        "repo_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
        ).stdout.strip(),
        "tiers": {},
        "files": {},
    }

    for subset, tier in TIERS.items():
        rows_in = [
            json.loads(line)
            for line in (SAIR_DIR / f"{subset}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        kept, dropped_vacuous = [], []
        for raw in rows_in:
            e = parse_equation(raw["equation1"])
            f = parse_equation(raw["equation2"])
            if is_vacuous_pair(e, f):
                dropped_vacuous.append(raw["id"])
                continue
            kept.append(build_row(raw, tier))

        out_path = OUT_DIR / f"eval_{tier}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        answers = [row["answer"] for row in kept]
        manifest["tiers"][tier] = {
            "sair_subset": subset,
            "input_rows": len(rows_in),
            "dropped_vacuous": dropped_vacuous,
            "n": len(kept),
            # SAIR ships some grading-equivalent duplicates within a tier;
            # report per-tier results against distinct classes, not raw n.
            "distinct_pair_classes": len({row["pair_hash"] for row in kept}),
            "answer_true": sum(answers),
            "answer_false": len(answers) - sum(answers),
            "themes": sorted({row["theme"] for row in kept}),
        }
        manifest["files"][out_path.name] = file_sha256(out_path)
        print(
            f"{tier:12s} kept {len(kept)}/{len(rows_in)} "
            f"(dropped {len(dropped_vacuous)} vacuous)  "
            f"TRUE/FALSE {sum(answers)}/{len(answers) - sum(answers)}"
        )

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {OUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
