#!/usr/bin/env python3
"""Independent verification of eval_v1 and train_v1.

Deliberately re-derives everything from primary sources — the raw SAIR
JSONL files, the ETP equation list, and the committed synthetic-laws
file — rather than trusting sair_index.json or any stored row field.
Every stored field is recomputed and compared; every rendering is
re-rendered and byte-compared; the grader itself checks every reference
RG and every training sample; disjointness is re-established from raw
data. Exits nonzero on the first failure.

Checks:
  eval_v1:  file sha256s vs manifest - per-row field recomputation -
            story/literal re-render byte-identity - grade() exact on
            reference RG - story & literal back-parse to source laws -
            unique ids - kept+dropped reconciles against raw subsets -
            dropped ids are exactly the vacuous problems
  train_v1: file sha256s vs manifest - RG text well-formed, both lines
            parse under checkform's parser - first-appearance letter
            discipline - labels resolve to source laws and reproduce the
            text byte-exactly - ops/depth/hash fields recomputed -
            grade() exact vs label-source canonicals - pair-class
            uniqueness - pair gate vs all raw SAIR rows - law gate vs
            raw evaluation subsets - determinism (full rebuild into a
            scratch dir, byte-compare)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
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
)
from ftlib import RG_LETTERS

from backparse import backparse  # noqa: E402
from checkform import grade, parse_prefix_equation  # noqa: E402
from literalform import backparse_literal, render_description  # noqa: E402
from storyform import render_story, select_theme  # noqa: E402

from config import PATHS  # noqa: E402  (stage config; paths from root registry)

HERE = Path(__file__).resolve().parent
SAIR_DIR = PATHS["sair"]
EVAL_DIR = PATHS["eval_v1"]
TRAIN_DIR = PATHS["train_v1"]

EVAL_SUBSETS = {
    "normal": "evaluation_normal",
    "hard": "evaluation_hard",
    "extra_hard": "evaluation_extra_hard",
    "order5": "evaluation_order5",
}
ALL_SUBSETS = list(EVAL_SUBSETS.values()) + ["normal", "hard", "hard1", "hard2", "hard3"]

failures = 0


def check(ok: bool, message: str) -> None:
    global failures
    if not ok:
        failures += 1
        print(f"FAIL: {message}")


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def verify_eval() -> None:
    manifest = json.loads((EVAL_DIR / "manifest.json").read_text())
    for tier, subset in EVAL_SUBSETS.items():
        path = EVAL_DIR / f"eval_{tier}.jsonl"
        check(
            file_sha256(path) == manifest["files"][path.name],
            f"{path.name}: sha256 differs from manifest",
        )
        rows = load_jsonl(path)
        raw = {r["id"]: r for r in load_jsonl(SAIR_DIR / f"{subset}.jsonl")}
        tier_meta = manifest["tiers"][tier]
        check(len(rows) == tier_meta["n"], f"{tier}: row count vs manifest")

        seen_ids = set()
        n_true = 0
        for row in rows:
            pid = row["problem_id"]
            check(pid not in seen_ids, f"{tier}: duplicate id {pid}")
            seen_ids.add(pid)
            src = raw.get(pid)
            check(src is not None, f"{tier}: {pid} not in raw subset")
            check(
                row["equation_e"] == src["equation1"]
                and row["equation_f"] == src["equation2"]
                and row["answer"] == src["answer"],
                f"{tier}: {pid} raw fields differ",
            )
            e = parse_equation(row["equation_e"])
            f = parse_equation(row["equation_f"])
            check(not is_vacuous_pair(e, f), f"{tier}: {pid} vacuous but kept")
            n_true += bool(row["answer"])

            recomputed = {
                "canonical_e": canonical(*e),
                "canonical_f": canonical(*f),
                "ops_e": equation_ops(e),
                "ops_f": equation_ops(f),
                "ops_total": equation_ops(e) + equation_ops(f),
                "max_depth": max(term_depth(t) for t in (*e, *f)),
                "n_vars_e": len(variables_in_order(*e)),
                "n_vars_f": len(variables_in_order(*f)),
                "pair_hash": pair_hash(e, f),
                "law_hash_e": law_hash(*e),
                "law_hash_f": law_hash(*f),
                "theme": select_theme(row["equation_e"], row["equation_f"]),
                "reference_rg": rg_text(e, f),
            }
            for key, want in recomputed.items():
                check(row[key] == want, f"{tier}: {pid} field {key} mismatch")

            story, _ = render_story(row["equation_e"], row["equation_f"])
            literal, _ = render_description(row["equation_e"], row["equation_f"])
            check(story == row["story"], f"{tier}: {pid} story not reproducible")
            check(literal == row["literal"], f"{tier}: {pid} literal not reproducible")

            verdict = grade(
                row["reference_rg"],
                {"canonical_e": row["canonical_e"], "canonical_f": row["canonical_f"]},
            )
            check(
                verdict["status"] == "correct"
                and verdict["transform"] == {"swap_e": False, "swap_f": False, "dual": False},
                f"{tier}: {pid} reference RG does not grade exact",
            )
            bp = backparse(story)
            check(
                canonical(*bp["habit_law"]) == row["canonical_e"]
                and canonical(*bp["question_law"]) == row["canonical_f"],
                f"{tier}: {pid} story back-parse mismatch",
            )
            bpl = backparse_literal(literal)
            check(
                canonical(*bpl["habit_law"]) == row["canonical_e"]
                and canonical(*bpl["question_law"]) == row["canonical_f"],
                f"{tier}: {pid} literal back-parse mismatch",
            )

        dropped = set(tier_meta["dropped_vacuous"])
        check(
            seen_ids | dropped == set(raw) and not (seen_ids & dropped),
            f"{tier}: kept+dropped does not reconcile with raw subset",
        )
        for pid in dropped:
            e = parse_equation(raw[pid]["equation1"])
            f = parse_equation(raw[pid]["equation2"])
            check(is_vacuous_pair(e, f), f"{tier}: dropped {pid} is not vacuous")
        check(
            n_true == tier_meta["answer_true"]
            and len(rows) - n_true == tier_meta["answer_false"],
            f"{tier}: answer balance vs manifest",
        )
        print(f"eval {tier}: {len(rows)} rows verified ({len(dropped)} drops reconciled)")


RG_RE = re.compile(r"\AASSUME: (.+)\nASK: (.+)\Z")


def first_appearance_ok(pair) -> bool:
    order = variables_in_order(*pair)
    return tuple(order) == RG_LETTERS[: len(order)]


def verify_train() -> None:
    manifest = json.loads((TRAIN_DIR / "manifest.json").read_text())

    # Primary-source disjointness sets (NOT from sair_index.json).
    sair_pairs, eval_laws = set(), set()
    for subset in ALL_SUBSETS:
        for r in load_jsonl(SAIR_DIR / f"{subset}.jsonl"):
            e = parse_equation(r["equation1"])
            f = parse_equation(r["equation2"])
            sair_pairs.add(pair_hash(e, f))
            if subset.startswith("evaluation_"):
                eval_laws.add(law_hash(*e))
                eval_laws.add(law_hash(*f))

    from benchmark import load_equations

    equations, etp_sha = load_equations()
    check(etp_sha == manifest["etp_equations_sha256"], "ETP equations sha mismatch")
    synthetic = (TRAIN_DIR / "synthetic-laws.txt").read_text().splitlines()

    def law_source(label: str) -> str:
        if label.startswith("E"):
            return equations[int(label[1:]) - 1]
        if label.startswith("G"):
            return synthetic[int(label[1:]) - 1]
        raise AssertionError(f"unknown label {label}")

    seen_pairs = set()
    counts = {"train": 0, "holdout": 0}
    for name, split in (("train.jsonl", "train"), ("holdout.jsonl", "holdout")):
        path = TRAIN_DIR / name
        check(
            file_sha256(path) == manifest["files"][name],
            f"{name}: sha256 differs from manifest",
        )
        for i, row in enumerate(load_jsonl(path)):
            counts[split] += 1
            tag = f"{name}:{i}"
            check(row["split"] == split, f"{tag}: split field")
            m = RG_RE.match(row["text"])
            check(m is not None, f"{tag}: text not two clean RG lines")
            if m is None:
                continue
            e = parse_prefix_equation(m.group(1))
            f = parse_prefix_equation(m.group(2))
            check(
                first_appearance_ok(e) and first_appearance_ok(f),
                f"{tag}: variables not first-appearance x,y,z,w,u,v",
            )
            src_e = parse_equation(law_source(row["e_label"]))
            src_f = parse_equation(law_source(row["f_label"]))
            check(
                rg_text(src_e, src_f) == row["text"],
                f"{tag}: text does not reproduce from labels",
            )
            verdict = grade(
                row["text"],
                {"canonical_e": canonical(*src_e), "canonical_f": canonical(*src_f)},
            )
            check(
                verdict["status"] == "correct"
                and verdict["transform"] == {"swap_e": False, "swap_f": False, "dual": False},
                f"{tag}: does not grade exact vs label sources",
            )
            recomputed = {
                "ops_e": equation_ops(e),
                "ops_f": equation_ops(f),
                "ops_total": equation_ops(e) + equation_ops(f),
                "max_depth": max(term_depth(t) for t in (*e, *f)),
                "pair_hash": pair_hash(e, f),
                "law_hash_e": law_hash(*e),
                "law_hash_f": law_hash(*f),
            }
            for key, want in recomputed.items():
                check(row[key] == want, f"{tag}: field {key} mismatch")
            check(row["pair_hash"] not in sair_pairs, f"{tag}: SAIR pair collision")
            check(row["law_hash_e"] not in eval_laws, f"{tag}: eval law collision (e)")
            check(row["law_hash_f"] not in eval_laws, f"{tag}: eval law collision (f)")
            check(row["pair_hash"] not in seen_pairs, f"{tag}: duplicate pair class")
            seen_pairs.add(row["pair_hash"])
            lo, hi = {
                "easy": (2, 4), "medium": (5, 8), "hard": (10, 12), "holdout": (14, 16),
            }[row["tier"]]
            check(lo <= row["ops_total"] <= hi, f"{tag}: ops_total outside tier range")
            check(
                row["source"] == ("etp" if row["tier"] in ("easy", "medium") else "genform"),
                f"{tag}: source/tier mismatch",
            )
    check(counts["train"] == manifest["n_train"], "train count vs manifest")
    check(counts["holdout"] == manifest["n_holdout"], "holdout count vs manifest")
    print(f"train_v1: {counts['train']} train + {counts['holdout']} holdout rows verified")

    # Determinism: full rebuild must be byte-identical.
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, str(HERE / "build_train.py"), "--out-dir", tmp],
            capture_output=True, text=True, cwd=HERE,
        )
        check(result.returncode == 0, f"rebuild failed: {result.stderr[-500:]}")
        for name in ("train.jsonl", "holdout.jsonl", "synthetic-laws.txt"):
            check(
                (Path(tmp) / name).read_bytes() == (TRAIN_DIR / name).read_bytes(),
                f"determinism: {name} differs on rebuild",
            )
    print("train_v1: deterministic rebuild byte-identical")


def main() -> int:
    verify_eval()
    verify_train()
    if failures:
        print(f"\n{failures} FAILURES")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
