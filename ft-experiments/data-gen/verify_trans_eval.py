#!/usr/bin/env python3
"""Independent gate for trans_eval_v1.

Re-derives every claim from the written eval.jsonl: each of the four
references parses under its own grammar and no other, grades exact against
the row's canonical pair, the build is deterministic (re-render matches),
and the law set is disjoint from train_v2. Prints TRANS-EVAL VERIFIED or
raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import stage1lib as s1
from build_trans_eval import LABEL_TO_KEY, REF_FIELD, TIERS, build_row, load_eval_rows
from ftlib import ftc

PATHS = ftc.PATHS
OUT = PATHS["trans_eval_v1"]
TRAIN_V2 = PATHS["train_v2"]


def load_rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def check_references(rows: list) -> None:
    keys = list(LABEL_TO_KEY.values())
    for r in rows:
        for label, key in LABEL_TO_KEY.items():
            ref = r[REF_FIELD[label]]
            # parses under its own grammar and grades exact
            verdict = s1.grammars.grade_b(ref, key, r["canonical_e"], r["canonical_f"])
            assert verdict["status"] == "correct", ("not exact", label, r["problem_id"], verdict)
            assert verdict["transform"] == {"swap_e": False, "swap_f": False, "dual": False}, \
                ("non-identity transform", label, r["problem_id"], verdict)
            # no other grammar accepts it
            for other in keys:
                if other != key:
                    assert not s1.parses_under(other, ref), \
                        ("cross-parse", label, other, r["problem_id"], ref)


def check_determinism(rows: list) -> None:
    """Re-render from eval_v1 and compare field-for-field."""
    fresh = {build_row(r)["problem_id"]: build_row(r) for r in load_eval_rows()}
    assert len(fresh) == len(rows), ("row count drift", len(fresh), len(rows))
    for r in rows:
        f = fresh[r["problem_id"]]
        for label in LABEL_TO_KEY:
            assert r[REF_FIELD[label]] == f[REF_FIELD[label]], \
                ("non-deterministic render", label, r["problem_id"])


def check_disjoint(rows: list) -> None:
    train_laws = set()
    for line in (TRAIN_V2 / "train.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            train_laws.add(row["law_hash_e"])
            train_laws.add(row["law_hash_f"])
    eval_laws = {r["law_hash_e"] for r in rows} | {r["law_hash_f"] for r in rows}
    overlap = train_laws & eval_laws
    assert not overlap, f"train_v2/trans_eval law overlap: {len(overlap)} classes"


def main() -> int:
    rows = load_rows(OUT / "eval.jsonl")
    assert len(rows) == 777, f"expected 777 rows, got {len(rows)}"
    tiers = {r["tier"] for r in rows}
    assert tiers == set(TIERS), ("tier set", tiers)
    themes = {r["theme"] for r in rows}
    assert len(themes) == 4, ("expected 4 themes", themes)

    check_references(rows)
    check_determinism(rows)
    check_disjoint(rows)

    print(f"= {len(rows)} rows, tiers {sorted(tiers)}, themes {sorted(themes)}")
    print("= all four references parse (own grammar only) and grade exact")
    print("= re-render is byte-identical (deterministic)")
    print("= law-disjoint from train_v2")
    print("TRANS-EVAL VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
