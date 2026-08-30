#!/usr/bin/env python3
"""Independent gate for the stage-2 training sets.

Checks counts, theme filtering, pair coverage, and that every completion
parses under its target grammar and grades exact. Prints STAGE2-TRAIN
VERIFIED or raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import stage1lib as s1
from ftlib import ftc

PATHS = ftc.PATHS
TRAIN_V2 = PATHS["train_v2"]
OUT = PATHS["stage2"]


def load(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def train_v2_rows() -> list:
    return load(TRAIN_V2 / "train.jsonl")


def check_grades(rows: list, key: str) -> None:
    for r in rows:
        v = s1.grammars.grade_b(r["completion"], key, r["canonical_e"], r["canonical_f"])
        assert v["status"] == "correct", ("completion not exact", key, r["pair_hash"], v)
        assert not any(s1.parses_under(o, r["completion"])
                       for o in s1.GRAMMARS if o != key), \
            ("cross-parse", key, r["pair_hash"])


def main() -> int:
    base = train_v2_rows()
    base_pairs = {r["pair_hash"] for r in base}
    manifest = json.loads((OUT / "manifest_train.json").read_text())

    rg2 = load(OUT / "train_v2_rg2.jsonl")
    assert len(rg2) == len(base), (len(rg2), len(base))
    assert {r["pair_hash"] for r in rg2} == base_pairs, "rg2 pair set != train_v2"
    assert "tea" not in {r["theme"] for r in rg2}, "tea leaked into rg2"
    assert all(r["grammar"] == "b_near" for r in rg2), "rg2 grammar tag wrong"
    check_grades(rg2, "b_near")

    single = load(OUT / "train_v2_rg1_singletheme.jsonl")
    assert len(single) == len(base), (len(single), len(base))
    themes = {r["theme"] for r in single}
    assert len(themes) == 1, ("single-theme not single", themes)
    theme = themes.pop()
    assert theme == manifest["singletheme"]["theme"], (theme, manifest["singletheme"]["theme"])
    distinct = {r["pair_hash"] for r in single}
    assert len(distinct) == manifest["singletheme"]["pool"], \
        ("pool size drift", len(distinct), manifest["singletheme"]["pool"])
    assert distinct <= base_pairs, "single-theme pairs not from train_v2"
    check_grades(single, "a")

    print(f"= train_v2_rg2.jsonl            {len(rg2)} rows, RG-2, exact, tea held out")
    print(f"= train_v2_rg1_singletheme.jsonl {len(single)} rows, theme={theme}, "
          f"{len(distinct)} distinct pairs upsampled")
    print("STAGE2-TRAIN VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
