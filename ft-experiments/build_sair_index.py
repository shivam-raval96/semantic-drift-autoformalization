#!/usr/bin/env python3
"""Build the SAIR disjointness index: every pair-class hash and every
individual-law-class hash across ALL nine subsets (2,669 rows).

The training corpus is gated on this index: a candidate training pair is
dropped if its pair hash appears here, or if either of its laws' hashes
does. Class keys collide under the grader's accepted symmetries
(variable renaming, side swap, consistent dualization), so nothing that
grades like a SAIR law can leak into training.

Output: ft-experiments/data/sair_index.json
"""

from __future__ import annotations

import json
from pathlib import Path

from ftlib import file_sha256, law_hash, pair_hash, parse_equation

HERE = Path(__file__).resolve().parent
SAIR_DIR = HERE / "data" / "sair"

SUBSETS = (
    "evaluation_normal",
    "evaluation_hard",
    "evaluation_extra_hard",
    "evaluation_order5",
    "normal",
    "hard",
    "hard1",
    "hard2",
    "hard3",
)


def main() -> int:
    pair_hashes, law_hashes = set(), set()
    rows_seen = 0
    per_subset = {}
    for subset in SUBSETS:
        path = SAIR_DIR / f"{subset}.jsonl"
        pairs_before, laws_before = len(pair_hashes), len(law_hashes)
        n = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            e = parse_equation(raw["equation1"])
            f = parse_equation(raw["equation2"])
            pair_hashes.add(pair_hash(e, f))
            law_hashes.add(law_hash(*e))
            law_hashes.add(law_hash(*f))
            n += 1
        rows_seen += n
        per_subset[subset] = {
            "rows": n,
            "source_sha256": file_sha256(path),
            "new_pair_hashes": len(pair_hashes) - pairs_before,
            "new_law_hashes": len(law_hashes) - laws_before,
        }
        print(f"{subset:24s} {n:5d} rows  (+{len(pair_hashes) - pairs_before} pairs, "
              f"+{len(law_hashes) - laws_before} laws)")

    index = {
        "source_dataset": "SAIRfoundation/equational-theories-selected-problems",
        "generator": "ft-experiments/build_sair_index.py",
        "rows_indexed": rows_seen,
        "distinct_pair_classes": len(pair_hashes),
        "distinct_law_classes": len(law_hashes),
        "per_subset": per_subset,
        "pair_hashes": sorted(pair_hashes),
        "law_hashes": sorted(law_hashes),
    }
    out = HERE / "data" / "sair_index.json"
    out.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    print(f"\n{rows_seen} rows -> {len(pair_hashes)} pair classes, "
          f"{len(law_hashes)} law classes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
