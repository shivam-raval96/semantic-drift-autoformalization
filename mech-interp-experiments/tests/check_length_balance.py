#!/usr/bin/env python3
"""Report text length across the depth grid, and fail if it is not flat.

The unit tests assert that length is constant; this prints the table behind
that assertion, which is what you want when a family is being changed or a new
one proposed. It renders a depth-stratified sample in all six surface forms and
reports word count per cell, exiting non-zero unless every (form, family) holds
a single word count across all of its depths.

    cd mech-interp-experiments && python3 tests/check_length_balance.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.dataset import (  # noqa: E402
    DEFAULT_CELLS,
    build_pool,
    parse_cells,
    render_all_forms,
    sample_depth_balanced,
)
from shared.stats import pearson, stdev  # noqa: E402

Family = Tuple[int, int, int]  # minor ops, major ops, variables
Row = Tuple[Family, int, int]  # family, depth, word count


def collect(samples: List[dict]) -> Dict[str, List[Row]]:
    """Per surface form, one row per sampled pair."""
    rows: Dict[str, List[Row]] = {}
    for sample in samples:
        minor, major, variables, depth = sample["shape"]
        for surface, text in render_all_forms(sample["metadata"]).items():
            rows.setdefault(surface, []).append(
                ((minor, major, variables), depth, len(text.split()))
            )
    return rows


def report(rows: Dict[str, List[Row]]) -> None:
    for surface in sorted(rows):
        data = rows[surface]
        print("\n{}".format(surface))
        print(
            "  {:>6} {:>6} {:>5} {:>6} {:>4} {:>9} {:>7}".format(
                "minor", "major", "vars", "depth", "n", "words", "sd"
            )
        )
        by_family: Dict[Family, List[Row]] = {}
        for row in data:
            by_family.setdefault(row[0], []).append(row)
        for family in sorted(by_family):
            group = by_family[family]
            by_depth: Dict[int, List[int]] = {}
            for _, depth, words in group:
                by_depth.setdefault(depth, []).append(words)
            for depth in sorted(by_depth):
                counts = by_depth[depth]
                print(
                    "  {:>6} {:>6} {:>5} {:>6} {:>4} {:>9.1f} {:>7.1f}".format(
                        family[0], family[1], family[2], depth, len(counts),
                        sum(counts) / len(counts), stdev(counts),
                    )
                )
            r = pearson([row[1] for row in group], [row[2] for row in group])
            label = "constant depth" if r is None else "{:+.3f}".format(r)
            print("    correlation of words with depth = {}".format(label))
        r_ops = pearson(
            [row[0][0] + row[0][1] for row in data], [row[2] for row in data]
        )
        label = "single operation count" if r_ops is None else "{:+.3f}".format(r_ops)
        print("    correlation of words with operations = {}".format(label))


def flat(rows: Dict[str, List[Row]]) -> bool:
    """True when every (form, family) holds one word count at every depth."""
    verdict = True
    for surface in sorted(rows):
        by_family: Dict[Family, set] = {}
        for family, _, words in rows[surface]:
            by_family.setdefault(family, set()).add(words)
        for family in sorted(by_family):
            counts = by_family[family]
            if len(counts) > 1:
                verdict = False
                print(
                    "\nFAIL {} {}: word counts {}".format(
                        surface, family, sorted(counts)
                    )
                )
    return verdict


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument(
        "--cells",
        type=parse_cells,
        default=parse_cells(DEFAULT_CELLS),
        metavar="MINOR:MAJOR:VARS:DEPTH,...",
        help="grid to check (default: {})".format(DEFAULT_CELLS),
    )
    cli.add_argument("--per-cell", type=int, default=40, help="pairs per cell")
    cli.add_argument("--seed", type=int, default=0)
    args = cli.parse_args(argv)

    samples = sample_depth_balanced(
        build_pool(), args.per_cell, args.seed, cells=args.cells
    )
    print("{} pairs x 6 forms = {} texts".format(len(samples), len(samples) * 6))
    rows = collect(samples)
    report(rows)
    if not flat(rows):
        return 1
    print("\nPASS: every form holds one word count across its family's depths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
