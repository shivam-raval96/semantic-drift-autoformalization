#!/usr/bin/env python3
"""Base-vs-FT comparison table (Phase 5a headline).

Reads runs/base-v1/8b-*/summary.json and runs/ft-v1/8b-*/summary.json
and emits the three-way comparison per arm x tier. Writes
runs/ft-v1/comparison.md and prints it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ft_root_config", Path(__file__).resolve().parents[1] / "config.py"
)
ftc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ftc)

TIERS = ftc.EVAL["tiers"]
ARMS = ftc.EVAL["arms"]


def load(tag: str, arm: str):
    path = ftc.PATHS["runs"] / tag / f"8b-{arm}" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    lines = [
        "# Base vs grammar-FT — Llama-3.1-8B, eval_v1, all arms",
        "",
        "| arm | tier | N | correct% base→FT | Δ | unparse% base→FT | Δ | wrong base→FT |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        base, ft = load("base-v1", arm), load("ft-v1", arm)
        if not base or not ft:
            lines.append(f"| {arm} | _missing run_ |" + " |" * 6)
            continue
        for tier in TIERS:
            b, f = base[tier], ft[tier]
            dc = round(f["correct_pct"] - b["correct_pct"], 1)
            du = round(f["unparseable_pct"] - b["unparseable_pct"], 1)
            lines.append(
                f"| {arm} | {tier} | {b['n']} "
                f"| {b['correct_pct']} → {f['correct_pct']} | {dc:+} "
                f"| {b['unparseable_pct']} → {f['unparseable_pct']} | {du:+} "
                f"| {b['wrong']} → {f['wrong']} |"
            )
    text = "\n".join(lines) + "\n"
    out = ftc.PATHS["runs"] / "ft-v1" / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
