#!/usr/bin/env python3
"""Assemble the Phase 3 base table from runs/base-v1/*/summary.json.

Per model x arm x tier: three-way verdicts (correct / wrong / unparseable,
with the correct split into exact/swapped/dualized), N, runtime. Markdown
to stdout; also writes runs/base-v1/base_table.md.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "base-v1"
TIERS = ("normal", "hard", "extra_hard", "order5")
MODEL_ORDER = ("1b", "8b", "70b")
ARMS = ("story", "literal")


def main() -> int:
    lines = ["# Base evals (Phase 3) — eval_v1, no-think, greedy, bf16 via vLLM/Modal", ""]
    lines.append(
        "| model | arm | tier | N | correct% | exact | swap | dual | wrong | unparse | unparse% | len-cap |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in MODEL_ORDER:
        for arm in ARMS:
            run = ROOT / f"{model}-{arm}"
            if not (run / "summary.json").exists():
                continue
            meta = json.loads((run / "run_meta.json").read_text())
            summary = json.loads((run / "summary.json").read_text())
            for tier in TIERS:
                s = summary.get(tier)
                if not s:
                    continue
                lines.append(
                    f"| {meta['model'].split('/')[-1]} | {arm} | {tier} | {s['n']} "
                    f"| {s['correct_pct']} | {s['exact']} | {s['correct-swapped']} "
                    f"| {s['correct-dualized']} | {s['wrong']} | {s['unparseable']} "
                    f"| {s['unparseable_pct']} | {s['length_finishes']} |"
                )
            t = meta.get("timing", {})
            lines.append(
                f"| _{meta['model'].split('/')[-1]} · {arm}: wall {t.get('wall_s')}s, "
                f"gen {t.get('generate_s')}s, {meta['backend'].get('gpu', '?')}_ "
                + "| " * 11 + "|"
            )
    text = "\n".join(lines) + "\n"
    (ROOT / "base_table.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
