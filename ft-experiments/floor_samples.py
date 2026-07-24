#!/usr/bin/env python3
"""Floor sanity check: sample 10 'wrong' and 10 'unparseable' generations
per model from the base runs (5 story + 5 literal per bucket, seeded),
paired with the reference RG, and write runs/base-v1/floor_samples.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "base-v1"


def load(name):
    return [json.loads(l) for l in (ROOT / name / "results.jsonl").read_text().splitlines()]


def main() -> int:
    refs = {}
    for tier in ("normal", "hard", "extra_hard", "order5"):
        for line in (HERE / "eval_v1" / f"eval_{tier}.jsonl").read_text().splitlines():
            r = json.loads(line)
            refs[r["problem_id"]] = r

    rng = random.Random(3)
    lines = ["# Floor sanity samples — base runs, 10 wrong + 10 unparseable per model", ""]
    for model in ("1b", "8b"):
        lines.append(f"## {model.upper()}")
        for bucket in ("wrong", "unparseable"):
            lines.append(f"\n### {bucket}\n")
            picks = []
            for arm in ("story", "literal"):
                pool = [r for r in load(f"{model}-{arm}") if r["bucket"] == bucket]
                picks += rng.sample(pool, min(5, len(pool)))
            for r in picks:
                ref = refs[r["pair_id"]]
                err = (r["verdict"] or {}).get("error")
                lines.append(
                    f"**{r['pair_id']}** · {r['form']} · ops {r['ops_total']} · "
                    f"finish `{r['finish_reason']}`"
                    + (f" · parse error: `{err}`" if err else "")
                )
                lines.append(f"- reference: `{ref['reference_rg'].replace(chr(10), '` / `')}`")
                resp = r["response"]
                shown = resp if len(resp) <= 400 else resp[:400] + f" …[{len(resp)} chars total]"
                lines.append("```\n" + shown + "\n```")
    out = ROOT / "floor_samples.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
