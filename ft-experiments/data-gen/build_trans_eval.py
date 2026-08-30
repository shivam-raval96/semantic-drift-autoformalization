#!/usr/bin/env python3
"""Build trans_eval_v1: the frozen translation eval set.

The 777 eval_v1 problems, each rendered into all four rigid grammars
(RG-1..RG-4) so one artifact serves every translation eval: stage-1
familiarity translation (RG-1/2/3), stage-2 in-grammar (RG-1) and the
held-out grammar (RG-4), and theme generalization (the tea subset).
References come from eval/grammars.py; canonical_e/f ride along from
eval_v1 for grading. Deterministic and law-disjoint from train_v2 by
construction (eval_v1 already is). Writes trans_eval_v1/eval.jsonl +
manifest.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import stage1lib as s1
from ftlib import ftc, file_sha256

PATHS = ftc.PATHS
EVAL_V1 = PATHS["eval_v1"]
TIERS = ("normal", "hard", "extra_hard", "order5")

# label -> grammar key (RG-1=a, RG-2=b_near, RG-3=b_far, RG-4=sexpr)
LABEL_TO_KEY = dict(s1.RG_LABELS)
REF_FIELD = {"RG-1": "ref_rg1", "RG-2": "ref_rg2", "RG-3": "ref_rg3", "RG-4": "ref_rg4"}

CARRY = ("problem_id", "tier", "theme", "story", "literal",
         "canonical_e", "canonical_f", "pair_hash",
         "law_hash_e", "law_hash_f", "ops_total", "max_depth")


def load_eval_rows() -> List[dict]:
    rows: List[dict] = []
    for tier in TIERS:
        for line in (EVAL_V1 / f"eval_{tier}.jsonl").read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_row(r: dict) -> dict:
    e, f = s1.pair_from_completion(r["reference_rg"])
    out = {k: r[k] for k in CARRY}
    for label, key in LABEL_TO_KEY.items():
        ref = s1.serialize(key, e, f)
        # every reference must grade exact against its own source pair
        verdict = s1.grammars.grade_b(ref, key, r["canonical_e"], r["canonical_f"])
        assert verdict["status"] == "correct", (label, verdict, ref)
        out[REF_FIELD[label]] = ref
    return out


def _tally(rows: List[dict], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(r[key]) for r in rows).items()))


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Build the frozen translation eval set.")
    cli.add_argument("--out-dir", type=Path, default=PATHS["trans_eval_v1"])
    args = cli.parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows()
    built = [build_row(r) for r in rows]

    path = out_dir / "eval.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in built:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "corpus_version": "trans_eval_v1",
        "generator": "ft-experiments/data-gen/build_trans_eval.py",
        "purpose": "translation eval: eval_v1 problems rendered into all four RGs",
        "source": "eval_v1 (777 problems, law-disjoint from train_v2)",
        "labels": LABEL_TO_KEY,
        "grammar_family": s1.GRAMMAR_FAMILY,
        "ref_fields": REF_FIELD,
        "eval_v1_files_sha256": {
            f"eval_{t}.jsonl": file_sha256(EVAL_V1 / f"eval_{t}.jsonl") for t in TIERS
        },
        "n": len(built),
        "by_tier": _tally(built, "tier"),
        "by_theme": _tally(built, "theme"),
        "files": {"eval.jsonl": file_sha256(path)},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"trans_eval_v1: {len(built)} rows -> {path}")
    print(f"  by tier  {manifest['by_tier']}")
    print(f"  by theme {manifest['by_theme']}")
    print(f"manifest -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
