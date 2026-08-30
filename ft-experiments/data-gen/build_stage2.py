#!/usr/bin/env python3
"""Build the stage-2 translation training sets.

T1 and D0 reuse train_v2/train.jsonl as-is (story -> RG-1, tea held out,
2,772 pairs). This script derives the two variants the other arms need:

  train_v2_rg2.jsonl              the same pairs, completion re-rendered
                                  into RG-2 (GIVEN/SHOW), for the T2
                                  A-then-B continuation.
  train_v2_rg1_singletheme.jsonl  a single theme (default signal) upsampled
                                  to the full train_v2 count, for T3 (one
                                  theme vs many, matched example budget).

Completions are re-rendered from the grammar-A source via eval/grammars.py
and each is verified to grade exact. Deterministic. Writes into stage2/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import stage1lib as s1
from ftlib import ftc, file_sha256

PATHS = ftc.PATHS
TRAIN_V2 = PATHS["train_v2"]
OUT = PATHS["stage2"]


def load_train_v2() -> List[dict]:
    return [json.loads(l) for l in (TRAIN_V2 / "train.jsonl").read_text().splitlines()
            if l.strip()]


def render_rg2(row: dict) -> dict:
    e, f = s1.pair_from_completion(row["completion"])
    out = dict(row)
    out["completion"] = s1.serialize("b_near", e, f)
    out["grammar"] = "b_near"
    out["label"] = "RG-2"
    return out


def build_rg2(rows: List[dict]) -> List[dict]:
    built = [render_rg2(r) for r in rows]
    for r in built:
        v = s1.grammars.grade_b(r["completion"], "b_near", r["canonical_e"], r["canonical_f"])
        assert v["status"] == "correct", ("rg2 completion not exact", r["pair_hash"], v)
    return built


def build_singletheme(rows: List[dict], theme: str, target: int) -> Tuple[List[dict], int]:
    pool = sorted([r for r in rows if r["theme"] == theme], key=lambda r: r["pair_hash"])
    assert pool, f"no train_v2 rows for theme {theme!r}"
    built = []
    for i in range(target):
        src = dict(pool[i % len(pool)])
        src["grammar"] = "a"
        src["label"] = "RG-1"
        src["upsample_rank"] = i
        built.append(src)
    for r in built:
        v = s1.grammars.grade_b(r["completion"], "a", r["canonical_e"], r["canonical_f"])
        assert v["status"] == "correct", ("singletheme completion not exact", r["pair_hash"], v)
    return built, len(pool)


def write_jsonl(path: Path, rows: List[dict]) -> str:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return file_sha256(path)


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Build stage-2 translation training sets.")
    cli.add_argument("--theme", default="signal", help="theme kept for the single-theme (T3) set")
    cli.add_argument("--out-dir", type=Path, default=OUT)
    args = cli.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    rows = load_train_v2()
    target = len(rows)
    rg2 = build_rg2(rows)
    single, pool_n = build_singletheme(rows, args.theme, target)

    files = {
        "train_v2_rg2.jsonl": write_jsonl(out / "train_v2_rg2.jsonl", rg2),
        "train_v2_rg1_singletheme.jsonl":
            write_jsonl(out / "train_v2_rg1_singletheme.jsonl", single),
    }

    manifest = {
        "corpus_version": "stage2_train",
        "generator": "ft-experiments/data-gen/build_stage2.py",
        "source": "train_v2/train.jsonl",
        "train_v2_train_sha256": file_sha256(TRAIN_V2 / "train.jsonl"),
        "rg1_reuses": "train_v2/train.jsonl (no re-render; used as-is for T1/D0)",
        "rg2": {"n": len(rg2), "grammar": "b_near", "label": "RG-2"},
        "singletheme": {
            "theme": args.theme, "pool": pool_n,
            "upsampled_to": target, "grammar": "a", "label": "RG-1",
        },
        "files": files,
    }
    (out / "manifest_train.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"train_v2_rg2.jsonl            {len(rg2)} rows (RG-2 completions)")
    print(f"train_v2_rg1_singletheme.jsonl {len(single)} rows "
          f"(theme={args.theme}, pool {pool_n} upsampled to {target})")
    print(f"manifest -> {out / 'manifest_train.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
