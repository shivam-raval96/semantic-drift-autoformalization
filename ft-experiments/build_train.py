#!/usr/bin/env python3
"""Build train_v1: the grammar-only fine-tuning corpus.

Pure rigid-grammar text (two lines, ASSUME/ASK, prefix op(...) with
x,y,z,w,u,v by first appearance), no stories, no instructions. Sources:

  easy    ops_total 2-4    ETP equation list (repo pipeline's source)
  medium  ops_total 5-8    ETP equation list
  hard    ops_total 10-12  genform synthetics (splits draw 5-7 ops/equation)
  holdout ops_total 14-16  genform synthetics (splits draw 6-8 ops/equation)
                           (~100 pairs, NEVER trained on - a beyond-trained-
                           LENGTH extrapolation probe: ops_total exceeds the
                           trained max 12, though tree depth overlaps training
                           and individual law classes may recur from train;
                           pair classes are always disjoint)

Disjointness (hard gate, signed off 2026-07-24): individual laws are
gated against eval_v1's law classes (any law appearing in any evaluated
problem, symmetries collapsed, is excluded from the pools), and pairs
are gated against ALL 2,669 SAIR rows' pair classes (no training pair
is any SAIR implication). Laws appearing only in the never-evaluated
SAIR subsets may feed training - they leak nothing into eval_v1.
Pair-class dedup ensures no two corpus rows are grading-equivalent.
Every emitted row's RG text is verified to grade "correct" with the
identity transform against its own source pair.

Deterministic: one seeded RNG stream in fixed order; rerunning yields
byte-identical files. Committed: this generator, the synthetic-laws
file, and the hash manifest - not the bulk JSONL (repo convention).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from ftlib import (
    Term,
    equation_ops,
    file_sha256,
    law_hash,
    pair_hash,
    parse_equation,
    rg_text,
    term_depth,
    verify_rg_round_trip,
)

from benchmark import load_equations  # noqa: E402  (repo module via ftlib path)
from genform import check_corpus, generate_corpus  # noqa: E402

HERE = Path(__file__).resolve().parent

SAMPLER_SEED = 0
GENFORM_SEED = 5
GENFORM_BINS = range(5, 9)  # 5-8 ops per equation
GENFORM_PER_BIN = 60

# tier -> (source, {ops_total: quota}); ETP tiers draw per-law ops 1-4,
# genform tiers 5-8. Quotas that a bin cannot fill (tiny 1-op pool after
# the SAIR exclusion) are redistributed within the tier and logged.
TIER_QUOTAS = {
    "easy": ("etp", {2: 334, 3: 333, 4: 333}),
    "medium": ("etp", {5: 250, 6: 250, 7: 250, 8: 250}),
    "hard": ("genform", {10: 334, 11: 333, 12: 333}),
    "holdout": ("genform", {14: 34, 15: 33, 16: 33}),
}
MAX_ATTEMPTS_PER_PAIR = 500

Law = Tuple[str, Term, Term]  # (label, lhs, rhs)


def build_pools(
    laws: List[Tuple[str, str]], sair_laws: set, lo: int, hi: int
) -> Tuple[Dict[int, List[Law]], int, int]:
    """Group laws by per-equation op count, excluding SAIR-colliding ones."""
    pools: Dict[int, List[Law]] = {}
    excluded = vacuous = 0
    for label, text in laws:
        lhs, rhs = parse_equation(text)
        ops = equation_ops((lhs, rhs))
        if ops == 0:
            vacuous += 1
            continue
        if not lo <= ops <= hi:
            continue
        if law_hash(lhs, rhs) in sair_laws:
            excluded += 1
            continue
        pools.setdefault(ops, []).append((label, lhs, rhs))
    return pools, excluded, vacuous


def make_row(e: Law, f: Law, tier: str, source: str, split: str) -> dict:
    e_pair, f_pair = (e[1], e[2]), (f[1], f[2])
    text = rg_text(e_pair, f_pair)
    verify_rg_round_trip(text, e_pair, f_pair)
    return {
        "text": text,
        "e_label": e[0],
        "f_label": f[0],
        "ops_e": equation_ops(e_pair),
        "ops_f": equation_ops(f_pair),
        "ops_total": equation_ops(e_pair) + equation_ops(f_pair),
        "max_depth": max(term_depth(t) for t in (*e_pair, *f_pair)),
        "tier": tier,
        "source": source,
        "pair_hash": pair_hash(e_pair, f_pair),
        "law_hash_e": law_hash(*e_pair),
        "law_hash_f": law_hash(*f_pair),
        "split": split,
    }


def sample_bin(
    rng: random.Random,
    pools: Dict[int, List[Law]],
    total: int,
    quota: int,
    seen_pairs: set,
    sair_pairs: set,
    drops: Dict[str, int],
    tier: str,
    source: str,
    split: str,
) -> List[dict]:
    splits = [
        (a, total - a)
        for a in sorted(pools)
        if (total - a) in pools and pools[a] and pools[total - a]
    ]
    if not splits:
        return []
    rows: List[dict] = []
    attempts = 0
    while len(rows) < quota and attempts < MAX_ATTEMPTS_PER_PAIR * quota:
        attempts += 1
        a, b = splits[rng.randrange(len(splits))]
        e = pools[a][rng.randrange(len(pools[a]))]
        f = pools[b][rng.randrange(len(pools[b]))]
        if e[0] == f[0]:
            continue
        row = make_row(e, f, tier, source, split)
        if row["pair_hash"] in sair_pairs:
            drops["sair_pair_collision"] += 1
            continue
        if row["pair_hash"] in seen_pairs:
            continue
        seen_pairs.add(row["pair_hash"])
        rows.append(row)
    return rows


def sample_tier(
    rng: random.Random,
    pools: Dict[int, List[Law]],
    quotas: Dict[int, int],
    seen_pairs: set,
    sair_pairs: set,
    drops: Dict[str, int],
    tier: str,
    source: str,
) -> Tuple[List[dict], dict]:
    split = "holdout" if tier == "holdout" else "train"
    rows: List[dict] = []
    achieved: Dict[int, int] = {}
    for total, quota in quotas.items():
        got = sample_bin(
            rng, pools, total, quota, seen_pairs, sair_pairs, drops, tier, source, split
        )
        achieved[total] = len(got)
        rows.extend(got)
    # Redistribute any shortfall to the tier's other bins, in order.
    target = sum(quotas.values())
    for total in quotas:
        missing = target - len(rows)
        if missing <= 0:
            break
        got = sample_bin(
            rng, pools, total, missing, seen_pairs, sair_pairs, drops, tier, source, split
        )
        achieved[total] += len(got)
        rows.extend(got)
    return rows, achieved


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Generate the grammar-only FT corpus.")
    cli.add_argument("--out-dir", type=Path, default=HERE / "train_v1")
    args = cli.parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sair_index = json.loads((HERE / "data" / "sair_index.json").read_text())
    eval_laws = set(sair_index["eval_law_hashes"])
    sair_pairs = set(sair_index["pair_hashes"])

    equations, etp_sha = load_equations()
    etp_laws = [(f"E{n}", text) for n, text in enumerate(equations, start=1)]

    synthetic_lines = generate_corpus(GENFORM_SEED, GENFORM_BINS, GENFORM_PER_BIN)
    check_corpus(synthetic_lines, GENFORM_BINS)
    synthetic_path = out_dir / "synthetic-laws.txt"
    synthetic_path.write_text("\n".join(synthetic_lines) + "\n", encoding="utf-8")
    gen_laws = [(f"G{n}", text) for n, text in enumerate(synthetic_lines, start=1)]

    etp_pools, etp_excluded, etp_vacuous = build_pools(etp_laws, eval_laws, 1, 4)
    gen_pools, gen_excluded, _ = build_pools(gen_laws, eval_laws, 5, 8)
    pool_sizes = {
        "etp": {k: len(v) for k, v in sorted(etp_pools.items())},
        "genform": {k: len(v) for k, v in sorted(gen_pools.items())},
    }
    print(f"ETP pool: {sum(len(v) for v in etp_pools.values())} laws by ops "
          f"{pool_sizes['etp']} ({etp_excluded} excluded via eval_v1 law hash, "
          f"{etp_vacuous} vacuous)")
    print(f"genform pool: {sum(len(v) for v in gen_pools.values())} laws by ops "
          f"{pool_sizes['genform']} ({gen_excluded} excluded via eval_v1 law hash)")

    rng = random.Random(SAMPLER_SEED)
    seen_pairs: set = set()
    drops = {"sair_pair_collision": 0}
    all_rows: List[dict] = []
    tier_stats = {}
    for tier, (source, quotas) in TIER_QUOTAS.items():
        pools = etp_pools if source == "etp" else gen_pools
        rows, achieved = sample_tier(
            rng, pools, quotas, seen_pairs, sair_pairs, drops, tier, source
        )
        tier_stats[tier] = {
            "source": source,
            "quotas": quotas,
            "achieved_per_ops_total": achieved,
            "n": len(rows),
        }
        all_rows.extend(rows)
        print(f"{tier:8s} {len(rows):5d} rows  per-ops_total {achieved}")
    print(f"sampled-pair drops: {drops}")

    # Hard gates, asserted independently of the sampler's own checks.
    for row in all_rows:
        assert row["pair_hash"] not in sair_pairs, f"SAIR pair collision: {row}"
        assert row["law_hash_e"] not in eval_laws, f"eval_v1 law collision: {row}"
        assert row["law_hash_f"] not in eval_laws, f"eval_v1 law collision: {row}"
    assert len({row["pair_hash"] for row in all_rows}) == len(all_rows)

    files = {}
    for split, name in (("train", "train.jsonl"), ("holdout", "holdout.jsonl")):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as fh:
            for row in all_rows:
                if row["split"] == split:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[name] = file_sha256(path)
    files["synthetic-laws.txt"] = file_sha256(synthetic_path)

    manifest = {
        "corpus_version": "train_v1",
        "generator": "ft-experiments/build_train.py",
        "repo_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
        ).stdout.strip(),
        "sampler_seed": SAMPLER_SEED,
        "genform": {
            "seed": GENFORM_SEED,
            "bins": [GENFORM_BINS.start, GENFORM_BINS.stop - 1],
            "per_bin": GENFORM_PER_BIN,
            "laws": len(synthetic_lines),
        },
        "etp_equations_sha256": etp_sha,
        "disjointness_gate": {
            "pair_hashes_vs": "all 9 SAIR subsets (2,669 rows)",
            "law_hashes_vs": "eval_v1 evaluation subsets only (decision 2026-07-24)",
        },
        "sair_index": {
            "rows_indexed": sair_index["rows_indexed"],
            "pair_classes": len(sair_pairs),
            "eval_law_classes": len(eval_laws),
        },
        "pool_sizes": pool_sizes,
        "pool_exclusions": {
            "etp_laws_eval_colliding": etp_excluded,
            "etp_laws_vacuous": etp_vacuous,
            "genform_laws_eval_colliding": gen_excluded,
        },
        "sampled_pair_drops": drops,
        "tiers": tier_stats,
        "n_train": sum(1 for r in all_rows if r["split"] == "train"),
        "n_holdout": sum(1 for r in all_rows if r["split"] == "holdout"),
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{manifest['n_train']} train + {manifest['n_holdout']} holdout rows "
          f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
