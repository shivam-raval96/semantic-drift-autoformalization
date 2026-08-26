#!/usr/bin/env python3
"""Build contrast_v1: the contrastive correct/wrong dataset for probing.

Per problem: sample an implication pair (ETP or genform synthetics,
stratified by ops_total, vacuous excluded), render the themed story with
the repo pipeline (back-parse verified), serialize the reference RG as
the CORRECT answer (round-trip verified), and produce a WRONG answer as
ONE surgical AST edit of the reference (arg swap / variable substitution
/ subtree prune / subtree grow) — accepted only if checkform grades it
"wrong" (well-formed, outside the grader's symmetry orbit) and both
equations keep at least one op (no zero-op surface tells).

Labels are mechanical end to end (R1). Nothing here is model training
data — probes are monitors (R5 untouched). Deterministic: one seeded RNG
stream in fixed order; rerunning yields byte-identical files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pxlib import PERTURBATIONS, STAGE, UnionFind, ftlib, pxc

from backparse import backparse  # noqa: E402  (repo modules via ftlib's path)
from genform import check_corpus, generate_corpus  # noqa: E402
from storyform import render_story  # noqa: E402

from config import (  # noqa: E402  (stage config)
    GENFORM_BINS,
    GENFORM_PER_BIN,
    GENFORM_SEED,
    MAX_SAMPLE_ATTEMPTS_PER_PAIR,
    MAX_VARS,
    PERTURB_ATTEMPTS_PER_TYPE,
    PERTURBATION_TYPES,
    SAMPLER_SEED,
    TIER_QUOTAS,
)

Law = Tuple[str, str, object, object]  # (label, text, lhs, rhs)


def load_etp_laws() -> Tuple[List[Tuple[str, str]], str]:
    path = Path(ftlib.REPO) / "data" / "equations.txt"
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return [(f"E{n}", text) for n, text in enumerate(lines, start=1)], sha


def build_pools(
    laws: List[Tuple[str, str]], lo: int, hi: int
) -> Tuple[Dict[int, List[Law]], int]:
    pools: Dict[int, List[Law]] = {}
    vacuous = 0
    for label, text in laws:
        lhs, rhs = ftlib.parse_equation(text)
        ops = ftlib.equation_ops((lhs, rhs))
        if ops == 0:
            vacuous += 1
            continue
        if not lo <= ops <= hi:
            continue
        pools.setdefault(ops, []).append((label, text, lhs, rhs))
    return pools, vacuous


def check_story_backparse(story: str, e_pair, f_pair) -> None:
    got = backparse(story)
    if ftlib.canonical(*got["habit_law"]) != ftlib.canonical(*e_pair) or ftlib.canonical(
        *got["question_law"]
    ) != ftlib.canonical(*f_pair):
        raise AssertionError("story back-parse does not recover the source laws")


def perturb_pair(
    rng: random.Random, idx: int, e_pair, f_pair, meta: dict, stats: Dict[str, int]
) -> Optional[Tuple[str, dict, tuple]]:
    """One accepted single-edit wrong answer, or None if exhausted."""
    rot = idx % len(PERTURBATION_TYPES)
    order = PERTURBATION_TYPES[rot:] + PERTURBATION_TYPES[:rot]
    targets = {"canonical_e": meta["canonical_e"], "canonical_f": meta["canonical_f"]}
    for ptype in order:
        for _ in range(PERTURB_ATTEMPTS_PER_TYPE):
            target = "assume" if rng.random() < 0.5 else "ask"
            source = e_pair if target == "assume" else f_pair
            candidate = PERTURBATIONS[ptype](rng, source, MAX_VARS)
            if candidate is None:
                continue
            new_e, new_f = (
                (candidate, f_pair) if target == "assume" else (e_pair, candidate)
            )
            if ftlib.equation_ops(new_e) == 0 or ftlib.equation_ops(new_f) == 0:
                continue
            try:
                wrong_rg = ftlib.rg_text(new_e, new_f)
            except ValueError:
                continue  # >6 variables after the edit; resample
            verdict = ftlib.grade(wrong_rg, targets)
            if verdict["status"] == "correct":
                stats["perturb_orbit_collisions"] += 1
                continue
            assert verdict["status"] == "wrong", f"serialized RG unparseable: {wrong_rg!r}"
            return wrong_rg, {"type": ptype, "target": target}, (new_e, new_f)
    return None


def make_problem(
    rng: random.Random, idx: int, e: Law, f: Law, tier: str, source: str,
    stats: Dict[str, int],
) -> Optional[dict]:
    e_pair, f_pair = (e[2], e[3]), (f[2], f[3])
    story, meta = render_story(e[1], f[1])
    check_story_backparse(story, e_pair, f_pair)
    correct_rg = ftlib.rg_text(e_pair, f_pair)
    ftlib.verify_rg_round_trip(correct_rg, e_pair, f_pair)

    perturbed = perturb_pair(rng, idx, e_pair, f_pair, meta, stats)
    if perturbed is None:
        stats["perturb_dropped_problems"] += 1
        return None
    wrong_rg, pinfo, wrong_pair = perturbed

    return {
        "problem_id": None,  # assigned after sampling, in final order
        "tier": tier,
        "source": source,
        "e_label": e[0],
        "f_label": f[0],
        "equation_e": e[1],
        "equation_f": f[1],
        "canonical_e": meta["canonical_e"],
        "canonical_f": meta["canonical_f"],
        "theme": meta["theme"],
        "palette_e": meta["palette_e"],
        "palette_f": meta["palette_f"],
        "story": story,
        "correct_rg": correct_rg,
        "wrong_rg": wrong_rg,
        "perturbation": pinfo,
        "ops_e": ftlib.equation_ops(e_pair),
        "ops_f": ftlib.equation_ops(f_pair),
        "ops_total": ftlib.equation_ops(e_pair) + ftlib.equation_ops(f_pair),
        "max_depth": max(ftlib.term_depth(t) for t in (*e_pair, *f_pair)),
        "wrong_ops_total": ftlib.equation_ops(wrong_pair[0])
        + ftlib.equation_ops(wrong_pair[1]),
        "wrong_max_depth": max(
            ftlib.term_depth(t) for t in (*wrong_pair[0], *wrong_pair[1])
        ),
        "pair_hash": ftlib.pair_hash(e_pair, f_pair),
        "law_hash_e": ftlib.law_hash(*e_pair),
        "law_hash_f": ftlib.law_hash(*f_pair),
        "contrast_version": "v1",
    }


def sample_bin(
    rng, pools, total, quota, seen_pairs, stats, tier, source, rows
) -> int:
    splits = [
        (a, total - a)
        for a in sorted(pools)
        if (total - a) in pools and pools[a] and pools[total - a]
    ]
    if not splits:
        return 0
    got = attempts = 0
    while got < quota and attempts < MAX_SAMPLE_ATTEMPTS_PER_PAIR * quota:
        attempts += 1
        a, b = splits[rng.randrange(len(splits))]
        e = pools[a][rng.randrange(len(pools[a]))]
        f = pools[b][rng.randrange(len(pools[b]))]
        if e[0] == f[0]:
            continue
        phash = ftlib.pair_hash((e[2], e[3]), (f[2], f[3]))
        if phash in seen_pairs:
            continue
        row = make_problem(rng, len(rows), e, f, tier, source, stats)
        if row is None:
            continue
        seen_pairs.add(phash)
        rows.append(row)
        got += 1
    return got


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Generate contrast_v1.")
    cli.add_argument("--out-dir", type=Path, default=pxc.PATHS["contrast_v1"])
    args = cli.parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    etp_laws, etp_sha = load_etp_laws()
    synthetic_lines = generate_corpus(GENFORM_SEED, GENFORM_BINS, GENFORM_PER_BIN)
    check_corpus(synthetic_lines, GENFORM_BINS)
    synthetic_path = out_dir / "synthetic-laws.txt"
    synthetic_path.write_text("\n".join(synthetic_lines) + "\n", encoding="utf-8")
    gen_laws = [(f"G{n}", text) for n, text in enumerate(synthetic_lines, start=1)]

    etp_pools, etp_vacuous = build_pools(etp_laws, 1, 4)
    gen_pools, _ = build_pools(gen_laws, 5, 8)
    pool_sizes = {
        "etp": {k: len(v) for k, v in sorted(etp_pools.items())},
        "genform": {k: len(v) for k, v in sorted(gen_pools.items())},
    }
    print(f"ETP pool by per-equation ops: {pool_sizes['etp']} "
          f"({etp_vacuous} vacuous excluded)")
    print(f"genform pool by per-equation ops: {pool_sizes['genform']}")

    rng = random.Random(SAMPLER_SEED)
    seen_pairs: set = set()
    stats = {"perturb_orbit_collisions": 0, "perturb_dropped_problems": 0}
    rows: List[dict] = []
    tier_stats = {}
    for tier, (source, quotas) in TIER_QUOTAS.items():
        pools = etp_pools if source == "etp" else gen_pools
        achieved: Dict[int, int] = {}
        for total, quota in quotas.items():
            achieved[total] = sample_bin(
                rng, pools, total, quota, seen_pairs, stats, tier, source, rows
            )
        # Redistribute any shortfall to the tier's other bins, in order.
        target = sum(quotas.values())
        tier_n = sum(achieved.values())
        for total in quotas:
            if tier_n >= target:
                break
            got = sample_bin(
                rng, pools, total, target - tier_n, seen_pairs, stats, tier, source, rows
            )
            achieved[total] += got
            tier_n += got
        tier_stats[tier] = {
            "source": source,
            "quotas": quotas,
            "achieved_per_ops_total": achieved,
            "n": tier_n,
        }
        print(f"{tier:8s} {tier_n:5d} problems  per-ops_total {achieved}")
    print(f"stats: {stats}")

    for i, row in enumerate(rows, start=1):
        row["problem_id"] = f"contrast_v1_{i:04d}"

    # Law connected components -> fully law-disjoint CV groups.
    uf = UnionFind()
    for row in rows:
        uf.union(row["law_hash_e"], row["law_hash_f"])
    component_ids: Dict[str, str] = {}
    for row in rows:
        root = uf.find(row["law_hash_e"])
        component_ids.setdefault(root, f"cc-{len(component_ids):04d}")
        row["group_lawcc"] = component_ids[root]

    assert len({row["pair_hash"] for row in rows}) == len(rows)

    # Info only (no gate): overlap with eval_v1 laws / SAIR pairs, for audits.
    overlap = "unavailable"
    sair_index_path = Path(ftlib.ftc.PATHS["sair_index"])
    if sair_index_path.exists():
        sair_index = json.loads(sair_index_path.read_text())
        eval_laws = set(sair_index["eval_law_hashes"])
        sair_pairs = set(sair_index["pair_hashes"])
        overlap = {
            "rows_sharing_a_law_with_eval_v1": sum(
                1 for r in rows
                if r["law_hash_e"] in eval_laws or r["law_hash_f"] in eval_laws
            ),
            "rows_whose_pair_is_a_sair_pair": sum(
                1 for r in rows if r["pair_hash"] in sair_pairs
            ),
        }

    out_path = out_dir / "contrast.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    perturb_counts = {
        "by_type": {
            t: sum(1 for r in rows if r["perturbation"]["type"] == t)
            for t in PERTURBATION_TYPES
        },
        "by_target": {
            t: sum(1 for r in rows if r["perturbation"]["target"] == t)
            for t in ("assume", "ask")
        },
    }
    manifest = {
        "contrast_version": "v1",
        "generator": "probe-experiments/data-gen/build_contrast.py",
        "repo_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=STAGE
        ).stdout.strip(),
        "sampler_seed": SAMPLER_SEED,
        "genform": {
            "seed": GENFORM_SEED,
            "bins": [GENFORM_BINS.start, GENFORM_BINS.stop - 1],
            "per_bin": GENFORM_PER_BIN,
            "laws": len(synthetic_lines),
        },
        "etp_equations_sha256": etp_sha,
        "perturbation": {
            "types": list(PERTURBATION_TYPES),
            "attempts_per_type": PERTURB_ATTEMPTS_PER_TYPE,
            "counts": perturb_counts,
        },
        "pool_sizes": pool_sizes,
        "pool_exclusions": {"etp_laws_vacuous": etp_vacuous},
        "stats": stats,
        "tiers": tier_stats,
        "n_problems": len(rows),
        "n_texts": 2 * len(rows),
        "n_lawcc_groups": len(component_ids),
        "themes": sorted({row["theme"] for row in rows}),
        "eval_v1_overlap_info_only": overlap,
        "files": {
            "contrast.jsonl": ftlib.file_sha256(out_path),
            "synthetic-laws.txt": ftlib.file_sha256(synthetic_path),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(rows)} problems ({2 * len(rows)} texts), "
          f"{len(component_ids)} law-cc groups -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
