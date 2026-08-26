#!/usr/bin/env python3
"""Independently verify the frozen contrast_v1 artifact.

Recomputes every derived field from the equation strings (canonical
forms, hashes, metrics, the reference RG), re-grades both answers with
checkform, re-derives the law-connected-component partition, and checks
the manifest's counts and file hashes. Trusts nothing stored beyond the
source equation strings themselves. Any failure raises: fix the
generator, never the artifact.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from pxlib import UnionFind, ftlib, pxc

from backparse import backparse  # noqa: E402

KNOWN_THEMES = {"graft", "paint", "signal", "tea"}


def fail(msg: str) -> None:
    raise AssertionError(msg)


def main() -> int:
    out_dir = Path(pxc.PATHS["contrast_v1"])
    manifest = json.loads((out_dir / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (out_dir / "contrast.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for name in ("contrast.jsonl", "synthetic-laws.txt"):
        got = ftlib.file_sha256(out_dir / name)
        if got != manifest["files"][name]:
            fail(f"{name}: sha256 {got} != manifest {manifest['files'][name]}")

    uf = UnionFind()
    seen_pairs = set()
    perturb_types = Counter()
    perturb_targets = Counter()
    tier_counts = Counter()

    for i, row in enumerate(rows, start=1):
        rid = row["problem_id"]
        if rid != f"contrast_v1_{i:04d}":
            fail(f"row {i}: problem_id {rid} out of sequence")

        e = ftlib.parse_equation(row["equation_e"])
        f = ftlib.parse_equation(row["equation_f"])
        if ftlib.canonical(*e) != row["canonical_e"] or ftlib.canonical(*f) != row["canonical_f"]:
            fail(f"{rid}: stored canonical forms do not match the equations")

        ops_e, ops_f = ftlib.equation_ops(e), ftlib.equation_ops(f)
        if ops_e == 0 or ops_f == 0:
            fail(f"{rid}: vacuous law present")
        if (ops_e, ops_f, ops_e + ops_f) != (row["ops_e"], row["ops_f"], row["ops_total"]):
            fail(f"{rid}: ops fields do not match")
        if max(ftlib.term_depth(t) for t in (*e, *f)) != row["max_depth"]:
            fail(f"{rid}: max_depth does not match")
        if ftlib.pair_hash(e, f) != row["pair_hash"]:
            fail(f"{rid}: pair_hash does not match")
        if ftlib.law_hash(*e) != row["law_hash_e"] or ftlib.law_hash(*f) != row["law_hash_f"]:
            fail(f"{rid}: law hashes do not match")
        if row["pair_hash"] in seen_pairs:
            fail(f"{rid}: duplicate pair class")
        seen_pairs.add(row["pair_hash"])

        if row["correct_rg"] != ftlib.rg_text(e, f):
            fail(f"{rid}: correct_rg is not the reference serialization")
        ftlib.verify_rg_round_trip(row["correct_rg"], e, f)

        targets = {"canonical_e": row["canonical_e"], "canonical_f": row["canonical_f"]}
        verdict = ftlib.grade(row["wrong_rg"], targets)
        if verdict["status"] != "wrong":
            fail(f"{rid}: wrong_rg grades {verdict['status']!r}, want 'wrong'")

        c_lines = row["correct_rg"].split("\n")
        w_lines = row["wrong_rg"].split("\n")
        if len(c_lines) != 2 or len(w_lines) != 2:
            fail(f"{rid}: RG text is not exactly two lines")
        target = row["perturbation"]["target"]
        changed = 0 if target == "assume" else 1
        if w_lines[changed] == c_lines[changed]:
            fail(f"{rid}: perturbation target line is unchanged")
        if w_lines[1 - changed] != c_lines[1 - changed]:
            fail(f"{rid}: non-target line was modified")
        if "op(" not in w_lines[0] or "op(" not in w_lines[1]:
            fail(f"{rid}: zero-op surface tell in wrong_rg")

        got = backparse(row["story"])
        if ftlib.canonical(*got["habit_law"]) != row["canonical_e"] or ftlib.canonical(
            *got["question_law"]
        ) != row["canonical_f"]:
            fail(f"{rid}: story does not back-parse to the source laws")
        if row["theme"] not in KNOWN_THEMES:
            fail(f"{rid}: unknown theme {row['theme']!r}")

        uf.union(row["law_hash_e"], row["law_hash_f"])
        perturb_types[row["perturbation"]["type"]] += 1
        perturb_targets[target] += 1
        tier_counts[row["tier"]] += 1

    # Law-cc partition: stored grouping must equal the recomputed partition.
    stored_by_root: dict = {}
    for row in rows:
        root = uf.find(row["law_hash_e"])
        stored_by_root.setdefault(root, set()).add(row["group_lawcc"])
    for root, groups in stored_by_root.items():
        if len(groups) != 1:
            fail(f"law component {root}: multiple stored groups {groups}")
    if len(stored_by_root) != len({r["group_lawcc"] for r in rows}):
        fail("stored groups merge distinct law components")
    if len(stored_by_root) != manifest["n_lawcc_groups"]:
        fail("n_lawcc_groups does not match manifest")

    if len(rows) != manifest["n_problems"] or 2 * len(rows) != manifest["n_texts"]:
        fail("row counts do not match manifest")
    for tier, info in manifest["tiers"].items():
        if tier_counts[tier] != info["n"]:
            fail(f"tier {tier}: {tier_counts[tier]} rows != manifest {info['n']}")
    m_counts = manifest["perturbation"]["counts"]
    if dict(perturb_types) != {k: v for k, v in m_counts["by_type"].items() if v} and dict(
        perturb_types
    ) != m_counts["by_type"]:
        fail(f"perturbation type counts {dict(perturb_types)} != manifest")
    if dict(perturb_targets) != m_counts["by_target"]:
        fail(f"perturbation target counts {dict(perturb_targets)} != manifest")

    group_sizes = Counter(r["group_lawcc"] for r in rows)
    print(f"VERIFY PASS: {len(rows)} problems ({2 * len(rows)} texts)")
    print(f"  tiers: {dict(tier_counts)}")
    print(f"  perturbations: {dict(perturb_types)}  targets: {dict(perturb_targets)}")
    print(f"  law-cc groups: {len(group_sizes)} "
          f"(largest {max(group_sizes.values())} problems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
