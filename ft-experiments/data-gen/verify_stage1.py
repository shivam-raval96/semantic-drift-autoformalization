#!/usr/bin/env python3
"""Post-generation gate for the stage-1 corpus — must pass before training.

Re-derives every claim independently of the builder (mirrors the audit
discipline of verify_artifacts.py):

  determinism  rebuild into a temp dir, assert byte-identical file shas
  balance      identify even across RG-1/2/3; validate 50/50 yes/no;
               negatives even across the three corruption types
  disjointness  train and eval share no law class and no pair class
  identify     each embedded statement parses under its claimed grammar
               and NEITHER other (gold label unambiguous)
  validate     every 'Yes' statement parses under its label's grammar;
               every 'No' statement genuinely fails that parser
  prompts      each row's prompt is exactly the frozen template around
               its statement (no wording drift)

Exits non-zero on the first failure. Run:  python3 verify_stage1.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import build_stage1 as b1
import stage1lib as s1
from ftlib import file_sha256, ftc

PATHS = ftc.PATHS
STAGE1 = PATHS["stage1"]


def load_rows(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def statement_of(row: dict) -> str:
    """Recover the embedded statement — everything after the blank line."""
    parts = row["prompt"].split("\n\n", 1)
    assert len(parts) == 2, f"prompt has no statement block: {row['prompt']!r}"
    return parts[1]


def check_prompt_wording(row: dict, statement: str) -> None:
    if row["task"] == "identify":
        expected = s1.identify_prompt(statement)
    else:
        expected = s1.validate_prompt(row["label"], statement)
    assert row["prompt"] == expected, ("prompt drift", row["prompt"])


def check_row(row: dict) -> None:
    statement = statement_of(row)
    check_prompt_wording(row, statement)
    if row["task"] == "identify":
        gk = row["grammar"]
        assert s1.RG_LABELS[row["completion"]] == gk, ("id label mismatch", row)
        assert row["completion"] == s1.GRAMMAR_TO_LABEL[gk]
        assert s1.parses_under(gk, statement), ("id not parsing own grammar", row)
        for other in s1.GRAMMAR_KEYS:
            if other != gk:
                assert not s1.parses_under(other, statement), ("id ambiguous", other, row)
    elif row["task"] == "validate":
        gk = s1.RG_LABELS[row["label"]]
        if row["polarity"] == "yes":
            assert row["completion"] == "Yes"
            assert row["corruption"] is None
            assert s1.parses_under(gk, statement), ("pos should parse", row)
        else:
            assert row["completion"] == "No"
            assert row["corruption"] in b1.NEG_TYPES, ("bad corruption", row)
            assert not s1.parses_under(gk, statement), ("neg should fail", row)
    else:
        raise AssertionError(f"unknown task {row['task']!r}")


def check_balance(rows: List[dict], quotas: Dict[str, int]) -> None:
    ident = [r for r in rows if r["task"] == "identify"]
    validate = [r for r in rows if r["task"] == "validate"]
    per_label = {l: sum(1 for r in ident if r["label"] == l) for l in s1.RG_LABELS}
    assert len(set(per_label.values())) == 1, ("identify imbalance", per_label)
    assert per_label[list(per_label)[0]] == quotas["identify"], per_label
    pos = [r for r in validate if r["polarity"] == "yes"]
    neg = [r for r in validate if r["polarity"] == "no"]
    assert len(pos) == len(neg), ("yes/no imbalance", len(pos), len(neg))
    per_corruption = {t: sum(1 for r in neg if r["corruption"] == t) for t in b1.NEG_TYPES}
    counts = per_corruption.values()
    assert max(counts) - min(counts) <= 1, ("corruption imbalance", per_corruption)


def check_determinism() -> None:
    manifest = json.loads((STAGE1 / "manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="stage1-verify-") as tmp:
        b1.main(["--out-dir", tmp])
        for name, want in manifest["files"].items():
            got = file_sha256(Path(tmp) / name)
            assert got == want, f"determinism: {name} sha {got} != committed {want}"


def main() -> int:
    train = load_rows(STAGE1 / "train.jsonl")
    eval_rows = load_rows(STAGE1 / "eval.jsonl")
    print(f"loaded {len(train)} train + {len(eval_rows)} eval rows")

    for split_name, rows, quotas in (
        ("train", train, b1.QUOTAS["train"]),
        ("eval", eval_rows, b1.QUOTAS["eval"]),
    ):
        for row in rows:
            check_row(row)
        check_balance(rows, quotas)
        print(f"  {split_name}: {len(rows)} rows — per-row correctness + balance OK")

    train_laws = {r["law_hash_e"] for r in train} | {r["law_hash_f"] for r in train}
    eval_laws = {r["law_hash_e"] for r in eval_rows} | {r["law_hash_f"] for r in eval_rows}
    assert not (train_laws & eval_laws), "train/eval law overlap"
    train_pairs = {r["pair_hash"] for r in train}
    eval_pairs = {r["pair_hash"] for r in eval_rows}
    assert not (train_pairs & eval_pairs), "train/eval pair overlap"
    print(f"  disjointness OK — {len(train_laws)} train laws, {len(eval_laws)} eval laws, "
          f"0 overlap")

    check_determinism()
    print("  determinism OK — rebuild byte-identical to committed shas")

    print("STAGE-1 CORPUS VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
