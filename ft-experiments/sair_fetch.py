#!/usr/bin/env python3
"""Download every subset of the SAIR selected-problems dataset to JSONL.

eval_v1 uses only the four evaluation_* subsets; the other five are kept
solely so the training-corpus disjointness audit covers all SAIR rows.
Every subset's only HF split is named "train" — an HF convention; nothing
from this dataset is ever trained on.
"""

from pathlib import Path

from datasets import load_dataset

DATASET = "SAIRfoundation/equational-theories-selected-problems"
SUBSETS = (
    "evaluation_normal",
    "evaluation_hard",
    "evaluation_extra_hard",
    "evaluation_order5",
    "normal",
    "hard",
    "hard1",
    "hard2",
    "hard3",
)
OUT_DIR = Path(__file__).resolve().parent / "data" / "sair"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        ds = load_dataset(DATASET, subset, split="train")
        out = OUT_DIR / f"{subset}.jsonl"
        ds.to_json(str(out))
        print(f"{subset}: {len(ds)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
