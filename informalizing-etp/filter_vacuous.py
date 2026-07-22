#!/usr/bin/env python3
"""Copy benchmark run directories with vacuous-law pairs removed.

A pair is vacuous if either of its laws is a zero-op law (E1 ``x = x``
or E2 ``x = y``) — equivalently, if either equation contains no
operation symbol. Experiment 07 established that such pairs are a
measurement hazard (see experiments/README.md), so re-analyses of
earlier runs exclude them.

Run directories are immutable; this script never touches its inputs.
For each run it writes ``run_meta.json`` (verbatim) plus filtered
``samples.jsonl`` and ``results.jsonl`` into
``OUT_ROOT/<experiment>/runs/<run>/``, preserving the experiment
layout so both ``charts.py`` and ``paired_analysis.py`` accept the
copies.

    python3 filter_vacuous.py EXPERIMENT_OR_RUN_DIR [...] \
        [--out-root results/no-vacuous]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

OP_SYMBOLS = "◇∘*"


def is_vacuous(sample: dict) -> bool:
    metadata = sample["metadata"]
    return any(
        not any(symbol in metadata[key] for symbol in OP_SYMBOLS)
        for key in ("equation_e", "equation_f")
    )


def run_dirs(path: Path) -> List[Path]:
    if (path / "samples.jsonl").exists():
        return [path]
    if (path / "runs").is_dir():
        return sorted(d for d in (path / "runs").iterdir() if (d / "samples.jsonl").exists())
    raise SystemExit(f"{path} is neither a run directory nor an experiment directory")


def out_dir_for(run_dir: Path, out_root: Path) -> Path:
    if run_dir.parent.name == "runs":
        return out_root / run_dir.parent.parent.name / "runs" / run_dir.name
    return out_root / run_dir.name


def filter_run(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "run_meta.json", out_dir / "run_meta.json")

    kept = set()
    total = 0
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as out:
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
            sample = json.loads(line)
            total += 1
            if not is_vacuous(sample):
                kept.add(sample["pair_id"])
                out.write(line + "\n")

    rows = 0
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as out:
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
            if json.loads(line)["pair_id"] in kept:
                out.write(line + "\n")
                rows += 1

    print(f"{run_dir}: kept {len(kept)}/{total} pairs ({rows} result rows) -> {out_dir}")


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(
        description="Copy run directories with vacuous-law pairs removed."
    )
    cli.add_argument("dirs", nargs="+", type=Path, metavar="DIR",
                     help="experiment or run directories")
    cli.add_argument("--out-root", type=Path, default=Path("results/no-vacuous"))
    args = cli.parse_args(argv)

    for path in args.dirs:
        for run_dir in run_dirs(path):
            filter_run(run_dir, out_dir_for(run_dir, args.out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
