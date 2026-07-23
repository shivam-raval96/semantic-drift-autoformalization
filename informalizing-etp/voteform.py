#!/usr/bin/env python3
"""Reduce repeated benchmark passes to majority-vote run directories.

Self-consistency for the formalization benchmark: the same pair set is
run K times at nonzero temperature (one benchmark.py run directory per
pass), and each (pair, model)'s final answer is the most popular
formalization among its passes. Answers are pooled by their grading
equivalence class (checkform.answer_class_key) — the orbit under the
eight accepted transforms — so a law's swap/dual variants vote together,
exactly as grade() treats them.

Pass directories are immutable; this script never touches its inputs.
For each requested k it writes a synthetic run directory
``OUT_ROOT/<stem>-vote<k>`` with the standard schema (run_meta.json,
samples.jsonl, results.jsonl, summary.json, summary.md), where each
results row is the earliest pass's row from the winning class plus a
``vote`` provenance field, so charts.py and paired_analysis.py accept
the output unchanged. vote@k uses the *prefix* passes 1..k.

Voting rules: unparseable and api-error passes abstain; ties go to the
class first seen in the earliest pass; if every pass abstains, the
earliest graded row (else the earliest api-error row) is emitted as-is.
The representative row keeps its own verdict and bucket — correctness
is class-invariant, so correct rates are unaffected, but the
exact/swapped split reflects the representative pass.

``--baseline RUN_DIR`` additionally writes ``OUT_ROOT/<stem>-t0-baseline``,
a copy of a reference run over the same pairs restricted to the passes'
models, labeled for side-by-side charting.

    python3 voteform.py PASS_DIR [PASS_DIR ...] --ks 1,3,5,7 \
        --out-root experiments/11-majority-voting/runs \
        [--name STEM] [--baseline RUN_DIR] [--baseline-label LABEL]
"""

import argparse
import copy
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark import aggregate, summary_table
from checkform import answer_class_key

RowKey = Tuple[str, str]  # (pair_id, model)


def load_pass(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: Dict[RowKey, dict] = {}
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # retries append; the last graded row per (pair, model) wins
        if row["bucket"] != "api-error" or (row["pair_id"], row["model"]) not in rows:
            rows[(row["pair_id"], row["model"])] = row
    return {"dir": run_dir, "meta": meta, "samples": samples, "rows": rows}


_SHARED_META = (
    "models",
    "form",
    "reasoning_regime",
    "prompt_template_sha256",
    "equations_sha256",
    "label_prefix",
    "seed",
    "temperature",
)


def check_consistency(passes: List[dict]) -> None:
    """Passes must sample the same prompts under the same conditions."""
    reference = passes[0]
    ref_ids = [(s["pair_id"], s["prompt_hash"]) for s in reference["samples"]]
    ref_meta = {key: reference["meta"].get(key) for key in _SHARED_META}
    for other in passes[1:]:
        ids = [(s["pair_id"], s["prompt_hash"]) for s in other["samples"]]
        if ids != ref_ids:
            raise SystemExit(
                f"{other['dir']}: sampled pairs/prompts differ from {reference['dir']}"
            )
        meta = {key: other["meta"].get(key) for key in _SHARED_META}
        if meta != ref_meta:
            diff = [key for key in _SHARED_META if meta[key] != ref_meta[key]]
            raise SystemExit(
                f"{other['dir']}: run_meta differs from {reference['dir']} "
                f"on {', '.join(diff)}"
            )


def ballot_of(row: dict) -> Optional[Tuple[str, str]]:
    """The row's grading-equivalence class, or None for an abstention."""
    if row["bucket"] in ("unparseable", "api-error"):
        return None
    verdict = row["verdict"]
    return answer_class_key(
        verdict["canonical_answer_e"], verdict["canonical_answer_f"]
    )


def vote_one(entries: List[Tuple[int, dict]]) -> dict:
    """Majority-vote one (pair, model) over its per-pass rows.

    entries: (1-based pass index, row), in pass order. Returns the
    representative row (a copy) with the ``vote`` field attached.
    """
    ballots = [(i, row, ballot_of(row)) for i, row in entries]
    valid = [(i, row, key) for i, row, key in ballots if key is not None]
    if not valid:
        # Every pass abstained: prefer a graded (unparseable) row over an
        # api-error row so aggregate()'s graded/api-error split is kept.
        source_i, source_row = next(
            ((i, row) for i, row in entries if row["bucket"] != "api-error"),
            entries[0],
        )
        return dict(
            source_row,
            vote={
                "k": len(entries),
                "valid_ballots": 0,
                "abstentions": len(entries),
                "votes_for_winner": 0,
                "distinct_classes": 0,
                "tie": False,
                "source_pass": source_i,
                "winner_class": None,
            },
        )

    tally = Counter(key for _, _, key in valid)
    first_seen: Dict[Tuple[str, str], int] = {}
    for i, _, key in valid:
        first_seen.setdefault(key, i)
    top = max(tally.values())
    tied = [key for key, count in tally.items() if count == top]
    winner = min(tied, key=lambda key: first_seen[key])
    source_i, source_row, _ = next(
        (i, row, key) for i, row, key in valid if key == winner
    )
    return dict(
        source_row,
        vote={
            "k": len(entries),
            "valid_ballots": len(valid),
            "abstentions": len(entries) - len(valid),
            "votes_for_winner": top,
            "distinct_classes": len(tally),
            "tie": len(tied) > 1,
            "source_pass": source_i,
            "winner_class": list(winner),
        },
    )


def voted_rows(passes: List[dict], k: int) -> List[dict]:
    """One voted row per (pair, model), in sample x model order."""
    prefix = passes[:k]
    meta = passes[0]["meta"]
    rows = []
    for sample in passes[0]["samples"]:
        for model in meta["models"]:
            key = (sample["pair_id"], model)
            entries = [
                (i, one["rows"][key])
                for i, one in enumerate(prefix, start=1)
                if key in one["rows"]
            ]
            if not entries:
                raise SystemExit(f"no pass has a row for pair={key[0]} model={key[1]}")
            if len(entries) < k:
                print(f"warning: only {len(entries)}/{k} passes cover {key}")
            rows.append(vote_one(entries))
    return rows


def write_run_dir(
    out_dir: Path,
    meta: dict,
    samples_src: Path,
    rows: List[dict],
    summary_title: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy(samples_src, out_dir / "samples.jsonl")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate(rows, meta["models"])
    table = summary_table(summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(
        f"# {summary_title}\n\n{table}\n", encoding="utf-8"
    )


def write_vote_dir(passes: List[dict], k: int, out_dir: Path) -> None:
    rows = voted_rows(passes, k)
    meta = copy.deepcopy(passes[0]["meta"])
    meta["condition_label"] = f"vote@{k}"
    meta["vote"] = {
        "k": k,
        "pass_dirs": [str(one["dir"]) for one in passes[:k]],
        "temperature": passes[0]["meta"].get("temperature"),
        "ballots_exclude": ["unparseable", "api-error"],
        "tie_break": "most votes, then earliest pass",
        "generated_by": "voteform.py",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_run_dir(
        out_dir,
        meta,
        passes[0]["dir"] / "samples.jsonl",
        rows,
        f"Majority vote: k={k}, "
        f"temp={passes[0]['meta'].get('temperature')}, "
        f"form={meta.get('form')}",
    )
    print(f"vote@{k}: {len(rows)} rows -> {out_dir}")


def write_baseline_subset(
    baseline_dir: Path, passes: List[dict], out_dir: Path, label: str
) -> None:
    baseline = load_pass(baseline_dir)
    pass_models = passes[0]["meta"]["models"]
    missing = [m for m in pass_models if m not in baseline["meta"]["models"]]
    if missing:
        raise SystemExit(f"{baseline_dir} lacks models: {', '.join(missing)}")
    ref_ids = [(s["pair_id"], s["prompt_hash"]) for s in passes[0]["samples"]]
    base_ids = [(s["pair_id"], s["prompt_hash"]) for s in baseline["samples"]]
    if base_ids != ref_ids:
        raise SystemExit(f"{baseline_dir}: sampled pairs/prompts differ from the passes")
    meta = copy.deepcopy(baseline["meta"])
    meta["models"] = list(pass_models)
    meta["condition_label"] = label
    meta["baseline_of"] = str(baseline_dir)
    rows = [
        baseline["rows"][(sample["pair_id"], model)]
        for sample in baseline["samples"]
        for model in pass_models
        if (sample["pair_id"], model) in baseline["rows"]
    ]
    write_run_dir(
        out_dir,
        meta,
        baseline_dir / "samples.jsonl",
        rows,
        f"Baseline subset of {baseline_dir.name}: {label}",
    )
    print(f"baseline: {len(rows)} rows -> {out_dir}")


def default_stem(pass_dir: Path) -> str:
    return re.sub(r"-pass\d+$", "", pass_dir.name)


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(
        description="Majority-vote repeated benchmark passes into run directories."
    )
    cli.add_argument("pass_dirs", nargs="+", type=Path, metavar="PASS_DIR",
                     help="pass run directories, in pass order (pass 1 first)")
    cli.add_argument("--ks", default="1,3,5,7",
                     help="comma-separated vote sizes; vote@k uses passes 1..k")
    cli.add_argument("--out-root", type=Path, default=Path("results/vote"))
    cli.add_argument("--name", default=None, metavar="STEM",
                     help="output stem (default: pass 1's name minus -passN)")
    cli.add_argument("--baseline", type=Path, default=None, metavar="RUN_DIR",
                     help="reference run over the same pairs; copied restricted "
                     "to the passes' models as <stem>-t0-baseline")
    cli.add_argument("--baseline-label", default="temp0 single-pass")
    args = cli.parse_args(argv)

    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    for k in ks:
        if not 1 <= k <= len(args.pass_dirs):
            raise SystemExit(f"k={k} needs {k} pass dirs, got {len(args.pass_dirs)}")
        if k % 2 == 0:
            print(f"warning: k={k} is even; ties fall back to the earliest pass")

    passes = [load_pass(run_dir) for run_dir in args.pass_dirs]
    check_consistency(passes)

    stem = args.name or default_stem(args.pass_dirs[0])
    for k in ks:
        write_vote_dir(passes, k, args.out_root / f"{stem}-vote{k}")
    if args.baseline:
        write_baseline_subset(
            args.baseline, passes, args.out_root / f"{stem}-t0-baseline",
            args.baseline_label,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
