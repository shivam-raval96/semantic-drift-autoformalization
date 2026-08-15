#!/usr/bin/env python3
"""Turn a model response into a graded record.

Grading itself is the vendored checkform module: it pulls the last `ASSUME:`
and `ASK:` lines out of the response, parses both as equations, and accepts an
answer if it matches the target up to a fixed set of eight transforms (renaming
variables, swapping the two sides, and dualizing the operator).

That "last line wins" rule is the reason the text handed in here matters. Give
it a response that still contains the model's reasoning and a row whose final
answer is unparseable can be rescued by an `ASSUME:` line the model wrote while
thinking, which inflates accuracy and deflates the unparseable rate. So grade
the *answer only*, which is what generation.generate_budgeted returns.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .stats import Rate, rate
from .vendor import ensure_on_path

ensure_on_path()

from checkform import grade as _grade  # noqa: E402

# Every outcome a graded row can have. The three correct buckets are split by
# which transform matched, because "right up to swapping the two sides" is a
# different kind of success from an exact match and the split has been
# informative before.
BUCKETS = (
    "exact",
    "correct-swapped",
    "correct-dualized",
    "wrong",
    "unparseable",
)

CORRECT_BUCKETS = ("exact", "correct-swapped", "correct-dualized")


def grade_record(
    answer: str,
    sample: dict,
    extra: Optional[dict] = None,
) -> dict:
    """Grade one answer against the pair it came from.

    `answer` must be the model's answer text with any reasoning removed (see
    the module docstring). `sample` is a row from the dataset module. Anything
    in `extra` — a condition name, a budget, a depth — is merged into the
    record, so a run's records carry the settings that produced them.
    """
    verdict = _grade(answer, sample["metadata"])
    record = {
        "pair_id": sample["pair_id"],
        "status": verdict["status"],
        "bucket": bucket_of(verdict),
        "transform": verdict["transform"],
        "ops_total": sample.get("ops_total"),
        "depth": sample.get("depth"),
        "shape": sample.get("shape"),
        "answer": answer,
    }
    if extra:
        record.update(extra)
    return record


def bucket_of(verdict: dict) -> str:
    """Which of BUCKETS a verdict falls into."""
    if verdict["status"] != "correct":
        return verdict["status"]  # "wrong" or "unparseable"
    transform = verdict["transform"] or {}
    if transform.get("dual"):
        return "correct-dualized"
    if transform.get("swap_e") or transform.get("swap_f"):
        return "correct-swapped"
    return "exact"


def correct_rate(records: Sequence[dict]) -> Rate:
    """Share of records graded correct, with denominator and interval."""
    return rate(sum(1 for r in records if r["status"] == "correct"), len(records))


def unparseable_rate(records: Sequence[dict]) -> Rate:
    """Share of records whose answer had no readable ASSUME/ASK pair.

    Worth reporting alongside accuracy: an intervention that mostly destroys
    the output format looks like an accuracy drop, and only this number tells
    the two apart.
    """
    return rate(sum(1 for r in records if r["status"] == "unparseable"), len(records))


def bucket_counts(records: Sequence[dict]) -> Dict[str, int]:
    """How many records landed in each bucket, including empty ones."""
    counts = {name: 0 for name in BUCKETS}
    for record in records:
        counts[record.get("bucket", record["status"])] += 1
    return counts


def group_by(records: Iterable[dict], field: str) -> Dict[object, List[dict]]:
    """Records grouped by one field, e.g. "depth" or "budget"."""
    groups: Dict[object, List[dict]] = {}
    for record in records:
        value = record.get(field)
        if isinstance(value, list):
            value = tuple(value)
        groups.setdefault(value, []).append(record)
    return groups
