#!/usr/bin/env python3
"""Paired analysis of Story -> RG and Literal NL -> RG benchmark runs.

The default invocation compares Experiment 02 (story) with Experiment 04
(structured literal) on their shared pair/model observations and writes a
small, self-contained report:

    python3 paired_analysis.py

No model calls are made.  This script only reads committed run artifacts.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from checkform import AnswerParseError, dual, extract_answer, parse_prefix_equation
from storyform import canonical


ROOT = Path(__file__).resolve().parent
DEFAULT_STORY_EXPERIMENT = ROOT / "experiments/02-reasoning-and-complexity"
DEFAULT_LITERAL_EXPERIMENT = ROOT / "experiments/04-structured-literal"
DEFAULT_OUT_DIR = ROOT / "experiments/05-story-literal-correlation/report"
CORRECT_BUCKETS = {"exact", "correct-swapped", "correct-dualized"}
GROUP_ORDER = {
    ("uniform", "off"): 0,
    ("uniform", "on"): 1,
    ("stratified", "off"): 2,
    ("stratified", "on"): 3,
}
ERROR_TAXONOMY_ORDER = {
    "assume_wrong_only": 0,
    "ask_wrong_only": 1,
    "both_wrong": 2,
    "inconsistent_operation_direction": 3,
    "unparseable": 4,
    "grader_mismatch": 5,
}
ERROR_TAXONOMY_LABELS = {
    "assume_wrong_only": "ASSUME wrong only",
    "ask_wrong_only": "ASK wrong only",
    "both_wrong": "Both laws wrong",
    "inconsistent_operation_direction": "Inconsistent operation direction",
    "unparseable": "Unparseable",
    "grader_mismatch": "Grader mismatch",
}


def load_result_rows(path: Path) -> Dict[Tuple[str, str], dict]:
    """Load the final usable row for each (pair, model).

    Resumed benchmark runs append retries.  A later graded row replaces an
    earlier row; an API error never replaces a response that was already
    graded.  This is the same policy used by charts.py.
    """
    rows: Dict[Tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["pair_id"], row["model"])
        if row["bucket"] != "api-error" or key not in rows:
            rows[key] = row
    return rows


def load_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sampling = "stratified" if meta.get("stratify_ops") else "uniform"
    regime = meta.get("reasoning_regime")
    if regime not in {"off", "on"}:
        raise ValueError(f"{run_dir}: expected reasoning regime on/off, got {regime!r}")
    return {
        "dir": run_dir,
        "meta": meta,
        "samples": {sample["pair_id"]: sample for sample in samples},
        "rows": load_result_rows(run_dir / "results.jsonl"),
        "key": (sampling, regime),
    }


def discover_runs(experiment_dir: Path) -> Dict[Tuple[str, str], dict]:
    runs = {}
    for run_dir in sorted((experiment_dir / "runs").iterdir()):
        if not (run_dir / "run_meta.json").exists():
            continue
        run = load_run(run_dir)
        if run["key"] in runs:
            raise ValueError(f"{experiment_dir}: duplicate run group {run['key']}")
        runs[run["key"]] = run
    return runs


def is_correct(row: dict) -> bool:
    return row["bucket"] in CORRECT_BUCKETS


def _matching_dual_options(equation: tuple, truth: str) -> set:
    lhs, rhs = equation
    matches = set()
    for dualize in (False, True):
        left, right = (dual(lhs), dual(rhs)) if dualize else (lhs, rhs)
        if canonical(left, right) == truth or canonical(right, left) == truth:
            matches.add(dualize)
    return matches


def classify_story_error(story_row: dict, sample: dict) -> str:
    """Classify a failed Story result by which law preserves its meaning."""
    if story_row["bucket"] == "unparseable":
        return "unparseable"
    try:
        assume_text, ask_text = extract_answer(story_row["response"])
        assume = parse_prefix_equation(assume_text)
        ask = parse_prefix_equation(ask_text)
    except AnswerParseError:
        return "unparseable"

    metadata = sample["metadata"]
    assume_options = _matching_dual_options(assume, metadata["canonical_e"])
    ask_options = _matching_dual_options(ask, metadata["canonical_f"])
    if assume_options and ask_options:
        if assume_options & ask_options:
            return "grader_mismatch"
        return "inconsistent_operation_direction"
    if assume_options:
        return "ask_wrong_only"
    if ask_options:
        return "assume_wrong_only"
    return "both_wrong"


def build_observations(
    story_dir: Path, literal_dir: Path, direct_runs: bool = False
) -> Tuple[List[dict], List[str]]:
    if direct_runs:
        story = load_run(story_dir)
        literal = load_run(literal_dir)
        story_runs = {story["key"]: story}
        literal_runs = {literal["key"]: literal}
    else:
        story_runs = discover_runs(story_dir)
        literal_runs = discover_runs(literal_dir)
    if set(story_runs) != set(literal_runs):
        raise ValueError(
            "story and literal experiments do not contain the same run groups: "
            f"story={sorted(story_runs)}, literal={sorted(literal_runs)}"
        )

    observations: List[dict] = []
    warnings: List[str] = []
    for group in sorted(story_runs, key=lambda key: GROUP_ORDER.get(key, 99)):
        story = story_runs[group]
        literal = literal_runs[group]
        story_pairs = set(story["samples"])
        literal_pairs = set(literal["samples"])
        if story_pairs != literal_pairs:
            raise ValueError(
                f"{group}: pair sets differ; story-only={sorted(story_pairs - literal_pairs)}, "
                f"literal-only={sorted(literal_pairs - story_pairs)}"
            )

        expected = {
            (pair_id, model)
            for pair_id in story_pairs
            for model in story["meta"]["models"]
        }
        if set(story["meta"]["models"]) != set(literal["meta"]["models"]):
            raise ValueError(f"{group}: model sets differ")

        story_graded = {
            key: row for key, row in story["rows"].items() if row["bucket"] != "api-error"
        }
        literal_graded = {
            key: row for key, row in literal["rows"].items() if row["bucket"] != "api-error"
        }
        common = expected & set(story_graded) & set(literal_graded)
        missing_story = expected - set(story_graded)
        missing_literal = expected - set(literal_graded)
        sampling, regime = group
        if missing_story:
            warnings.append(
                f"{sampling}/{regime}: {len(missing_story)} ungraded story evaluations excluded"
            )
        if missing_literal:
            warnings.append(
                f"{sampling}/{regime}: {len(missing_literal)} ungraded literal evaluations excluded"
            )

        for pair_id, model in sorted(common):
            sample = story["samples"][pair_id]
            story_row = story_graded[(pair_id, model)]
            literal_row = literal_graded[(pair_id, model)]
            story_correct = is_correct(story_row)
            literal_correct = is_correct(literal_row)
            observations.append(
                {
                    "sampling": sampling,
                    "regime": regime,
                    "pair_id": pair_id,
                    "model": model,
                    "ops_total": sample.get("ops_total"),
                    "theme": story_row.get("theme"),
                    "story_correct": story_correct,
                    "literal_correct": literal_correct,
                    "story_bucket": story_row["bucket"],
                    "literal_bucket": literal_row["bucket"],
                    "story_error_type": (
                        classify_story_error(story_row, sample)
                        if literal_correct and not story_correct
                        else None
                    ),
                }
            )
    return observations, warnings


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    both = sum(row["story_correct"] and row["literal_correct"] for row in rows)
    literal_only = sum(
        (not row["story_correct"]) and row["literal_correct"] for row in rows
    )
    story_only = sum(
        row["story_correct"] and (not row["literal_correct"]) for row in rows
    )
    neither = sum(
        (not row["story_correct"]) and (not row["literal_correct"]) for row in rows
    )
    n = len(rows)
    literal_correct_n = both + literal_only
    literal_wrong_n = story_only + neither
    story_correct_n = both + story_only

    denominator = math.sqrt(
        (both + literal_only)
        * (story_only + neither)
        * (both + story_only)
        * (literal_only + neither)
    )
    phi = (both * neither - literal_only * story_only) / denominator if denominator else None
    p_story_given_literal_correct = _ratio(both, literal_correct_n)
    p_story_given_literal_wrong = _ratio(story_only, literal_wrong_n)
    risk_difference = (
        p_story_given_literal_correct - p_story_given_literal_wrong
        if p_story_given_literal_correct is not None
        and p_story_given_literal_wrong is not None
        else None
    )
    return {
        "n": n,
        "both_correct": both,
        "literal_only": literal_only,
        "story_only": story_only,
        "neither": neither,
        "story_accuracy": _ratio(story_correct_n, n),
        "literal_accuracy": _ratio(literal_correct_n, n),
        "literal_correct_n": literal_correct_n,
        "literal_wrong_n": literal_wrong_n,
        "p_story_given_literal_correct": p_story_given_literal_correct,
        "p_story_given_literal_wrong": p_story_given_literal_wrong,
        "risk_difference": risk_difference,
        "phi": phi,
    }


def grouped_summaries(observations: List[dict], fields: Tuple[str, ...]) -> List[dict]:
    grouped: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)
    for row in observations:
        grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, rows in grouped.items():
        record = dict(zip(fields, key))
        record.update(summarize(rows))
        output.append(record)
    return output


def error_taxonomy(observations: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    pair_counts: Dict[Tuple[str, str], set] = defaultdict(set)
    for row in observations:
        group = (row["sampling"], row["regime"])
        pair_counts[group].add(row["pair_id"])
        if row.get("story_error_type"):
            grouped[group].append(row["story_error_type"])

    output = []
    for (sampling, regime), errors in grouped.items():
        total = len(errors)
        for error_type in ERROR_TAXONOMY_ORDER:
            count = errors.count(error_type)
            if count:
                output.append(
                    {
                        "sampling": sampling,
                        "regime": regime,
                        "pair_n": len(pair_counts[(sampling, regime)]),
                        "error_type": error_type,
                        "count": count,
                        "share": count / total,
                    }
                )
    return sorted(
        output,
        key=lambda row: (
            GROUP_ORDER.get((row["sampling"], row["regime"]), 99),
            ERROR_TAXONOMY_ORDER[row["error_type"]],
        ),
    )


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if not denominator:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def average_ranks(values: List[float]) -> List[float]:
    """Return one-based ranks, averaging ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + end + 2) / 2
        for offset in range(start, end + 1):
            ranks[order[offset]] = rank
        start = end + 1
    return ranks


def add_model_correlations(summary_rows: List[dict], model_rows: List[dict]) -> None:
    by_group: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in model_rows:
        by_group[(row["sampling"], row["regime"])].append(row)
    for summary in summary_rows:
        rows = by_group[(summary["sampling"], summary["regime"])]
        literal = [row["literal_accuracy"] for row in rows]
        story = [row["story_accuracy"] for row in rows]
        summary["model_n"] = len(rows)
        summary["model_pearson"] = pearson(literal, story)
        summary["model_spearman"] = pearson(average_ranks(literal), average_ranks(story))


def _sort_key(row: dict) -> tuple:
    group = (row.get("sampling"), row.get("regime"))
    return (
        GROUP_ORDER.get(group, 99),
        str(row.get("model", "")),
        row.get("ops_total") if row.get("ops_total") is not None else -1,
    )


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{100 * value:.{digits}f}%"


def number(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def sample_name(row: dict) -> str:
    sample = "Complex" if row["sampling"] == "uniform" else "Stratified"
    pair_n = row.get("pair_n")
    if pair_n is None:
        model_n = row.get("model_n")
        pair_n = row["n"] // model_n if model_n else row["n"]
    return f"{sample} {pair_n}"


def group_name(row: dict) -> str:
    reasoning = "Reasoning off" if row["regime"] == "off" else "Reasoning on"
    return f"{sample_name(row)} / {reasoning}"


def markdown_report(
    summary_rows: List[dict], warnings: List[str], taxonomy_rows: Optional[List[dict]] = None
) -> str:
    lines = [
        "# Story–Literal Paired Correlation",
        "",
        "This analysis pairs Story → RG with Structured Literal NL → RG by "
        "`pair_id × model`. No new model calls were made.",
        "",
        "## Results",
        "",
        "| Sample | Reasoning | N | Story → RG | Structured Literal NL → RG | "
        "P(Story → RG ✓ given Literal → RG ✓) | "
        "P(Story → RG ✓ given Literal → RG ✗) | Difference | Pair-model phi | "
        "Model Pearson r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {sampling} | {regime} | {n} | {story} | {literal} | {p_yes} | "
            "{p_no} | {diff} | {phi} | {model_r} |".format(
                sampling=sample_name(row),
                regime=row["regime"],
                n=row["n"],
                story=pct(row["story_accuracy"]),
                literal=pct(row["literal_accuracy"]),
                p_yes=pct(row["p_story_given_literal_correct"]),
                p_no=pct(row["p_story_given_literal_wrong"]),
                diff=pct(row["risk_difference"]),
                phi=number(row["phi"]),
                model_r=number(row["model_pearson"]),
            )
        )
    if taxonomy_rows:
        lines.extend(
            [
                "",
                "## Literal-correct / Story-wrong error taxonomy",
                "",
                "| Sample | Reasoning | Error type | N | Share |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for row in taxonomy_rows:
            lines.append(
                f"| {sample_name(row)} | {row['regime']} | "
                f"{ERROR_TAXONOMY_LABELS[row['error_type']]} | "
                f"{row['count']} | {pct(row['share'])} |"
            )
    lines.extend(["", "## Interpretation", ""])
    off_rows = [row for row in summary_rows if row["regime"] == "off"]
    for row in off_rows:
        lines.append(
            f"- {sample_name(row)}: Story accuracy rises from "
            f"{pct(row['p_story_given_literal_wrong'])} when Literal fails to "
            f"{pct(row['p_story_given_literal_correct'])} when Literal succeeds "
            f"(phi {number(row['phi'])}; model Pearson r "
            f"{number(row['model_pearson'])})."
        )
    if any(row["regime"] == "on" for row in summary_rows):
        lines.append(
            "- With reasoning on, near-ceiling results can make conditional rates "
            "and correlations unstable."
        )
    lines.append(
        "- Literal-correct/Story-wrong cases isolate a likely narrative-abstraction "
        "bottleneck; failures on both forms indicate structural-encoding difficulty."
    )
    model_counts = [row.get("model_n") for row in summary_rows if row.get("model_n")]
    model_count = max(model_counts) if model_counts else 0
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            f"The model-level correlations use {model_count} models. The pair-model "
            "associations are descriptive: rows from the same model and equation are "
            "not statistically independent; a confirmatory analysis should use a "
            "mixed-effects model with model and equation effects.",
        ]
    )
    if warnings:
        lines.extend(["", "## Data notes", ""] + [f"- {warning}" for warning in warnings])
    return "\n".join(lines) + "\n"


def _bar(label: str, value: Optional[float], color: str, denominator: int) -> str:
    width = 0 if value is None else 100 * value
    value_text = pct(value)
    return (
        '<div class="bar-row">'
        f'<span class="bar-label">{html.escape(label)}</span>'
        '<span class="track">'
        f'<span class="fill" style="width:{width:.1f}%;background:{color}"></span>'
        "</span>"
        f'<strong>{value_text}</strong><small>n={denominator}</small>'
        "</div>"
    )


def html_report(
    summary_rows: List[dict],
    model_rows: List[dict],
    warnings: List[str],
    taxonomy_rows: Optional[List[dict]] = None,
) -> str:
    task_panels = []
    conditional_panels = []
    matrices_off = []
    matrices_on = []
    for row in summary_rows:
        sample = sample_name(row)
        if row["regime"] == "off":
            accuracy_gap = row["literal_accuracy"] - row["story_accuracy"]
            task_panels.append(
                '<section class="panel"><h3>'
                + html.escape(sample)
                + "</h3>"
                + _bar("Direct Story → RG", row["story_accuracy"], "#1d63ad", row["n"])
                + _bar(
                    "Oracle Literal NL → RG",
                    row["literal_accuracy"],
                    "#8fb5d9",
                    row["n"],
                )
                + f'<p class="metric">Literal advantage {pct(accuracy_gap)} · '
                + f'model-level Pearson r {number(row["model_pearson"])}</p>'
                + "</section>"
            )
            conditional_panels.append(
                '<section class="panel"><h3>'
                + html.escape(sample)
                + "</h3>"
                + _bar(
                    "Literal → RG succeeded",
                    row["p_story_given_literal_correct"],
                    "#1d63ad",
                    row["literal_correct_n"],
                )
                + _bar(
                    "Literal → RG failed",
                    row["p_story_given_literal_wrong"],
                    "#8fb5d9",
                    row["literal_wrong_n"],
                )
                + f'<p class="metric">Conditional difference {pct(row["risk_difference"])} · '
                + f'pair-model phi {number(row["phi"])}</p>'
                + "</section>"
            )
        matrix = (
            "<tr>"
            f"<th>{html.escape(group_name(row))}</th>"
            f"<td>{row['both_correct']}</td><td>{row['literal_only']}</td>"
            f"<td>{row['story_only']}</td><td>{row['neither']}</td>"
            "</tr>"
        )
        (matrices_off if row["regime"] == "off" else matrices_on).append(matrix)

    model_table = []
    for row in model_rows:
        model_table.append(
            "<tr>"
            f"<td>{html.escape(group_name(row))}</td>"
            f"<td>{html.escape(row['model'].split('/', 1)[-1])}</td>"
            f"<td>{row['n']}</td><td>{pct(row['story_accuracy'])}</td>"
            f"<td>{pct(row['literal_accuracy'])}</td>"
            f"<td>{pct(row['p_story_given_literal_correct'])}</td>"
            f"<td>{pct(row['p_story_given_literal_wrong'])}</td>"
            f"<td>{number(row['phi'])}</td>"
            "</tr>"
        )
    notes = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
    notes_block = f"<h2>Data notes</h2><ul>{notes}</ul>" if notes else ""
    taxonomy_table = "".join(
        "<tr>"
        f"<td>{html.escape(sample_name(row))}</td>"
        f"<td>{html.escape(row['regime'])}</td>"
        f"<td>{html.escape(ERROR_TAXONOMY_LABELS[row['error_type']])}</td>"
        f"<td>{row['count']}</td><td>{pct(row['share'])}</td>"
        "</tr>"
        for row in (taxonomy_rows or [])
    )
    taxonomy_block = (
        '<section class="section"><h2>Literal-correct / Story-wrong error taxonomy</h2>'
        '<div class="table-wrap"><table><thead><tr><th>Condition</th>'
        '<th>Reasoning</th><th>Error type</th><th>N</th><th>Share</th></tr></thead>'
        f"<tbody>{taxonomy_table}</tbody></table></div></section>"
        if taxonomy_table
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Story–Literal Paired Correlation</title>
<style>
:root {{ --ink:#17212b; --muted:#607080; --line:#d6dee6; --bg:#f5f8fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--bg); font:15px/1.45 Arial,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 24px 56px; }}
h1 {{ margin:0 0 6px; font-size:28px; letter-spacing:0; }}
h2 {{ margin:0 0 16px; font-size:17px; letter-spacing:0; }}
h3 {{ margin:0 0 16px; font-size:16px; letter-spacing:0; }}
p {{ max-width:850px; }}
.lead {{ color:var(--muted); margin:0 0 28px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
.panel {{ background:white; border:1px solid var(--line); border-radius:6px; padding:18px; }}
.bar-row {{ display:grid; grid-template-columns:190px minmax(120px,1fr) 58px 44px; gap:10px; align-items:center; margin:12px 0; }}
.bar-label {{ font-weight:600; }}
.track {{ height:20px; background:#e8edf2; display:block; overflow:hidden; }}
.fill {{ height:100%; display:block; }}
.bar-row strong {{ text-align:right; }}
.bar-row small {{ color:var(--muted); }}
.metric {{ color:var(--muted); margin:14px 0 0; }}
.section {{ margin-top:30px; }}
.table-wrap {{ overflow-x:auto; background:white; border:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
thead th {{ background:#edf3f8; font-size:13px; }}
tbody tr:last-child td, tbody tr:last-child th {{ border-bottom:0; }}
.note {{ border-left:4px solid #1d63ad; padding:2px 0 2px 14px; max-width:900px; }}
details {{ margin-top:20px; }}
summary {{ cursor:pointer; font-weight:700; }}
@media (max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} .bar-row {{ grid-template-columns:150px 1fr 52px; }} .bar-row small {{ display:none; }} }}
</style>
</head>
<body><main>
<h1>Direct Story vs Oracle Literal</h1>
<p class="lead">Same ETP pairs and models. Story → RG uses the generated story; Oracle Literal NL → RG uses the program-generated correct Structured Literal. This is not a generated two-hop experiment.</p>
<section>
<h2>Task accuracy · reasoning off</h2>
<div class="grid">{''.join(task_panels)}</div>
</section>
<section class="section">
<h2>Does Literal success predict Story success? · reasoning off</h2>
<p>The bars below are Story → RG success rates, conditioned on whether the matching Oracle Literal NL → RG attempt succeeded or failed.</p>
<div class="grid">{''.join(conditional_panels)}</div>
</section>
<section class="section">
<h2>Paired outcomes · reasoning off</h2>
<div class="table-wrap"><table>
<thead><tr><th>Condition</th><th>Both correct</th><th>Literal only</th><th>Story only</th><th>Neither</th></tr></thead>
<tbody>{''.join(matrices_off)}</tbody></table></div>
</section>
{taxonomy_block}
<section class="section note">
<h2>Reading the result</h2>
<p>With reasoning off, Story success is substantially more likely when the matching Literal item succeeds. Near-ceiling conditions can make both forms of correlation unstable.</p>
</section>
<details><summary>Reasoning-on ceiling results</summary>
<div class="table-wrap"><table>
<thead><tr><th>Condition</th><th>Both correct</th><th>Literal only</th><th>Story only</th><th>Neither</th></tr></thead>
<tbody>{''.join(matrices_on)}</tbody></table></div>
</details>
<section class="section">
<h2>By model</h2>
<div class="table-wrap"><table>
<thead><tr><th>Condition</th><th>Model</th><th>N</th><th>Story</th><th>Literal</th><th>P(S ✓ | L ✓)</th><th>P(S ✓ | L ✗)</th><th>Phi</th></tr></thead>
<tbody>{''.join(model_table)}</tbody></table></div>
</section>
{notes_block}
</main></body></html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair Story and Literal benchmark outcomes; no API calls are made."
    )
    parser.add_argument("--story", type=Path, default=DEFAULT_STORY_EXPERIMENT)
    parser.add_argument("--literal", type=Path, default=DEFAULT_LITERAL_EXPERIMENT)
    parser.add_argument("--story-run", type=Path)
    parser.add_argument("--literal-run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if (args.story_run is None) != (args.literal_run is None):
        parser.error("--story-run and --literal-run must be provided together")
    if args.story_run is not None and args.out is None:
        parser.error("--out is required with --story-run/--literal-run")
    if args.out is None:
        args.out = DEFAULT_OUT_DIR
    return args


def main() -> int:
    args = parse_args()
    if args.story_run is not None:
        observations, warnings = build_observations(
            args.story_run, args.literal_run, direct_runs=True
        )
    else:
        observations, warnings = build_observations(args.story, args.literal)
    summary_rows = sorted(
        grouped_summaries(observations, ("sampling", "regime")), key=_sort_key
    )
    model_rows = sorted(
        grouped_summaries(observations, ("sampling", "regime", "model")), key=_sort_key
    )
    add_model_correlations(summary_rows, model_rows)
    taxonomy_rows = error_taxonomy(observations)
    complexity_rows = sorted(
        grouped_summaries(observations, ("sampling", "regime", "ops_total")), key=_sort_key
    )

    args.out.mkdir(parents=True, exist_ok=True)
    stat_fields = [
        "n", "both_correct", "literal_only", "story_only", "neither",
        "story_accuracy", "literal_accuracy", "literal_correct_n", "literal_wrong_n",
        "p_story_given_literal_correct", "p_story_given_literal_wrong",
        "risk_difference", "phi",
    ]
    write_csv(
        args.out / "summary.csv",
        summary_rows,
        [
            "sampling", "regime", *stat_fields,
            "model_n", "model_pearson", "model_spearman",
        ],
    )
    write_csv(
        args.out / "by_model.csv",
        model_rows,
        ["sampling", "regime", "model", *stat_fields],
    )
    write_csv(
        args.out / "by_complexity.csv",
        complexity_rows,
        ["sampling", "regime", "ops_total", *stat_fields],
    )
    write_csv(
        args.out / "paired_observations.csv",
        observations,
        [
            "sampling", "regime", "pair_id", "model", "ops_total", "theme",
            "story_correct", "literal_correct", "story_bucket", "literal_bucket",
            "story_error_type",
        ],
    )
    write_csv(
        args.out / "error_taxonomy.csv",
        taxonomy_rows,
        ["sampling", "regime", "pair_n", "error_type", "count", "share"],
    )
    (args.out / "summary.md").write_text(
        markdown_report(summary_rows, warnings, taxonomy_rows), encoding="utf-8"
    )
    (args.out / "paired-analysis.html").write_text(
        html_report(summary_rows, model_rows, warnings, taxonomy_rows), encoding="utf-8"
    )

    print(f"Paired observations: {len(observations)}")
    for row in summary_rows:
        print(
            f"{row['sampling']}/{row['regime']}: n={row['n']}, "
            f"P(S|L correct)={pct(row['p_story_given_literal_correct'])}, "
            f"P(S|L wrong)={pct(row['p_story_given_literal_wrong'])}, "
            f"phi={number(row['phi'])}, model-r={number(row['model_pearson'])}"
        )
    if taxonomy_rows:
        print("Literal-correct / Story-wrong error taxonomy:")
        for row in taxonomy_rows:
            print(
                f"  {sample_name(row)}/{row['regime']} - "
                f"{ERROR_TAXONOMY_LABELS[row['error_type']]}: "
                f"{row['count']} ({pct(row['share'])})"
            )
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Report: {args.out / 'paired-analysis.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
