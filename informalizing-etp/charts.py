#!/usr/bin/env python3
"""Charts: render benchmark runs into a self-contained HTML report.

Reads one or more run directories produced by benchmark.py and emits a
single HTML file with inline SVG charts — no dependencies, no network,
deterministic for a given set of inputs.

Runs are grouped into experiments by their sampled pair set; each
experiment section gets, as its data allows:

  * a regime-comparison dumbbell (one row per model) when the group has
    two or more runs (e.g. --reasoning off vs on over the same pairs);
  * an accuracy-vs-complexity line chart when the sample spans three or
    more total-operation bins (see benchmark.py --stratify-ops);
  * a verdict-composition stacked bar per run;
  * a collapsible data table under every figure.

Usage:
    python3 charts.py RUN_DIR [RUN_DIR ...] [--out report.html] [--title TITLE]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CORRECT_BUCKETS = ("exact", "correct-swapped", "correct-dualized")
COMPOSITION = (
    ("exact", "var(--c-exact)"),
    ("correct-swapped", "var(--c-swap)"),
    ("correct-dualized", "var(--c-dual)"),
    ("wrong", "var(--c-wrong)"),
    ("unparseable", "var(--c-unp)"),
)
SERIES_VARS = ("var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)")

VB_W = 860  # SVG viewBox width shared by all charts


# -------------------------------------------------------------- Data model


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    rows: Dict[Tuple[str, str], dict] = {}
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # retries append; the last graded row per (pair, model) wins
        if row["bucket"] != "api-error" or (row["pair_id"], row["model"]) not in rows:
            rows[(row["pair_id"], row["model"])] = row
    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    regime = meta.get("reasoning_regime")
    return {
        "dir": run_dir,
        "regime": regime,
        "label": {"off": "no-think", "on": "all-think", None: "legacy"}[regime],
        "meta": meta,
        "rows": list(rows.values()),
        "samples": samples,
        "models": meta["models"],
        "pair_ids": frozenset(s["pair_id"] for s in samples),
    }


def group_runs(runs: List[dict]) -> List[List[dict]]:
    """Group runs sampling the same pairs; groups ordered by first run given."""
    groups: Dict[frozenset, List[dict]] = {}
    for run in runs:
        groups.setdefault(run["pair_ids"], []).append(run)
    return list(groups.values())


def experiment_title(group: List[dict]) -> str:
    meta = group[0]["meta"]
    pairs = len(group[0]["samples"])
    if meta.get("stratify_ops"):
        how = f"{pairs} pairs, {meta['stratify_ops']} per operation-count bin 1–8"
    else:
        how = f"{pairs} uniformly sampled pairs"
    regimes = " vs ".join(run["label"] for run in group)
    return f"{how} · seed {meta['seed']} · {regimes}"


def short_model(model: str) -> str:
    return model.split("/", 1)[-1]


def correct_pct(rows: List[dict]) -> Optional[float]:
    graded = [r for r in rows if r["bucket"] != "api-error"]
    if not graded:
        return None
    return 100 * sum(r["bucket"] in CORRECT_BUCKETS for r in graded) / len(graded)


# ------------------------------------------------------------ SVG helpers


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _tip(text: str) -> str:
    return f'data-tip="{esc(text)}" tabindex="0"'


def _grid_and_yaxis(x0: int, x1: int, y_of, ticks: List[int], unit: str = "%") -> List[str]:
    parts = []
    for tick in ticks:
        y = y_of(tick)
        parts.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{tick}{unit}</text>'
        )
    return parts


def legend_chips(entries: List[Tuple[str, str]]) -> str:
    chips = "".join(
        f'<span class="chip"><span class="swatch" style="background:{color}"></span>{esc(name)}</span>'
        for name, color in entries
    )
    return f'<div class="legend">{chips}</div>'


def line_chart(
    series: List[dict], x_values: List[int], x_title: str, height: int = 320
) -> str:
    """series: [{name, color, points: {x: value}}] with values in 0..100."""
    left, right, top, bottom = 56, 120, 16, 44
    plot_w, plot_h = VB_W - left - right, height - top - bottom
    x_of = lambda x: left + plot_w * (x_values.index(x) / max(1, len(x_values) - 1))
    y_of = lambda v: top + plot_h * (1 - v / 100)

    parts = _grid_and_yaxis(left, left + plot_w, y_of, [0, 25, 50, 75, 100])
    for x in x_values:
        parts.append(
            f'<text x="{x_of(x):.1f}" y="{height - 22}" class="tick" text-anchor="middle">{x}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 4}" class="axis-title" '
        f'text-anchor="middle">{esc(x_title)}</text>'
    )
    for s in series:
        pts = [(x_of(x), y_of(s["points"][x])) for x in x_values if x in s["points"]]
        path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        parts.append(f'<path d="{path}" fill="none" stroke="{s["color"]}" '
                     'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for x in x_values:
            if x not in s["points"]:
                continue
            cx, cy, v = x_of(x), y_of(s["points"][x]), s["points"][x]
            tip = _tip(f'{s["name"]} · {x_title} {x} · {v:.0f}% correct')
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" class="ring"/>'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{s["color"]}" {tip}/>'
            )
        end_x, end_y = pts[-1]
        end_v = s["points"][x_values[-1]]
        parts.append(
            f'<text x="{end_x + 12:.1f}" y="{end_y + 4:.1f}" class="end-label">'
            f'{esc(s["name"])} {end_v:.0f}%</text>'
        )
    svg = "".join(parts)
    return (
        legend_chips([(s["name"], s["color"]) for s in series])
        + f'<svg viewBox="0 0 {VB_W} {height}" role="img">{svg}</svg>'
    )


def dumbbell_chart(rows: List[dict], a_name: str, b_name: str) -> str:
    """rows: [{label, a, b, tip_a, tip_b}], values 0..100; a/b may be None."""
    left, right, top, row_h = 250, 60, 12, 34
    plot_w = VB_W - left - right
    height = top + row_h * len(rows) + 30
    x_of = lambda v: left + plot_w * v / 100
    parts = []
    for tick in (0, 25, 50, 75, 100):
        x = x_of(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + row_h * len(rows)}" class="grid"/>'
            f'<text x="{x:.1f}" y="{height - 8}" class="tick" text-anchor="middle">{tick}%</text>'
        )
    for i, row in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        parts.append(
            f'<text x="{left - 14}" y="{cy + 4:.1f}" class="row-label" text-anchor="end">'
            f"{esc(row['label'])}</text>"
        )
        a, b = row["a"], row["b"]
        if a is not None and b is not None:
            parts.append(
                f'<line x1="{x_of(a):.1f}" y1="{cy:.1f}" x2="{x_of(b):.1f}" y2="{cy:.1f}" '
                'class="connector"/>'
            )
        for value, var, tip in ((a, "var(--s1)", row["tip_a"]), (b, "var(--s2)", row["tip_b"])):
            if value is None:
                continue
            parts.append(
                f'<circle cx="{x_of(value):.1f}" cy="{cy:.1f}" r="7" class="ring"/>'
                f'<circle cx="{x_of(value):.1f}" cy="{cy:.1f}" r="5" fill="{var}" {_tip(tip)}/>'
            )
        # label the pair's values outside the dumbbell, low side left, high side
        # right; a low value too close to the label gutter moves above its dot
        if a is not None and b is not None:
            lo, hi = sorted((a, b))
            if x_of(lo) - 12 > left + 8:
                lo_label = (
                    f'<text x="{x_of(lo) - 12:.1f}" y="{cy + 4:.1f}" class="value" '
                    f'text-anchor="end">{lo:.0f}</text>'
                )
            else:
                lo_label = (
                    f'<text x="{x_of(lo):.1f}" y="{cy - 12:.1f}" class="value" '
                    f'text-anchor="middle">{lo:.0f}</text>'
                )
            parts.append(
                lo_label
                + f'<text x="{x_of(hi) + 12:.1f}" y="{cy + 4:.1f}" class="value">{hi:.0f}</text>'
            )
    svg = "".join(parts)
    return (
        legend_chips([(a_name, "var(--s1)"), (b_name, "var(--s2)")])
        + f'<svg viewBox="0 0 {VB_W} {height}" role="img">{svg}</svg>'
    )


def stacked_bars(rows: List[dict]) -> str:
    """rows: [{label, counts: {bucket: n}}]; one horizontal bar per row."""
    left, right, top, row_h, bar_h = 250, 70, 8, 30, 18
    plot_w = VB_W - left - right
    height = top + row_h * len(rows) + 8
    parts = []
    for i, row in enumerate(rows):
        y = top + row_h * i + (row_h - bar_h) / 2
        total = sum(row["counts"].values()) or 1
        cy = y + bar_h / 2
        parts.append(
            f'<text x="{left - 14}" y="{cy + 4:.1f}" class="row-label" text-anchor="end">'
            f"{esc(row['label'])}</text>"
        )
        x = float(left)
        for bucket, color in COMPOSITION:
            n = row["counts"].get(bucket, 0)
            if not n:
                continue
            w = plot_w * n / total
            tip = _tip(f"{row['label']} · {bucket}: {n}/{total}")
            parts.append(
                f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{max(0.5, w - 2):.1f}" '
                f'height="{bar_h}" rx="2" fill="{color}" {tip}/>'
            )
            x += w
        pct = correct_pct_counts(row["counts"])
        parts.append(
            f'<text x="{left + plot_w + 12}" y="{cy + 4:.1f}" class="value">{pct:.0f}%</text>'
        )
    svg = "".join(parts)
    return (
        legend_chips([(name, color) for name, color in COMPOSITION])
        + f'<svg viewBox="0 0 {VB_W} {height}" role="img">{svg}</svg>'
    )


def correct_pct_counts(counts: Dict[str, int]) -> float:
    total = sum(counts.values()) or 1
    return 100 * sum(counts.get(b, 0) for b in CORRECT_BUCKETS) / total


# ------------------------------------------------------------ Aggregation


def by_model_counts(run: dict) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {m: {} for m in run["models"]}
    for row in run["rows"]:
        counts = out.setdefault(row["model"], {})
        counts[row["bucket"]] = counts.get(row["bucket"], 0) + 1
    return out


def ops_bins(run: dict) -> List[int]:
    return sorted({r.get("ops_total") for r in run["rows"] if r.get("ops_total")})


def acc_by_ops(run: dict) -> Dict[int, float]:
    out = {}
    for b in ops_bins(run):
        rows = [r for r in run["rows"] if r.get("ops_total") == b]
        pct = correct_pct(rows)
        if pct is not None:
            out[b] = pct
    return out


def data_table(headers: List[str], rows: List[List[object]], summary: str) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        f"<details><summary>{esc(summary)}</summary>"
        f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div></details>"
    )


# ---------------------------------------------------------------- Figures


def figure(number: int, title: str, caption: str, body: str) -> str:
    return (
        f'<figure><figcaption><span class="fig-no">Figure {number}</span> '
        f"<strong>{esc(title)}</strong><br><span class='cap'>{esc(caption)}</span>"
        f"</figcaption>{body}</figure>"
    )


def render_group(group: List[dict], fig_no: List[int]) -> str:
    sections = [f"<h2>{esc(experiment_title(group))}</h2>"]
    models = group[0]["models"]

    if len(group) >= 2:
        a, b = group[0], group[1]
        a_by, b_by = by_model_counts(a), by_model_counts(b)
        rows = []
        for m in models:
            av = correct_pct_counts(a_by[m])
            bv = correct_pct_counts(b_by[m])
            rows.append(
                {
                    "label": short_model(m),
                    "a": av,
                    "b": bv,
                    "tip_a": f"{m} · {a['label']}: {av:.1f}% correct",
                    "tip_b": f"{m} · {b['label']}: {bv:.1f}% correct",
                }
            )
        rows.sort(key=lambda r: (r["b"], r["a"]), reverse=True)
        fig_no[0] += 1
        table = data_table(
            ["model", f"{a['label']} %", f"{b['label']} %", "Δ"],
            [[r["label"], f"{r['a']:.1f}", f"{r['b']:.1f}", f"{r['b'] - r['a']:+.1f}"] for r in rows],
            "Data: correct rate per model and regime",
        )
        sections.append(
            figure(
                fig_no[0],
                f"Correct rate by model: {a['label']} vs {b['label']}",
                f"Same {len(a['samples'])} stories under both regimes; "
                "each row connects a model's cold-pass rate to its deliberate rate.",
                dumbbell_chart(rows, a["label"], b["label"]) + table,
            )
        )

    bins = ops_bins(group[0])
    if len(bins) >= 3:
        series = []
        for i, run in enumerate(group):
            series.append(
                {"name": run["label"], "color": SERIES_VARS[i % len(SERIES_VARS)],
                 "points": acc_by_ops(run)}
            )
        fig_no[0] += 1
        table = data_table(
            ["total operations", *[s["name"] + " %" for s in series]],
            [[b, *[f"{s['points'].get(b, float('nan')):.0f}" for s in series]] for b in bins],
            "Data: pooled correct rate per operation-count bin",
        )
        sections.append(
            figure(
                fig_no[0],
                "Accuracy vs implication complexity",
                "Correct rate pooled over all models, by the implication's total "
                "operation count (both equations combined).",
                line_chart(series, bins, "total operations in the implication") + table,
            )
        )

    for run in group:
        counts_by = by_model_counts(run)
        rows = [
            {"label": short_model(m), "counts": counts_by[m]} for m in models
        ]
        rows.sort(key=lambda r: correct_pct_counts(r["counts"]), reverse=True)
        fig_no[0] += 1
        table = data_table(
            ["model", *[name for name, _ in COMPOSITION], "correct %"],
            [
                [r["label"], *[r["counts"].get(name, 0) for name, _ in COMPOSITION],
                 f"{correct_pct_counts(r['counts']):.1f}"]
                for r in rows
            ],
            "Data: verdict counts per model",
        )
        sections.append(
            figure(
                fig_no[0],
                f"Verdict composition — {run['label']} regime",
                "How each model's answers grade out; the right-hand figure is the "
                "correct rate (exact + swapped + dualized).",
                stacked_bars(rows) + table,
            )
        )
    return "".join(sections)


# ------------------------------------------------------------------ Page


CSS = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #1baf7a; --s3: #4a3aa7; --s4: #eb6834;
  --c-exact: #256abf; --c-swap: #6da7ec; --c-dual: #b7d3f6;
  --c-wrong: #e34948; --c-unp: #898781;
}
@media (prefers-color-scheme: dark) { :root {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #199e70; --s3: #9085e9; --s4: #d95926;
  --c-exact: #3987e5; --c-swap: #86b6ef; --c-dual: #cde2fb;
  --c-wrong: #e66767; --c-unp: #898781;
} }
:root[data-theme="light"] {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #1baf7a; --s3: #4a3aa7; --s4: #eb6834;
  --c-exact: #256abf; --c-swap: #6da7ec; --c-dual: #b7d3f6;
  --c-wrong: #e34948; --c-unp: #898781;
}
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #199e70; --s3: #9085e9; --s4: #d95926;
  --c-exact: #3987e5; --c-swap: #86b6ef; --c-dual: #cde2fb;
  --c-wrong: #e66767; --c-unp: #898781;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 940px; margin: 0 auto; padding: 40px 24px 80px; }
header h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; text-wrap: balance; }
header p.sub { color: var(--ink-2); margin: 0 0 8px; max-width: 65ch; }
header p.meta { color: var(--muted); font-size: 12.5px; margin: 0; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 28px 0 8px; }
.tile {
  flex: 1 1 150px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
}
.tile .label { color: var(--ink-2); font-size: 12.5px; }
.tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
h2 {
  font-size: 15px; color: var(--ink-2); font-weight: 600; margin: 44px 0 6px;
  padding-top: 20px; border-top: 1px solid var(--grid);
}
figure {
  margin: 18px 0 0; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px 20px 14px;
}
figcaption { margin-bottom: 12px; }
.fig-no { color: var(--muted); font-size: 12px; letter-spacing: 0.06em;
  text-transform: uppercase; margin-right: 6px; }
figcaption strong { font-weight: 600; }
figcaption .cap { color: var(--ink-2); font-size: 13px; }
svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--grid); stroke-width: 1; }
.tick, .axis-title { fill: var(--muted); font-size: 11px; }
.axis-title { font-size: 11.5px; }
.row-label { fill: var(--ink-2); font-size: 12.5px; }
.end-label { fill: var(--ink-2); font-size: 11.5px; font-weight: 600; }
.value { fill: var(--ink-2); font-size: 11px; font-variant-numeric: tabular-nums; }
.ring { fill: var(--surface); }
.connector { stroke: var(--grid); stroke-width: 2; stroke-linecap: round; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 0 0 10px; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px;
  color: var(--ink-2); }
.swatch { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
details { margin-top: 10px; border-top: 1px solid var(--grid); padding-top: 8px; }
summary { color: var(--muted); font-size: 12.5px; cursor: pointer; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; margin-top: 8px; font-size: 13px; width: 100%; }
th, td { text-align: left; padding: 4px 12px 4px 0; border-bottom: 1px solid var(--grid); }
td { font-variant-numeric: tabular-nums; color: var(--ink-2); }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
#tooltip {
  position: fixed; pointer-events: none; background: var(--ink); color: var(--page);
  padding: 5px 9px; border-radius: 6px; font-size: 12px; opacity: 0;
  transition: opacity 80ms; z-index: 10; max-width: 320px;
}
@media (prefers-reduced-motion: reduce) { #tooltip { transition: none; } }
[data-tip]:focus { outline: 2px solid var(--s1); outline-offset: 2px; }
"""

JS = """
const tip = document.getElementById('tooltip');
function show(e, el) {
  tip.textContent = el.getAttribute('data-tip');
  tip.style.opacity = 1;
  const x = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
}
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => show(e, el));
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
  el.addEventListener('focus', () => {
    const r = el.getBoundingClientRect();
    show({clientX: r.left + r.width / 2, clientY: r.bottom}, el);
  });
  el.addEventListener('blur', () => tip.style.opacity = 0);
});
"""


def render_report(runs: List[dict], title: str) -> str:
    total_rows = sum(len(r["rows"]) for r in runs)
    models = sorted({m for r in runs for m in r["models"]})
    stories = len({pid for r in runs for pid in r["pair_ids"]})
    overall = correct_pct([row for r in runs for row in r["rows"]])
    tiles = "".join(
        f'<div class="tile"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div></div>'
        for label, value in (
            ("Runs", len(runs)),
            ("Models", len(models)),
            ("Stories", stories),
            ("Graded calls", f"{total_rows:,}"),
            ("Overall correct", f"{overall:.0f}%"),
        )
    )
    fig_no = [0]
    body = "".join(render_group(g, fig_no) for g in group_runs(runs))
    run_list = " · ".join(esc(str(r["dir"])) for r in runs)
    return f"""<title>{esc(title)}</title>
<style>{CSS}</style>
<main>
<header>
<h1>{esc(title)}</h1>
<p class="sub">Stories generated from Equational Theories Project implications
(<code>storyform.py</code>), formalized back by each model over OpenRouter, and
graded syntactically (<code>checkform.py</code>). Correct = exact, or an accepted
symmetry (sides swapped, or both equations uniformly dualized).</p>
<p class="meta">{run_list}</p>
</header>
<div class="tiles">{tiles}</div>
{body}
</main>
<div id="tooltip" role="status"></div>
<script>{JS}</script>
"""


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(
        description="Render benchmark.py run directories into an HTML chart report."
    )
    cli.add_argument("run_dirs", nargs="+", type=Path)
    cli.add_argument("--out", type=Path, default=Path("results/report.html"))
    cli.add_argument("--title", default="ETP story-formalization benchmark")
    args = cli.parse_args(argv)

    runs = [load_run(d) for d in args.run_dirs]
    report = render_report(runs, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
