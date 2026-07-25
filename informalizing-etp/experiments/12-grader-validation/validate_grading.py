#!/usr/bin/env python3
"""Drift-anatomy validation: does the 8:1 silent:loud headline survive the
grader's own degrees of freedom?

V-A  Grader-variant specification curve. Every committed response is
     re-graded under variants of the grading specification:
       replay            exact checkform.grade (must reproduce stored verdicts)
       strict_convention no accepted transforms (exact sides, exact argument order)
       no_dual           swaps accepted, dualization not
       first_line        extraction takes the FIRST ASSUME/ASK lines, not the last
       plain_lines       extraction without markdown-decoration tolerance
     For each variant x run x model: accuracy, silent (wrong) share, loud
     (unparseable) share. The headline claims that must survive: silent >>
     loud, and the model ranking / complexity trend are stable.

V-B  Rescue grading of silent failures. Every default-wrong response is
     probed with diagnostic (deliberately ILLEGAL) transforms:
       direction   answer matches with ASSUME and ASK exchanged
       onesided_dual  dual applied to exactly one equation matches
       e_only      ASSUME equation correct (under legal transforms), ASK wrong
       f_only      ASK correct, ASSUME wrong
       structural  none of the above -- genuine structural drift
     If most silent failures were near-misses, the drift story would be
     overclaimed; if they are structural, it stands.

Reads run dirs given on the CLI (defaults: experiments 07 + 09), writes
report.json + summary.md next to this script. Pure stdlib + repo modules;
no network, deterministic.
"""

import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # informalizing-etp/
sys.path.insert(0, str(REPO))

import checkform  # noqa: E402
from checkform import (  # noqa: E402
    AnswerParseError,
    _TRANSFORMS,
    _clean,
    _transformed,
    dual,
    parse_prefix_equation,
)
from storyform import canonical  # noqa: E402

DEFAULT_RUNS = sorted(
    itertools.chain(
        (REPO / "experiments" / "07-two-stage-scale" / "runs").iterdir(),
        (REPO / "experiments" / "09-synthetic-complexity" / "runs").iterdir(),
    )
)

# ---------------------------------------------------------- extraction variants

_LINE_RE_LAST = checkform._LINE_RE
_LINE_RE_PLAIN = re.compile(
    r"^\s*(?P<label>ASSUME|ASK)\b\s*:\s*(?P<value>.+)$", re.IGNORECASE | re.MULTILINE
)


def extract(response, which):
    found = {}
    regex = _LINE_RE_PLAIN if which == "plain_lines" else _LINE_RE_LAST
    for m in regex.finditer(response):
        label = m.group("label").upper()
        if which == "first_line" and label in found:
            continue
        found[label] = _clean(m.group("value"))
    for label in ("ASSUME", "ASK"):
        if label not in found:
            raise AnswerParseError(f"no '{label}:' line")
    return found["ASSUME"], found["ASK"]


def grade_variant(response, truth, variant):
    """Return 'correct' | 'wrong' | 'unparseable' under a grading variant."""
    extraction = variant if variant in ("first_line", "plain_lines") else "last"
    try:
        a_text, k_text = extract(response, extraction)
        e = parse_prefix_equation(a_text)
        f = parse_prefix_equation(k_text)
    except AnswerParseError:
        return "unparseable"
    if variant == "strict_convention":
        transforms = [(False, False, False)]
    elif variant == "no_dual":
        transforms = [t for t in _TRANSFORMS if not t[2]]
    else:
        transforms = _TRANSFORMS
    for swap_e, swap_f, dualize in transforms:
        te, tf = _transformed(e, f, swap_e, swap_f, dualize)
        if (canonical(*te), canonical(*tf)) == truth:
            return "correct"
    return "wrong"


# ---------------------------------------------------------------- V-B probes


def rescue_probe(response, truth):
    """Diagnostic taxonomy for a default-wrong answer."""
    a_text, k_text = extract(response, "last")
    e = parse_prefix_equation(a_text)
    f = parse_prefix_equation(k_text)
    out = []

    def matches(e_, f_, transforms=_TRANSFORMS):
        return any(
            (canonical(*te), canonical(*tf)) == truth
            for t in transforms
            for te, tf in [_transformed(e_, f_, *t)]
        )

    if matches(f, e):
        out.append("direction")
    for de, df in (((dual(e[0]), dual(e[1])), f), (e, (dual(f[0]), dual(f[1])))):
        # one-sided dual, still allowing swaps (but no further dual)
        if any(
            (canonical(*te), canonical(*tf)) == truth
            for t in [(s1, s2, False) for s1 in (False, True) for s2 in (False, True)]
            for te, tf in [_transformed(de, df, *t)]
        ):
            out.append("onesided_dual")
            break
    e_ok = any(
        canonical(*_transformed(e, f, s, False, d)[0]) == truth[0]
        for s in (False, True) for d in (False, True)
    )
    f_ok = any(
        canonical(*_transformed(e, f, False, s, d)[1]) == truth[1]
        for s in (False, True) for d in (False, True)
    )
    if e_ok and not f_ok:
        out.append("f_wrong_only")
    if f_ok and not e_ok:
        out.append("e_wrong_only")
    if not out:
        out.append("structural")
    return out


# --------------------------------------------------------------------- main


def load_run(run_dir):
    run_dir = Path(run_dir)
    truth_by_pair = {}
    with open(run_dir / "samples.jsonl") as fh:
        for line in fh:
            s = json.loads(line)
            truth_by_pair[s["pair_id"]] = (
                s["metadata"]["canonical_e"],
                s["metadata"]["canonical_f"],
            )
    rows = []
    with open(run_dir / "results.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("api_error"):
                continue
            rows.append(r)
    return truth_by_pair, rows


VARIANTS = ["replay", "strict_convention", "no_dual", "first_line", "plain_lines"]


def main():
    run_dirs = [Path(a) for a in sys.argv[1:]] or DEFAULT_RUNS
    report = {"runs": {}, "va_by_model": defaultdict(lambda: defaultdict(Counter)),
              "vb": Counter(), "vb_by_form": defaultdict(Counter),
              "vb_by_model": defaultdict(Counter)}
    replay_mismatch = 0
    total = 0

    va = defaultdict(Counter)          # variant -> status counter (pooled)
    va_by_bin = defaultdict(lambda: defaultdict(Counter))  # variant -> ops_bin -> status
    va_model_acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # variant -> model -> [correct, n]

    for run_dir in run_dirs:
        truth_by_pair, rows = load_run(run_dir)
        run_name = f"{run_dir.parent.parent.name}/{run_dir.name}"
        run_counts = defaultdict(Counter)
        for r in rows:
            total += 1
            truth = truth_by_pair[r["pair_id"]]
            stored = r["verdict"]["status"]
            ops_bin = min((r.get("ops_total", 2) - 1) // 2, 7)
            for variant in VARIANTS:
                s = grade_variant(r["response"], truth, variant)
                va[variant][s] += 1
                va_by_bin[variant][ops_bin][s] += 1
                acc = va_model_acc[variant][r["model"]]
                acc[0] += s == "correct"
                acc[1] += 1
                run_counts[variant][s] += 1
                if variant == "replay" and s != stored:
                    replay_mismatch += 1
            if stored == "wrong":
                for tag in rescue_probe(r["response"], truth):
                    report["vb"][tag] += 1
                    report["vb_by_form"][r.get("form", "?")][tag] += 1
                    report["vb_by_model"][r["model"]][tag] += 1
        report["runs"][run_name] = {v: dict(c) for v, c in run_counts.items()}

    # ------- assemble summary -------
    lines = ["# Experiment 12: grader validation (V-A specification curve, V-B rescue grading)",
             "",
             f"Re-graded {total} committed responses from {len(run_dirs)} run dirs, offline.",
             f"Replay fidelity: {total - replay_mismatch}/{total} verdicts identical "
             f"({replay_mismatch} mismatches).", "",
             "## V-A: does the headline survive the grader's degrees of freedom?", "",
             "| variant | correct | wrong (silent) | unparseable (loud) | silent:loud |",
             "|---|---|---|---|---|"]
    summary_va = {}
    for v in VARIANTS:
        c, w, u = va[v]["correct"], va[v]["wrong"], va[v]["unparseable"]
        ratio = w / u if u else float("inf")
        summary_va[v] = dict(correct=c, wrong=w, unparseable=u, silent_loud=round(ratio, 2))
        lines.append(f"| {v} | {c} | {w} | {u} | {ratio:.1f} |")
    report["va_pooled"] = summary_va

    # model-ranking stability: Spearman of model accuracy, each variant vs replay
    def rank(d):
        order = sorted(d, key=lambda m: -(d[m][0] / max(d[m][1], 1)))
        return {m: i for i, m in enumerate(order)}

    def spearman(r1, r2):
        ms = sorted(set(r1) & set(r2))
        n = len(ms)
        if n < 3:
            return float("nan")
        dsum = sum((r1[m] - r2[m]) ** 2 for m in ms)
        return 1 - 6 * dsum / (n * (n * n - 1))

    base_rank = rank(va_model_acc["replay"])
    lines += ["", "Model-ranking stability (Spearman rho vs replay):", ""]
    report["va_rank_stability"] = {}
    for v in VARIANTS[1:]:
        rho = spearman(base_rank, rank(va_model_acc[v]))
        report["va_rank_stability"][v] = round(rho, 4)
        lines.append(f"- {v}: rho = {rho:.3f}")

    # silent:loud per complexity bin under each variant
    lines += ["", "Silent:loud ratio by ops bin (bin = (ops_total-1)//2, capped 7):", "",
              "| variant | " + " | ".join(f"bin{b}" for b in range(8)) + " |",
              "|" + "---|" * 9]
    for v in VARIANTS:
        cells = []
        for b in range(8):
            w, u = va_by_bin[v][b]["wrong"], va_by_bin[v][b]["unparseable"]
            cells.append(f"{w}/{u}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    report["va_by_bin"] = {v: {b: dict(va_by_bin[v][b]) for b in va_by_bin[v]} for v in va_by_bin}

    # ------- V-B -------
    n_wrong = sum(report["vb"].values()) and None
    total_wrong = sum(v for k, v in report["vb"].items() if k == "structural") + 0
    all_tags = report["vb"]
    n_rows_wrong = sum(all_tags.values())  # note: multi-tag rows counted per tag
    lines += ["", "## V-B: anatomy of silent failures (default-wrong answers)", "",
              "Tags are diagnostic, deliberately-illegal rescues; a row can carry "
              "several tags; 'structural' = no near-miss explains it.", "",
              "| tag | count |", "|---|---|"]
    for tag, cnt in all_tags.most_common():
        lines.append(f"| {tag} | {cnt} |")
    lines += ["", "By form:", ""]
    for form, c in sorted(report["vb_by_form"].items()):
        tot = sum(c.values())
        struct = c.get("structural", 0)
        lines.append(f"- {form}: structural {struct}/{tot} tags "
                     f"({100*struct/max(tot,1):.0f}%)")

    report["vb"] = dict(report["vb"])
    report["vb_by_form"] = {k: dict(v) for k, v in report["vb_by_form"].items()}
    report["vb_by_model"] = {k: dict(v) for k, v in report["vb_by_model"].items()}
    report["va_by_model"] = {
        v: {m: dict(correct=a[0], n=a[1]) for m, a in d.items()}
        for v, d in va_model_acc.items()
    }
    report["replay_mismatch"] = replay_mismatch
    report["total"] = total

    (HERE / "report.json").write_text(json.dumps(report, indent=1, default=dict))
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
