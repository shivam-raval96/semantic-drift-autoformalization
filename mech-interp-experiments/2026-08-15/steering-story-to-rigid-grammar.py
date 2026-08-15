#!/usr/bin/env python3
"""Does pushing activations toward the answer notation change what gets written?

QUESTION
--------
The earlier steering experiment fitted a direction between two ways of *stating*
a problem — a themed story and a plain-English description — and adding it to
the residual stream changed accuracy by nothing. One reading of that null is
that the direction was too weak a lever: both endpoints are ordinary English
prose about the same content, so whatever separates them may be superficial.

This fits the direction between the story and Rigid Grammar instead. Rigid
Grammar is the terse formal notation the task asks for as an answer, two lines
of the form `ASSUME: ...` / `ASK: ...` with no prose at all. It is a far bigger
representational jump than story-to-plain-English, and it points at the thing
the model is being asked to produce rather than at another way of reading the
question.

That difference in kind changes what a result would mean. Story-to-plain-English
could at best buy the gap between two input forms. Story-to-Rigid-Grammar points
at the answer itself, so if the direction were causally used the ceiling is
every problem the model currently gets wrong.

Both regimes are run: no reasoning at all, and 512 reasoning tokens.

WHAT EACH OUTCOME WOULD MEAN
----------------------------
- Accuracy rises with dose and the random control does not: a direction fitted
  purely from surface form is doing work the model uses to answer, which would
  be the first causal result in this line and would justify a proper injection
  study.
- Nothing moves at any dose that leaves the output coherent: the second and
  larger form direction is also inert, and the earlier null generalizes from
  one direction to the class. That is a real finding about the method, not an
  absence of one.
- The output becomes better-formed — more answers parse as `ASSUME:`/`ASK:` —
  without becoming more correct: the direction carries the notation and not the
  content. This is the outcome most specific to this vector, and it is worth
  reporting on its own; it would say the model keeps form and meaning in
  separable directions.
- Coherence collapses at the smallest dose that registers at all: the direction
  is too far outside the model's working range to test this way, and the next
  step is a narrower injection site rather than a smaller dose.
- The effect appears with no reasoning but vanishes at 512 tokens: reasoning
  overwrites whatever the injection put there, which would bound how long an
  injected representation survives.

CONTROLS AND WHAT LIMITS THIS
-----------------------------
- A random direction of the same length and the negated vector, at the same
  layer, dose and budget. A response symmetric in +v and -v is a disturbance
  response, not a direction response.
- Dose is expressed as a fraction of the typical residual-stream length at that
  layer, not as a multiple of the raw vector. The story and Rigid Grammar
  renderings differ enormously in length, so their mean difference is large in a
  way that says nothing about how hard it should be pushed; scaling by the
  residual norm makes "dose 0.2" mean the same thing at any layer.
- Every arm shares its problems with the untouched baseline, so all comparisons
  are paired and use McNemar's test.
- The share of answers that fail to parse is reported beside every cell, since
  accuracy alone cannot separate "answered wrongly" from "stopped producing
  answers".
- There is no separate ceiling arm here, unlike the story-to-plain-English
  experiment, and the reason is worth stating: feeding the model Rigid Grammar
  as *input* would be handing it the answer. The direction points at the target,
  so the room available is simply everything the untouched model gets wrong.
- The vector is fitted on a disjoint half of the problems.
- What this cannot show: Rigid Grammar texts are short and stories are long, so
  the fitted direction certainly contains a length and register component. A
  positive result would need a follow-up separating "formal notation" from
  "short and terse" before it could be called a notation direction.

    python3 steering-story-to-rigid-grammar.py --quick        # smoke test
    python3 steering-story-to-rigid-grammar.py                # the real run
    python3 steering-story-to-rigid-grammar.py --analyze-only # redo tables
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import dataset, grading, runs, stats
from shared.vendor import ensure_on_path

ensure_on_path()

from benchmark import wrap_prompt  # noqa: E402

EXPERIMENT = "steering-story-to-rigid-grammar"

DEFAULT_BUDGETS = (0, 512)
DEFAULT_LAYER = 24

# Doses as fractions of the median residual-stream norm at the injection layer.
# 0.1 is a light touch, 0.4 is heavy enough that the earlier work would expect
# coherence to be at risk; running both is how the usable range gets found.
DEFAULT_ALPHAS = (0.1, 0.2, 0.4)
DEFAULT_CONTROL_ALPHA = 0.2

DEFAULT_PER_BIN = 20
DEFAULT_ANSWER_TOKENS = 512
CONTROL_SEED = 1234


def build_conditions(
    budgets: Sequence[int],
    layer: int,
    alphas: Sequence[float],
    control_alpha: float,
    per_bin: int,
) -> List[runs.Condition]:
    """The sweep, as data: an untouched baseline, a dose series, two controls."""
    conditions: List[runs.Condition] = []
    shared = dict(layer=layer, per_bin=per_bin)
    for budget in budgets:
        conditions.append(
            runs.Condition(
                "untouched_B{}".format(budget), arm="untouched", budget=budget, alpha=0.0, **shared
            )
        )
        for alpha in alphas:
            conditions.append(
                runs.Condition(
                    "steer_a{:g}_B{}".format(alpha, budget),
                    arm="steer",
                    budget=budget,
                    alpha=float(alpha),
                    **shared
                )
            )
        for arm in ("random", "negated"):
            conditions.append(
                runs.Condition(
                    "{}_a{:g}_B{}".format(arm, control_alpha, budget),
                    arm=arm,
                    budget=budget,
                    alpha=float(control_alpha),
                    **shared
                )
            )
    return conditions


# ------------------------------------------------------------------ the vector


def fit_direction(model, tokenizer, fit_samples, layer: int, batch_size: int):
    """Mean activation on Rigid Grammar texts minus mean on story texts.

    Both read at the last token of the bare rendering. The returned vector is
    rescaled to the median length of a residual-stream vector at this layer, so
    the dose applied later is a plain fraction of that length: dose 0.2 adds a
    vector one fifth as long as the activations it is being added to.
    """
    from shared import hooks

    stories = [sample["story"] for sample in fit_samples]
    rigid = [dataset.render_all_forms(sample["metadata"])["rg"] for sample in fit_samples]

    story_acts = hooks.capture_residuals(
        model, tokenizer, stories, [layer], batch_size=batch_size, progress="fit: story"
    )["last"][layer]
    rigid_acts = hooks.capture_residuals(
        model, tokenizer, rigid, [layer], batch_size=batch_size, progress="fit: rigid grammar"
    )["last"][layer]

    raw = hooks.contrast_vector(rigid_acts, story_acts)
    residual_norm = float(story_acts.norm(dim=-1).median())
    unit_scaled = raw / float(raw.norm()) * residual_norm
    report = {
        "layer": layer,
        "raw_vector_norm": float(raw.norm()),
        "median_residual_norm": residual_norm,
        "raw_relative_size": float(raw.norm()) / residual_norm if residual_norm else None,
        "coherence": _coherence(rigid_acts - story_acts),
        "n_fit_pairs": len(fit_samples),
        "example_rigid_grammar": rigid[0] if rigid else None,
    }
    return unit_scaled, report


def _coherence(differences) -> float:
    """Mean pairwise cosine between the per-problem differences.

    A direction averaged from differences that point every which way is not a
    direction; this says how much they agree.
    """
    import torch

    unit = torch.nn.functional.normalize(differences.float(), dim=-1)
    n = unit.shape[0]
    if n < 2:
        return float("nan")
    sims = unit @ unit.T
    return float((sims.sum() - sims.diagonal().sum()) / (n * (n - 1)))


def intervention_for(condition: runs.Condition, model, vector):
    from shared import hooks

    arm = condition["arm"]
    if arm == "untouched":
        return None
    patch = hooks.ResidualAdd(model, condition["layer"], vector, condition["alpha"])
    if arm == "steer":
        return patch
    if arm == "random":
        return patch.randomized(CONTROL_SEED)
    if arm == "negated":
        return patch.negated()
    raise ValueError("unknown arm {!r}".format(arm))


# ------------------------------------------------------------------ generation


def run_experiment(args, run: runs.RunDirectory, conditions: List[runs.Condition]) -> None:
    from shared import generation, model as model_module

    todo = runs.pending(run, conditions, force=args.force)
    if not todo:
        print("every condition already has records; nothing to generate")
        return

    print(runs.describe(conditions, run))
    print("\ngenerating {} of {} conditions".format(len(todo), len(conditions)))

    samples = dataset.sample_etp_stratified(args.per_bin, args.seed, form="story")
    fit_samples, eval_samples = dataset.split_alternating(samples)
    print(
        "{} pairs: {} to fit the direction, {} held out for grading".format(
            len(samples), len(fit_samples), len(eval_samples)
        )
    )

    model, tokenizer = model_module.load(args.model)
    model_module.check_layers(model, [args.layer])
    generator = generation.Generator(model, tokenizer, batch_size=args.batch_size)

    vector, report = fit_direction(model, tokenizer, fit_samples, args.layer, args.batch_size)
    print(
        "\ndirection at layer {}: raw norm {:.1f}, which is {:.1f} times the "
        "median residual norm; per-pair coherence {:.2f}".format(
            report["layer"], report["raw_vector_norm"], report["raw_relative_size"],
            report["coherence"],
        )
    )
    (run.path / "direction.json").write_text(json.dumps(report, indent=2) + "\n")

    prompts = [wrap_prompt(s["prompt"], "on", "", "story") for s in eval_samples]

    for condition in todo:
        model_module.set_seed(args.seed)
        completions = generator.generate_budgeted(
            prompts,
            condition["budget"],
            max_answer_tokens=args.answer_tokens,
            intervention=intervention_for(condition, model, vector),
            progress=condition.name,
        )
        with run.writing(condition.name) as writer:
            for sample, completion in zip(eval_samples, completions):
                writer.write(
                    grading.grade_record(
                        completion.answer,
                        sample,
                        extra=dict(
                            condition.settings,
                            condition=condition.name,
                            **completion.as_dict()
                        ),
                    )
                )
        rows = run.read(condition.name)
        print(
            "  {:<24} accuracy {}  unparseable {}".format(
                condition.name, grading.correct_rate(rows), grading.unparseable_rate(rows)
            )
        )


# -------------------------------------------------------------------- analysis


def analyze(run: runs.RunDirectory, conditions: List[runs.Condition]) -> dict:
    finished = [c for c in conditions if run.has(c.name)]
    if not finished:
        raise SystemExit(
            "no condition has records yet in {}; run without --analyze-only "
            "first".format(run.path)
        )
    if len(finished) < len(conditions):
        print(
            "WARNING: analysing {} of {} conditions; the rest have not "
            "run".format(len(finished), len(conditions))
        )

    records = {c.name: run.read(c.name) for c in finished}
    summary: dict = {"budgets": {}}

    for budget in sorted({c["budget"] for c in finished}):
        at_budget = [c for c in finished if c["budget"] == budget]
        block: dict = {"cells": {}}
        for condition in at_budget:
            rows = records[condition.name]
            block["cells"][condition.name] = {
                "arm": condition["arm"],
                "alpha": condition["alpha"],
                "accuracy": grading.correct_rate(rows).as_dict(),
                "unparseable": grading.unparseable_rate(rows).as_dict(),
                "buckets": grading.bucket_counts(rows),
                "wrote_required_notation": stats.rate(
                    sum(1 for r in rows if r["status"] != "unparseable"), len(rows)
                ).as_dict(),
                "think_tokens_mean": _mean(r["think_tokens"] for r in rows),
            }

        baseline = "untouched_B{}".format(budget)
        if baseline in records:
            floor_rate = grading.correct_rate(records[baseline])
            block["room_available"] = 1.0 - floor_rate.value
            block["vs_untouched"] = {
                condition.name: dict(
                    stats.pair_records(records[baseline], records[condition.name]).as_dict(),
                    notation_delta=(
                        block["cells"][condition.name]["wrote_required_notation"]["rate"]
                        - block["cells"][baseline]["wrote_required_notation"]["rate"]
                    ),
                )
                for condition in at_budget
                if condition.name != baseline
            }
        summary["budgets"][str(budget)] = block

    summary["dose_response"] = dose_response(summary)
    print_summary(summary)
    return summary


def dose_response(summary: dict) -> dict:
    """Is accuracy ordered by dose, and is the random control ordered too?

    A steering result that is monotone in dose is much harder to explain away
    than one cell that happened to differ; a control that is monotone in dose
    the same way means the ordering is a disturbance effect.
    """
    out: Dict[str, dict] = {}
    for budget, block in summary["budgets"].items():
        by_arm: Dict[str, List[tuple]] = {}
        for cell in block["cells"].values():
            by_arm.setdefault(cell["arm"], []).append(
                (cell["alpha"], cell["accuracy"]["rate"])
            )
        entry = {}
        for arm, points in by_arm.items():
            if len(points) >= 3:
                points.sort()
                entry[arm] = stats.spearman(
                    [p[0] for p in points], [p[1] for p in points]
                )
        if entry:
            out[budget] = entry
    return out


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------- output


def print_summary(summary: dict) -> None:
    for budget, block in sorted(summary["budgets"].items(), key=lambda kv: int(kv[0])):
        print("\nthinking budget {} tokens".format(budget))
        print(
            "  {:<24} {:>20} {:>14} {:>18}".format(
                "condition", "accuracy", "95% CI", "wrote the notation"
            )
        )
        for name, cell in sorted(block["cells"].items()):
            accuracy = cell["accuracy"]
            print(
                "  {:<24} {:>20} {:>14} {:>18.0%}".format(
                    name,
                    "{:.1%} ({}/{})".format(
                        accuracy["rate"], accuracy["successes"], accuracy["total"]
                    ),
                    "{:.0%}-{:.0%}".format(*accuracy["ci95"]),
                    cell["wrote_required_notation"]["rate"],
                )
            )
        if "room_available" in block:
            print(
                "  the untouched model gets {:.0%} of these problems wrong, so "
                "that is the room a working direction could win".format(
                    block["room_available"]
                )
            )
        for name, comparison in sorted(block.get("vs_untouched", {}).items()):
            print(
                "  {:<24} vs untouched: {:+.1f} points ({} gained, {} lost, "
                "McNemar p = {:.3g}), notation {:+.1f} points".format(
                    name,
                    100 * comparison["delta"],
                    comparison["gained"],
                    comparison["lost"],
                    comparison["mcnemar_p"],
                    100 * comparison["notation_delta"],
                )
            )

    if summary.get("dose_response"):
        print("\nrank correlation between dose and accuracy (1 means strictly rising)")
        for budget, arms in sorted(summary["dose_response"].items(), key=lambda kv: int(kv[0])):
            for arm, value in sorted(arms.items()):
                print(
                    "  budget {:>4}, {:<9} {}".format(
                        budget, arm, "n/a" if value is None else "{:+.2f}".format(value)
                    )
                )


def make_figure(summary: dict, run: runs.RunDirectory) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the figure")
        return None

    budgets = sorted(summary["budgets"], key=int)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colours = {0: "#d95f02", 512: "#1f77b4"}

    for budget in budgets:
        cells = summary["budgets"][budget]["cells"]
        steered = sorted(
            ((c["alpha"], c) for c in cells.values() if c["arm"] == "steer"),
            key=lambda pair: pair[0],
        )
        untouched = [c for c in cells.values() if c["arm"] == "untouched"]
        colour = colours.get(int(budget))

        doses = [a for a, _ in steered]
        for axis, key in ((axes[0], "accuracy"), (axes[1], "wrote_required_notation")):
            values = [c[key]["rate"] for _, c in steered]
            low = [v - c[key]["ci95"][0] for v, (_, c) in zip(values, steered)]
            high = [c[key]["ci95"][1] - v for v, (_, c) in zip(values, steered)]
            axis.errorbar(
                doses, values, yerr=[low, high], marker="o", capsize=3, color=colour,
                label="budget {} tokens".format(budget),
            )
            if untouched:
                axis.axhline(
                    untouched[0][key]["rate"], linestyle="--", linewidth=1, color=colour,
                    alpha=0.6,
                )
        for arm, marker in (("random", "x"), ("negated", "s")):
            control = [c for c in cells.values() if c["arm"] == arm]
            if control:
                axes[0].scatter(
                    [control[0]["alpha"]], [control[0]["accuracy"]["rate"]],
                    marker=marker, color=colour, zorder=5,
                    label="{}, budget {}".format(arm, budget),
                )

    axes[0].set_ylabel("share of answers correct")
    axes[0].set_title("Accuracy against dose\n(dashed line: the untouched model)")
    axes[1].set_ylabel("share of answers in the required ASSUME/ASK notation")
    axes[1].set_title("Output format against dose")
    for axis in axes:
        axis.set_xlabel("dose, as a fraction of the typical residual-stream length")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.suptitle("Steering from story toward the Rigid Grammar answer notation")
    fig.tight_layout()
    path = run.figure_path("{}.png".format(EXPERIMENT))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------ main


def main(argv: Optional[List[str]] = None) -> int:
    cli = runs.base_parser(__doc__.split("\n")[0])
    cli.add_argument(
        "--budgets",
        type=lambda text: tuple(int(b) for b in text.split(",")),
        default=DEFAULT_BUDGETS,
        help="thinking-token budgets, comma separated (default: %(default)s)",
    )
    cli.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="residual-stream layer to fit and inject at (default: %(default)s)",
    )
    cli.add_argument(
        "--alphas",
        type=lambda text: tuple(float(a) for a in text.split(",")),
        default=DEFAULT_ALPHAS,
        help="doses as fractions of the median residual-stream norm "
        "(default: %(default)s)",
    )
    cli.add_argument(
        "--control-alpha",
        type=float,
        default=DEFAULT_CONTROL_ALPHA,
        help="dose the random and negated controls run at (default: %(default)s)",
    )
    cli.add_argument(
        "--per-bin",
        type=int,
        default=DEFAULT_PER_BIN,
        help="pairs sampled per operation-count bin, before the fit/eval split "
        "(default: %(default)s)",
    )
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument(
        "--answer-tokens",
        type=int,
        default=DEFAULT_ANSWER_TOKENS,
        help="cap on the answer, after the thinking budget (default: %(default)s)",
    )
    args = cli.parse_args(argv)

    if args.quick:
        args.per_bin = min(args.per_bin, 2)
        args.budgets = (0, 32)
        args.alphas = (0.2,)
        # Enough room for an answer to finish. Below roughly 256 tokens the
        # model gets cut off mid-answer and every row grades as unparseable,
        # which would leave the grading path untested by the smoke run.
        args.answer_tokens = min(args.answer_tokens, 256)

    conditions = build_conditions(
        args.budgets, args.layer, args.alphas, args.control_alpha, args.per_bin
    )
    out_dir = runs.resolve_out_dir(args, __file__)
    run = runs.RunDirectory.open(
        out_dir,
        EXPERIMENT,
        {
            "model": args.model,
            "seed": args.seed,
            "budgets": list(args.budgets),
            "layer": args.layer,
            "alphas": list(args.alphas),
            "control_alpha": args.control_alpha,
            "per_bin": args.per_bin,
            "answer_tokens": args.answer_tokens,
            "control_seed": CONTROL_SEED,
            "conditions": [c.as_dict() for c in conditions],
        },
        force=args.force,
    )

    if not args.analyze_only:
        run_experiment(args, run, conditions)

    summary = analyze(run, conditions)
    path = run.write_summary(summary)
    print("\nsummary -> {}".format(path))
    figure = make_figure(summary, run)
    if figure:
        print("figure  -> {}".format(figure))
    return 0


if __name__ == "__main__":
    sys.exit(main())
