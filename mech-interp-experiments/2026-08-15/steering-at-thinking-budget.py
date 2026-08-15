#!/usr/bin/env python3
"""Does the story-to-literal steering direction help once the model can think?

QUESTION
--------
An earlier experiment fitted a contrastive direction — the mean difference
between the model's activations on plain-English renderings of a problem and on
themed-story renderings of the same problem — and added it to the residual
stream while the model answered. With no reasoning allowed, it changed accuracy
by exactly nothing: the best surviving cell scored 9.6%, the untouched model's
9.6%, and a random direction of the same length scored 9.6% too.

That null was measured where almost nothing could have been detected. A
direction that perfectly converted the story representation into the
plain-English one can only ever buy what separates those two inputs, and with
no reasoning that is 9.6% against 19.2%, under ten points. At a budget of 512
reasoning tokens the same two inputs score 38.5% and 96.2%, so the same
intervention has almost sixty points of room. This runs the identical test
where it can actually resolve an effect.

WHAT EACH OUTCOME WOULD MEAN
----------------------------
- Steered accuracy at budget 512 rises toward the plain-English arm and the
  random control does not: the direction does something real, and the earlier
  null was a power problem rather than a finding. The direction acts as a
  partial substitute for reasoning — same accuracy from fewer tokens.
- Steered, random and untouched are indistinguishable at 512 as well: the null
  survives at six times the resolution, and "represent this abstractly" can be
  dropped as a candidate mechanism rather than left open.
- Steered beats untouched at 512 but the random control moves just as much:
  what is being measured is disturbance, not direction. This is the outcome the
  controls exist to catch.
- Accuracy rises at 512 but not at 0: the direction is a premise the model has
  to use, worthless without a trace long enough to exploit it.
- Everything collapses into repetition at every dose that registers: the
  intervention is too heavy to sit inside the model's working range, and the
  finding is about the injection method rather than about the direction.

CONTROLS AND WHAT LIMITS THIS
-----------------------------
- A random direction of the same length, and the negated vector, both at the
  same layer, dose and budget. A response symmetric in +v and -v means the
  measurement is of how hard the activations were pushed, not of what the
  direction means.
- The plain-English arm is run here rather than quoted from the earlier report,
  so the headroom a perfect form conversion could buy is measured on the same
  problems, in the same run, with the same grader.
- Untouched and steered arms share their problems, so every comparison is
  paired and uses McNemar's test.
- Accuracy alone cannot tell "the intervention broke the output" from "the
  intervention gave a wrong answer", so the unparseable rate is reported beside
  every cell.
- The injection site is held fixed at every token position, prompt and
  generation alike, which is what the earlier run did. That is a heavier
  intervention than the literature usually applies and is the most likely
  reason coherence dies at small doses; it is held fixed here so that budget is
  the only thing that changed. Varying the site is a separate experiment.
- The direction is fitted on a disjoint half of the problems, so nothing about
  a graded problem's own answer enters the vector.

    python3 steering-at-thinking-budget.py --quick        # smoke test
    python3 steering-at-thinking-budget.py                # the real run
    python3 steering-at-thinking-budget.py --analyze-only # redo tables/figures
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import dataset, grading, runs, stats
from shared.vendor import ensure_on_path

ensure_on_path()

from benchmark import wrap_prompt  # noqa: E402

EXPERIMENT = "steering-at-thinking-budget"

# 512 is where the two input forms are furthest apart, so it is where a form
# intervention has the most room; 0 repeats the earlier null's regime on the
# same problems, which anchors the curve at both ends.
DEFAULT_BUDGETS = (0, 512)

# Layer 24 was the earlier sweep's best surviving cell. Alpha multiplies the
# raw mean-difference vector, exactly as before, so alpha 1 here means what
# alpha 1 meant there; the vector's size relative to the residual stream is
# recorded in the summary rather than left implicit.
DEFAULT_LAYER = 24

# At alpha 1 this vector is about a third of the typical residual-stream
# length, and a smoke run at that dose left the steered *and* the random arms
# unable to produce a readable answer at all — which measures disturbance, not
# direction. 0.25 and 0.5 are there so the series reaches doses light enough to
# leave the output intact, and alpha 1 is kept because it is what the earlier
# null was measured at.
DEFAULT_ALPHAS = (0.25, 0.5, 1.0)

DEFAULT_PER_BIN = 20
DEFAULT_ANSWER_TOKENS = 512
CONTROL_SEED = 1234


def build_conditions(
    budgets: Sequence[int],
    layer: int,
    alphas: Sequence[float],
    per_bin: int,
) -> List[runs.Condition]:
    """The sweep, as data.

    Four arms per budget: the untouched story baseline, the plain-English arm
    that sets the ceiling a form intervention could reach, the steered cells,
    and the two controls.

    The controls run at every dose the steered arm runs at, rather than at one
    representative dose. A dose big enough to change anything is also big enough
    to disturb the model, and which dose that is cannot be known before the run;
    a control measured at a different dose than the effect cannot rule the
    disturbance explanation out.
    """
    conditions: List[runs.Condition] = []
    shared = dict(layer=layer, per_bin=per_bin)

    for budget in budgets:
        conditions.append(
            runs.Condition("story_B{}".format(budget), arm="story", budget=budget, alpha=0.0, **shared)
        )
        conditions.append(
            runs.Condition(
                "literal_B{}".format(budget), arm="literal", budget=budget, alpha=0.0, **shared
            )
        )
        for arm in ("steer", "random", "negated"):
            for alpha in alphas:
                conditions.append(
                    runs.Condition(
                        "{}_a{:g}_B{}".format(arm, alpha, budget),
                        arm=arm,
                        budget=budget,
                        alpha=float(alpha),
                        **shared
                    )
                )
    return conditions


# ------------------------------------------------------------------ the vector


def fit_direction(model, tokenizer, fit_story, fit_literal, layer: int, batch_size: int):
    """Mean activation on plain-English texts minus mean on story texts.

    Read at the last token of the *bare* rendering rather than of the full
    formalization prompt, so the direction describes the difference between the
    two ways of stating a problem and not the shared instruction boilerplate.
    """
    from shared import hooks

    # The vendored sampler calls the rendered text "story" whatever form it is
    # in, so the plain-English arm's description is under that key too.
    story = hooks.capture_residuals(
        model, tokenizer, [s["story"] for s in fit_story], [layer],
        batch_size=batch_size, progress="fit: story",
    )["last"][layer]
    literal = hooks.capture_residuals(
        model, tokenizer, [s["story"] for s in fit_literal], [layer],
        batch_size=batch_size, progress="fit: literal",
    )["last"][layer]

    vector = hooks.contrast_vector(literal, story)
    residual_norm = float(story.norm(dim=-1).median())
    return vector, {
        "layer": layer,
        "vector_norm": float(vector.norm()),
        "median_residual_norm": residual_norm,
        "relative_size": float(vector.norm()) / residual_norm if residual_norm else None,
        "coherence": _coherence(literal - story),
        "n_fit_pairs": len(fit_story),
    }


def _coherence(differences) -> float:
    """Mean pairwise cosine between the per-problem differences.

    Near zero would mean the direction is the average of unrelated differences
    rather than a consistent one; the earlier run measured 0.63 to 0.91.
    """
    import torch

    unit = torch.nn.functional.normalize(differences.float(), dim=-1)
    n = unit.shape[0]
    if n < 2:
        return float("nan")
    sims = unit @ unit.T
    return float((sims.sum() - sims.diagonal().sum()) / (n * (n - 1)))


def intervention_for(condition: runs.Condition, model, vector):
    """The residual-stream patch this condition runs under, or None.

    Every arm is produced by the same generation path; only this object differs
    between them, which is what makes the comparison a comparison.
    """
    from shared import hooks

    arm = condition["arm"]
    if arm in ("story", "literal"):
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
    from shared import generation, hooks, model as model_module

    todo = runs.pending(run, conditions, force=args.force)
    if not todo:
        print("every condition already has records; nothing to generate")
        return

    print(runs.describe(conditions, run))
    print("\ngenerating {} of {} conditions".format(len(todo), len(conditions)))

    arms = dataset.sample_etp_matched(args.per_bin, args.seed, ("story", "literal"))
    fit_story, eval_story = dataset.split_alternating(arms["story"])
    fit_literal, eval_literal = dataset.split_alternating(arms["literal"])
    print(
        "{} pairs: {} to fit the direction, {} held out for grading".format(
            len(arms["story"]), len(fit_story), len(eval_story)
        )
    )

    model, tokenizer = model_module.load(args.model)
    model_module.check_layers(model, [args.layer])
    generator = generation.Generator(model, tokenizer, batch_size=args.batch_size)

    vector, vector_report = fit_direction(
        model, tokenizer, fit_story, fit_literal, args.layer, args.batch_size
    )
    print(
        "\ndirection at layer {}: norm {:.1f}, {:.0%} of the median residual "
        "norm, per-pair coherence {:.2f}".format(
            vector_report["layer"],
            vector_report["vector_norm"],
            vector_report["relative_size"],
            vector_report["coherence"],
        )
    )
    _save_vector_report(run, vector_report)

    for condition in todo:
        samples = eval_literal if condition["arm"] == "literal" else eval_story
        form = "literal" if condition["arm"] == "literal" else "story"
        prompts = [wrap_prompt(s["prompt"], "on", "", form) for s in samples]

        model_module.set_seed(args.seed)
        completions = generator.generate_budgeted(
            prompts,
            condition["budget"],
            max_answer_tokens=args.answer_tokens,
            intervention=intervention_for(condition, model, vector),
            progress=condition.name,
        )
        with run.writing(condition.name) as writer:
            for sample, completion in zip(samples, completions):
                writer.write(
                    grading.grade_record(
                        completion.answer,
                        sample,
                        extra=dict(
                            condition.settings,
                            condition=condition.name,
                            form=form,
                            **completion.as_dict()
                        ),
                    )
                )
        rows = run.read(condition.name)
        print(
            "  {:<22} accuracy {}  unparseable {}".format(
                condition.name, grading.correct_rate(rows), grading.unparseable_rate(rows)
            )
        )


def _save_vector_report(run: runs.RunDirectory, report: dict) -> None:
    import json

    (run.path / "direction.json").write_text(json.dumps(report, indent=2) + "\n")


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
                "think_tokens_mean": _mean(r["think_tokens"] for r in rows),
                "cut_off": stats.rate_of(not r["closed_naturally"] for r in rows).as_dict(),
            }

        baseline = "story_B{}".format(budget)
        ceiling = "literal_B{}".format(budget)
        if baseline in records:
            block["headroom"] = _headroom(records, baseline, ceiling)
            block["vs_untouched"] = {
                condition.name: dict(
                    stats.pair_records(records[baseline], records[condition.name]).as_dict(),
                    share_of_headroom=_share_of_headroom(
                        records, baseline, ceiling, condition.name
                    ),
                )
                for condition in at_budget
                if condition.name != baseline and condition["arm"] != "literal"
            }
        summary["budgets"][str(budget)] = block

    print_summary(summary)
    return summary


def _headroom(records: Dict[str, List[dict]], baseline: str, ceiling: str) -> dict:
    """How much a perfect form conversion could buy at this budget.

    The plain-English arm is the honest ceiling for an intervention whose whole
    claim is to make a story look like plain English. Without it, a null has no
    scale to be null against.
    """
    floor_rate = grading.correct_rate(records[baseline])
    if ceiling not in records:
        return {"story": floor_rate.as_dict(), "literal": None, "points_available": None}
    ceiling_rate = grading.correct_rate(records[ceiling])
    return {
        "story": floor_rate.as_dict(),
        "literal": ceiling_rate.as_dict(),
        "points_available": ceiling_rate.value - floor_rate.value,
    }


def _share_of_headroom(
    records: Dict[str, List[dict]], baseline: str, ceiling: str, name: str
) -> Optional[float]:
    """Gain as a fraction of the room the plain-English arm shows is there.

    Raw point differences are not comparable across budgets, because an arm
    already at 79% has only 21 points left to win. This divides by the room
    actually available.
    """
    if ceiling not in records:
        return None
    floor_value = grading.correct_rate(records[baseline]).value
    room = grading.correct_rate(records[ceiling]).value - floor_value
    if room <= 0:
        return None
    return (grading.correct_rate(records[name]).value - floor_value) / room


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------- output


def print_summary(summary: dict) -> None:
    for budget, block in sorted(summary["budgets"].items(), key=lambda kv: int(kv[0])):
        print("\nthinking budget {} tokens".format(budget))
        print(
            "  {:<22} {:>20} {:>14} {:>13} {:>8}".format(
                "condition", "accuracy", "95% CI", "unparseable", "cut off"
            )
        )
        for name, cell in sorted(block["cells"].items()):
            accuracy = cell["accuracy"]
            print(
                "  {:<22} {:>20} {:>14} {:>13.0%} {:>8.0%}".format(
                    name,
                    "{:.1%} ({}/{})".format(
                        accuracy["rate"], accuracy["successes"], accuracy["total"]
                    ),
                    "{:.0%}-{:.0%}".format(*accuracy["ci95"]),
                    cell["unparseable"]["rate"],
                    cell["cut_off"]["rate"],
                )
            )

        room = block.get("headroom", {}).get("points_available")
        if room is not None:
            print(
                "  a perfect story-to-plain-English conversion is worth "
                "{:+.1f} points here".format(100 * room)
            )
        for name, comparison in sorted(block.get("vs_untouched", {}).items()):
            share = comparison["share_of_headroom"]
            print(
                "  {:<22} vs untouched: {:+.1f} points ({} gained, {} lost, "
                "McNemar p = {:.3g}){}".format(
                    name,
                    100 * comparison["delta"],
                    comparison["gained"],
                    comparison["lost"],
                    comparison["mcnemar_p"],
                    "" if share is None else ", {:.0%} of the headroom".format(share),
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
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.8 * len(budgets), 4.6), squeeze=False)
    styles = {
        "steer": ("o", "-", "#1f77b4"),
        "random": ("x", ":", "#d62728"),
        "negated": ("s", "--", "#9467bd"),
    }

    for column, budget in enumerate(budgets):
        axis = axes[0][column]
        cells = summary["budgets"][budget]["cells"]

        for arm, (marker, line, colour) in styles.items():
            series = sorted(
                ((c["alpha"], c) for c in cells.values() if c["arm"] == arm),
                key=lambda pair: pair[0],
            )
            if not series:
                continue
            rates = [c["accuracy"]["rate"] for _, c in series]
            low = [r - c["accuracy"]["ci95"][0] for r, (_, c) in zip(rates, series)]
            high = [c["accuracy"]["ci95"][1] - r for r, (_, c) in zip(rates, series)]
            axis.errorbar(
                [a for a, _ in series], rates, yerr=[low, high], marker=marker,
                linestyle=line, capsize=3, color=colour, label=arm,
            )

        # The two untouched arms bracket what any form intervention could do:
        # the story arm is where it starts, the plain-English arm is what a
        # perfect conversion would be worth.
        for arm, colour, label in (
            ("story", "#888888", "untouched, story input"),
            ("literal", "#2a9d3f", "untouched, plain-English input"),
        ):
            reference = [c for c in cells.values() if c["arm"] == arm]
            if reference:
                axis.axhline(
                    reference[0]["accuracy"]["rate"], linestyle="-", linewidth=1.4,
                    color=colour, alpha=0.8, label=label,
                )

        axis.set_xlabel("dose, as a multiple of the difference vector")
        axis.set_ylim(0, 1)
        axis.set_ylabel("share of answers correct")
        axis.set_title("thinking budget {} tokens".format(budget))
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.suptitle("Steering the story-to-plain-English direction, by thinking budget")
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
        help="doses, as multiples of the raw difference vector (default: %(default)s)",
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
        args.alphas = (1.0,)
        # Enough room for an answer to finish; below roughly 256 tokens every
        # row grades as unparseable and the smoke run stops testing grading.
        args.answer_tokens = min(args.answer_tokens, 256)

    conditions = build_conditions(args.budgets, args.layer, args.alphas, args.per_bin)
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
