#!/usr/bin/env python3
"""Does accuracy fall as a problem's serial depth rises, with length held fixed?

QUESTION
--------
Every complexity result in this project so far is confounded with text length:
word count correlates with the number of operations in a law at 0.93 to 1.00
within every surface form, so "the model struggles with complex problems" has
never been separable from "the model struggles with long inputs". The dataset
module fixes that by varying *depth* — the height of the term tree, i.e. how
many steps must be evaluated one after another — while pinning everything a
renderer's length depends on. Within a family, depth 3 and depth 6 problems
have the same number of operations, the same number of variables, the same
number of definition steps, and identical word counts, to a standard deviation
of zero.

So: at a fixed thinking-token budget, does accuracy fall as depth rises?

This is a real prediction rather than a fishing expedition. The earlier budget
sweep found accuracy is a smooth function of how many reasoning tokens the
model is allowed, which says the bottleneck is serial computation. Depth is
serial chain length. If that reading is right, depth should cost accuracy even
though nothing about the text got longer.

WHAT EACH OUTCOME WOULD MEAN
----------------------------
- Accuracy falls monotonically with depth, in both families: the model has a
  genuine difficulty axis that length cannot explain. Every later geometric
  claim about "complexity" gets a variable it can stand on, and the depth axis
  becomes worth looking for in the activations.
- Accuracy is flat across depth: the earlier complexity gradients were length
  gradients, and the honest report is that this dataset's difficulty is
  dominated by how much text there is to read, not by how much serial work the
  problem requires. A geometric depth axis found later would then need a
  reading other than "the model represents difficulty".
- Accuracy falls but only in one family: the effect is tied to that family's
  operation count or variable count rather than to depth, so it is not the
  variable it appears to be. This is why there are two families.
- Thinking tokens rise with depth while accuracy stays flat: the model notices
  the extra serial work and pays for it, and the budget was generous enough to
  absorb the cost. Informative, and an argument for rerunning at a budget tight
  enough to bind.

CONTROLS AND WHAT LIMITS THIS
-----------------------------
- Two families, run identically, so every result carries a replication. A
  finding that appears in one and not the other is not a depth finding.
- Length is not merely balanced but constant, verified before any run by
  tests/check_length_balance.py, which fails if any surface form's word count
  varies across the depths of a family.
- The comparison across depth is *unpaired* and cannot be otherwise: a law
  cannot be both depth 4 and depth 7, so no problem is shared between depth
  levels and McNemar's test does not apply. Differences across depth are
  therefore reported with Newcombe intervals, which are wide, and this is the
  reason the default sample is 200 pairs per cell rather than the ~50 used
  where pairing was available. Comparisons across *budget* at one depth do use
  the same problems, and those are reported paired.
- Word count is held constant, token count is only expected to be. The run
  records the tokenized length of every prompt and the analysis fails loudly if
  it varies within a family, since that would reintroduce the confound in the
  units the model actually sees.
- The law pool is large enough that each cell draws on far more distinct laws
  than it needs, so the problems within a cell are close to independent. The
  shallowest cell of the six-operation family is the binding one; see
  DEFAULT_POOL_PER_BIN.
- The literal description is the default surface form because it scores highest,
  which leaves the most room for accuracy to fall. On the story form a depth
  effect could be hidden by a floor.

    python3 depth-at-fixed-length.py --quick        # smoke test the plumbing
    python3 depth-at-fixed-length.py                # the real run
    python3 depth-at-fixed-length.py --analyze-only # redo tables and figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import dataset, grading, runs, stats
from shared.vendor import ensure_on_path

ensure_on_path()

from benchmark import wrap_prompt  # noqa: E402

EXPERIMENT = "depth-at-fixed-length"

# One budget by default: depth is the variable, and every extra budget
# multiplies the run. 512 thinking tokens is generous enough that the earlier
# sweep had accuracy near its ceiling, so a fall here is attributable to the
# problem rather than to the cap.
DEFAULT_BUDGETS = (512,)
DEFAULT_PER_CELL = 200
DEFAULT_ANSWER_TOKENS = 512

# How many laws to synthesize per operation count. The shared default of 4000
# is far too small here: the shallowest cell of the six-operation family is
# rare, and at that size it holds 13 laws, so 200 problems would reuse each of
# them about 15 times and would not be 200 independent problems. At 60000 that
# cell holds 254 laws and each is used about 1.6 times. Generation is a pure
# function of the seed and takes about ten seconds.
DEFAULT_POOL_PER_BIN = 60000


def build_conditions(
    cells: Sequence[dataset.Shape],
    budgets: Sequence[int],
    per_cell: int,
    form: str,
) -> List[runs.Condition]:
    """The sweep, as data rather than as nested loops.

    One condition per (cell, budget), so each has a name, its settings land in
    its records, and an interrupted run resumes at the cell it died in rather
    than at the beginning. The cells come from the grid the length-balance
    check verifies, so no depth is requested that the law pool cannot fill at
    constant length.
    """
    conditions = []
    for minor, major, variables, depth in cells:
        for budget in budgets:
            conditions.append(
                runs.Condition(
                    "f{}-{}-{}_d{}_B{}".format(minor, major, variables, depth, budget),
                    minor=minor,
                    major=major,
                    variables=variables,
                    depth=depth,
                    budget=budget,
                    per_cell=per_cell,
                    form=form,
                )
            )
    return conditions


def cells_of(condition: runs.Condition) -> Tuple[dataset.Shape, ...]:
    return (
        (
            condition["minor"],
            condition["major"],
            condition["variables"],
            condition["depth"],
        ),
    )


def samples_for(pool: Sequence[str], condition: runs.Condition, seed: int) -> List[dict]:
    """The pairs one condition runs on.

    Drawn from that cell alone with a fixed seed, so the same cell yields the
    same pairs no matter which conditions ran before it. That is what lets two
    budgets at one depth be compared on identical problems.
    """
    return dataset.sample_depth_balanced(
        pool,
        condition["per_cell"],
        seed,
        cells=cells_of(condition),
        form=condition["form"],
    )


# ------------------------------------------------------------------ generation


def run_experiment(args, run: runs.RunDirectory, conditions: List[runs.Condition]) -> None:
    from shared import generation, model as model_module

    todo = runs.pending(run, conditions, force=args.force)
    if not todo:
        print("every condition already has records; nothing to generate")
        return

    print(runs.describe(conditions, run))
    print("\ngenerating {} of {} conditions".format(len(todo), len(conditions)))

    # Build every condition's problems before the model is loaded. A cell whose
    # law pool is too small to fill raises here, in seconds, rather than after
    # a model load and however many hours of generation came before it.
    print("building datasets for all {} conditions".format(len(todo)))
    pool = dataset.build_pool(per_bin=args.pool_per_bin)
    problems = {c.name: samples_for(pool, c, args.seed) for c in todo}

    model, tokenizer = model_module.load(args.model)
    generator = generation.Generator(model, tokenizer, batch_size=args.batch_size)

    for condition in todo:
        samples = problems[condition.name]
        prompts = [
            # The same "thinking on" wrapper at every budget, so the budget is
            # the only thing that differs between them.
            wrap_prompt(sample["prompt"], "on", "", condition["form"])
            for sample in samples
        ]
        prompt_tokens = [
            len(tokenizer(prompt)["input_ids"]) for prompt in prompts
        ]
        model_module.set_seed(args.seed)
        completions = generator.generate_budgeted(
            prompts,
            condition["budget"],
            max_answer_tokens=args.answer_tokens,
            progress=condition.name,
        )
        with run.writing(condition.name) as writer:
            for sample, completion, n_prompt in zip(samples, completions, prompt_tokens):
                record = grading.grade_record(
                    completion.answer,
                    sample,
                    extra=dict(
                        condition.settings,
                        condition=condition.name,
                        prompt_tokens=n_prompt,
                        **completion.as_dict()
                    ),
                )
                writer.write(record)
        rows = run.read(condition.name)
        print(
            "  {:<24} accuracy {}  thinking {:.0f} tokens  cut off {}".format(
                condition.name,
                grading.correct_rate(rows),
                _mean(r["think_tokens"] for r in rows),
                stats.rate_of(not r["closed_naturally"] for r in rows),
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
    summary = {"families": {}, "token_length_check": check_token_lengths(records, finished)}

    for family in sorted({(c["minor"], c["major"], c["variables"]) for c in finished}):
        family_key = "{}-{}-{}".format(*family)
        in_family = [
            c for c in finished if (c["minor"], c["major"], c["variables"]) == family
        ]
        summary["families"][family_key] = {
            "operations_per_equation": family[1] + family[0],
            "variables": family[2],
            "budgets": {},
        }
        for budget in sorted({c["budget"] for c in in_family}):
            cells = sorted(
                (c for c in in_family if c["budget"] == budget),
                key=lambda c: c["depth"],
            )
            summary["families"][family_key]["budgets"][str(budget)] = depth_curve(
                cells, records
            )
        paired = budget_comparisons(in_family, records)
        if paired:
            summary["families"][family_key]["budget_comparisons"] = paired

    print_summary(summary)
    return summary


def depth_curve(cells: List[runs.Condition], records: Dict[str, List[dict]]) -> dict:
    """Accuracy and reasoning length at each depth, plus the trend across them."""
    per_depth = {}
    flat_depths: List[float] = []
    flat_correct: List[float] = []
    for cell in cells:
        rows = records[cell.name]
        per_depth[str(cell["depth"])] = {
            "accuracy": grading.correct_rate(rows).as_dict(),
            "unparseable": grading.unparseable_rate(rows).as_dict(),
            "buckets": grading.bucket_counts(rows),
            "think_tokens_mean": _mean(r["think_tokens"] for r in rows),
            "cut_off": stats.rate_of(
                not r["closed_naturally"] for r in rows
            ).as_dict(),
            "prompt_tokens": sorted({r["prompt_tokens"] for r in rows}),
        }
        for row in rows:
            flat_depths.append(cell["depth"])
            flat_correct.append(1.0 if row["status"] == "correct" else 0.0)

    out: dict = {"by_depth": per_depth}
    if len(cells) >= 2:
        shallow, deep = records[cells[0].name], records[cells[-1].name]
        difference = stats.unpaired_difference(
            grading.correct_rate(shallow), grading.correct_rate(deep)
        )
        out["deepest_minus_shallowest"] = dict(
            difference.as_dict(),
            shallowest_depth=cells[0]["depth"],
            deepest_depth=cells[-1]["depth"],
        )
        out["trend"] = {
            "spearman_depth_vs_correct": stats.spearman(flat_depths, flat_correct),
            "spearman_depth_vs_think_tokens": _think_trend(cells, records),
        }
    return out


def _think_trend(
    cells: List[runs.Condition], records: Dict[str, List[dict]]
) -> Optional[float]:
    depths, tokens = [], []
    for cell in cells:
        for row in records[cell.name]:
            depths.append(cell["depth"])
            tokens.append(row["think_tokens"])
    return stats.spearman(depths, tokens)


def budget_comparisons(
    cells: List[runs.Condition], records: Dict[str, List[dict]]
) -> dict:
    """Budget against budget at one depth, on identical problems.

    Unlike the depth axis these conditions share their pairs, so the comparison
    is paired and McNemar's test applies.
    """
    out = {}
    budgets = sorted({c["budget"] for c in cells})
    if len(budgets) < 2:
        return out
    baseline = budgets[0]
    for depth in sorted({c["depth"] for c in cells}):
        at_depth = {c["budget"]: c.name for c in cells if c["depth"] == depth}
        if baseline not in at_depth:
            continue
        for budget in budgets[1:]:
            if budget not in at_depth:
                continue
            result = stats.pair_records(
                records[at_depth[baseline]], records[at_depth[budget]]
            )
            out["depth{}_B{}_vs_B{}".format(depth, budget, baseline)] = result.as_dict()
    return out


def check_token_lengths(
    records: Dict[str, List[dict]], conditions: List[runs.Condition]
) -> dict:
    """Is the prompt the same length in tokens at every depth of a family?

    Word count is constant by construction, and the tokenizer should follow
    since the texts share their step count, their value labels and their
    variables. Should is not is, and if this fails the length confound is back
    in the units the model actually reads.
    """
    by_family: Dict[str, set] = {}
    for condition in conditions:
        family = "{}-{}-{}".format(
            condition["minor"], condition["major"], condition["variables"]
        )
        for row in records[condition.name]:
            by_family.setdefault(family, set()).add(row["prompt_tokens"])

    out = {}
    for family, lengths in sorted(by_family.items()):
        out[family] = {
            "prompt_token_lengths": sorted(lengths),
            "constant": len(lengths) == 1,
        }
    return out


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


# --------------------------------------------------------------------- output


def print_summary(summary: dict) -> None:
    for family, block in sorted(summary["families"].items()):
        print(
            "\nfamily {}: {} operations per equation, {} variables".format(
                family, block["operations_per_equation"], block["variables"]
            )
        )
        for budget, curve in sorted(block["budgets"].items()):
            print("  thinking budget {} tokens".format(budget))
            print(
                "    {:>5} {:>20} {:>14} {:>16} {:>8}".format(
                    "depth", "accuracy", "95% CI", "thinking tokens", "cut off"
                )
            )
            for depth, cell in sorted(curve["by_depth"].items(), key=lambda kv: int(kv[0])):
                accuracy = cell["accuracy"]
                print(
                    "    {:>5} {:>20} {:>14} {:>16.0f} {:>8.0%}".format(
                        depth,
                        "{:.1%} ({}/{})".format(
                            accuracy["rate"], accuracy["successes"], accuracy["total"]
                        ),
                        "{:.0%}-{:.0%}".format(*accuracy["ci95"]),
                        cell["think_tokens_mean"],
                        cell["cut_off"]["rate"],
                    )
                )
            if "deepest_minus_shallowest" in curve:
                d = curve["deepest_minus_shallowest"]
                print(
                    "    depth {} minus depth {}: {:+.1f} points, 95% CI {:+.1f} "
                    "to {:+.1f}{}".format(
                        d["deepest_depth"],
                        d["shallowest_depth"],
                        100 * d["delta"],
                        100 * d["ci95"][0],
                        100 * d["ci95"][1],
                        "" if d["excludes_zero"] else "  (interval includes zero)",
                    )
                )
                trend = curve["trend"]["spearman_depth_vs_correct"]
                tokens = curve["trend"]["spearman_depth_vs_think_tokens"]
                print(
                    "    rank correlation with depth: accuracy {}, thinking "
                    "tokens {}".format(_fmt(trend), _fmt(tokens))
                )

    print("\nprompt length in tokens, which must not vary within a family")
    for family, check in sorted(summary["token_length_check"].items()):
        lengths = check["prompt_token_lengths"]
        print(
            "  family {}: {}{}".format(
                family,
                lengths if len(lengths) <= 4 else "{} distinct values".format(len(lengths)),
                "" if check["constant"] else "  <-- NOT CONSTANT, the confound is back",
            )
        )


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:+.3f}".format(value)


def make_figure(summary: dict, run: runs.RunDirectory) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the figure")
        return None

    families = sorted(summary["families"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for family in families:
        for budget, curve in sorted(summary["families"][family]["budgets"].items()):
            depths = sorted(int(d) for d in curve["by_depth"])
            cells = [curve["by_depth"][str(d)] for d in depths]
            label = "{} ops, budget {}".format(
                summary["families"][family]["operations_per_equation"], budget
            )
            rates = [c["accuracy"]["rate"] for c in cells]
            low = [r - c["accuracy"]["ci95"][0] for r, c in zip(rates, cells)]
            high = [c["accuracy"]["ci95"][1] - r for r, c in zip(rates, cells)]
            axes[0].errorbar(
                depths, rates, yerr=[low, high], marker="o", capsize=3, label=label
            )
            axes[1].plot(
                depths,
                [c["think_tokens_mean"] for c in cells],
                marker="o",
                label=label,
            )

    axes[0].set_xlabel("depth of the term tree (serial chain length)")
    axes[0].set_ylabel("share of answers correct")
    axes[0].set_title("Accuracy vs depth, with text length held constant")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("depth of the term tree (serial chain length)")
    axes[1].set_ylabel("mean thinking tokens used")
    axes[1].set_title("Reasoning spent vs depth")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

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
        help="thinking-token budgets to run, comma separated (default: %(default)s)",
    )
    cli.add_argument(
        "--per-cell",
        type=int,
        default=DEFAULT_PER_CELL,
        help="pairs per (family, depth, budget) cell (default: %(default)s)",
    )
    cli.add_argument(
        "--form",
        default="literal",
        choices=("literal", "story"),
        help="surface form the problem is presented in (default: %(default)s)",
    )
    cli.add_argument(
        "--cells",
        type=dataset.parse_cells,
        default=dataset.parse_cells(dataset.DEFAULT_CELLS),
        metavar="MINOR:MAJOR:VARS:DEPTH,...",
        help="the depth grid to run (default: {}, the two families the "
        "length-balance check verifies)".format(dataset.DEFAULT_CELLS),
    )
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument(
        "--pool-per-bin",
        type=int,
        default=DEFAULT_POOL_PER_BIN,
        help="laws to synthesize per operation count; sets how many distinct "
        "laws each cell can draw on (default: %(default)s)",
    )
    cli.add_argument(
        "--answer-tokens",
        type=int,
        default=DEFAULT_ANSWER_TOKENS,
        help="cap on the answer, after the thinking budget (default: %(default)s)",
    )
    args = cli.parse_args(argv)

    if args.quick:
        args.per_cell = min(args.per_cell, 4)
        args.budgets = (0, 64)
        # A small pool is enough for four problems a cell and keeps the smoke
        # test quick. The answer cap is deliberately left at its real value:
        # capping it lower truncates answers mid-expression and every row grades
        # as unreadable, which hides whether grading works at all.
        args.pool_per_bin = min(args.pool_per_bin, 4000)

    conditions = build_conditions(args.cells, args.budgets, args.per_cell, args.form)
    out_dir = runs.resolve_out_dir(args, __file__)
    run = runs.RunDirectory.open(
        out_dir,
        EXPERIMENT,
        {
            "model": args.model,
            "seed": args.seed,
            "budgets": list(args.budgets),
            "per_cell": args.per_cell,
            "form": args.form,
            "answer_tokens": args.answer_tokens,
            "pool_per_bin": args.pool_per_bin,
            "cells": [list(cell) for cell in args.cells],
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
