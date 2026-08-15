#!/usr/bin/env python3
"""Does the model still hold which law it was given after it finishes reasoning?

QUESTION
--------
An earlier experiment read the model's internal activations at the end of the
problem statement, with no reasoning allowed, and asked: taking one problem's
activations, is the closest other problem in the set the one describing the same
underlying law, even when the two are written in completely different surface
forms? It was, 86% of the time, against 1.9% for guessing. So the model builds a
representation of the law that survives being dressed up as four different
stories or as a plain-English description.

That reading was taken before any reasoning happened, which is the one place the
answer was never in doubt: the model has just read the problem, so of course the
problem is still there. The interesting question is what survives 512 tokens of
reasoning. The behavioural results all point at reasoning being where the work
happens — accuracy climbs from 38.5% with no reasoning to 96.2% at 512 tokens on
plain-English input — so if the law representation is load-bearing, it should be
readable at the end of the trace too.

The measurement is the same one, taken at two places: the end of the problem
statement, and the last token after the reasoning trace has closed.

WHAT EACH OUTCOME WOULD MEAN
----------------------------
- The law is still identifiable after reasoning, at a similar rate: the
  representation persists through the whole computation, which makes it a real
  candidate for the thing the model works with rather than an artifact of having
  just read the text. Interventions on it become worth trying at that position.
- Identification collapses after reasoning: whatever the model carries forward is
  not organized by law identity. That would be a genuine constraint — it would
  say the pre-reasoning geometry is a reading-comprehension representation, and
  that later steering work should target the prompt, not the trace.
- Identification is higher after reasoning: the trace sharpens the
  representation rather than consuming it, and reasoning is partly a process of
  making the law explicit. Watch for the confound below before believing it.
- Identification after reasoning is much better on problems the model got right
  than on ones it got wrong: the representation is tied to solving the problem
  rather than merely to having read it, which is the strongest version of the
  result available from a correlational measurement.

CONTROLS AND WHAT LIMITS THIS
-----------------------------
- The nearest neighbour is always searched among texts in a *different* surface
  form, so a match cannot be won on shared wording. Chance is one over the
  number of laws and is printed beside every number.
- The same laws are measured at both positions, so the before-and-after
  comparison is on identical problems.
- The obvious confound on any post-reasoning gain: a reasoning trace usually
  restates the problem in the model's own words, and traces about the same law
  will converge on similar wording whatever the original surface form was. That
  would raise identification for a boring reason. The run records every trace's
  length and whether the budget cut it off, and the honest reading of a large
  gain is that this experiment cannot separate the two accounts; the follow-up
  that could is reading at a fixed early position inside the trace.
- Reading after reasoning is only defined where reasoning happened, so this runs
  at a 512-token budget and not at zero.
- The correct-versus-wrong split is correlational. Problems the model answers
  correctly are easier problems, and easier problems may be more identifiable for
  reasons unrelated to reasoning.

    python3 law-representation-after-reasoning.py --quick        # smoke test
    python3 law-representation-after-reasoning.py                # the real run
    python3 law-representation-after-reasoning.py --analyze-only # redo figures
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import dataset, grading, runs, stats
from shared.vendor import ensure_on_path

ensure_on_path()

from benchmark import FORMS, wrap_prompt  # noqa: E402
from checkform import build_prompt  # noqa: E402

EXPERIMENT = "law-representation-after-reasoning"

# Four themed stories and the plain-English description: five ways of writing
# the same law, which is what makes a cross-form match meaningful. Rigid
# Grammar, the terse answer notation, is deliberately excluded — it *is* the
# answer, so asking the model to formalize it is not the same task.
DEFAULT_SURFACES = dataset.STORY_THEMES + ("literal",)

# Where in the network to read. Every fourth layer of a 36-layer model, which
# is enough to see the shape of the curve without nine times the memory.
DEFAULT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32, 36)

# The positions read, and what each one means in plain terms.
#
# Three of these are read after the reasoning, not one, because the obvious
# choice is the worst of them. The last token of the whole sequence is the
# newline that follows the closing `</think>` tag, and that token is identical
# in every sequence in the run; a low number there could mean the law was lost
# or could just mean structural tokens carry little content. Reading inside the
# trace avoids that, and the average over the trace is the fairest counterpart
# to the pre-reasoning reading, which is also taken over content.
POSITIONS = {
    "problem_end": "the last token of the problem statement, before any reasoning",
    "prompt_end": "the last token of the prompt, including the task instructions",
    "reasoning_end": "the last token of the reasoning trace itself",
    "reasoning_mean": "averaged over every token of the reasoning trace",
    "after_reasoning": "the last token before the answer starts, just after the trace closes",
}

# The positions that only exist once the model has reasoned. Identification at
# these is what the experiment is about; the first two are the reference.
POST_REASONING = ("reasoning_end", "reasoning_mean", "after_reasoning")

DEFAULT_BUDGET = 512
# 10 per operation-count bin gives 53 laws once vacuous ones are dropped, close
# to the 52 the earlier reading used, so the two are directly comparable.
DEFAULT_PER_BIN = 10
DEFAULT_ANSWER_TOKENS = 512


def build_conditions(
    surfaces: Sequence[str], budget: int, per_bin: int
) -> List[runs.Condition]:
    """One condition per surface form.

    Each writes both its graded answers and its activations, so an interrupted
    run resumes at the surface it died in. Splitting by surface rather than by
    position is what lets a single expensive generation pass serve both the
    before and after readings.
    """
    return [
        runs.Condition(surface, surface=surface, budget=budget, per_bin=per_bin)
        for surface in surfaces
    ]


def activations_path(run: runs.RunDirectory, name: str) -> Path:
    return run.path / "activations-{}.pt".format(name)


def texts_for(samples: Sequence[dict], surface: str) -> List[Tuple[str, str]]:
    """(bare rendering, formalization prompt) for each pair in one surface form.

    The prompt template depends only on whether the text is a story or a plain
    description, so all four themes share the story template.
    """
    kind = "literal" if surface == "literal" else "story"
    template = FORMS[kind][1]
    out = []
    for sample in samples:
        text = dataset.render_all_forms(sample["metadata"])[surface]
        prompt = build_prompt(
            {"story": text, "metadata": sample["metadata"]}, template_path=template
        )
        out.append((text, wrap_prompt(prompt, "on", "", kind)))
    return out


# ------------------------------------------------------- generation and reading


def run_experiment(args, run: runs.RunDirectory, conditions: List[runs.Condition]) -> None:
    import torch

    from shared import generation, hooks, model as model_module

    todo = pending(run, conditions, force=args.force)
    if not todo:
        print("every surface already has records and activations")
        return

    print(runs.describe(conditions, run))
    print("\nrunning {} of {} surfaces".format(len(todo), len(conditions)))

    samples = dataset.sample_etp_stratified(args.per_bin, args.seed, form="story")
    print("{} laws, so guessing would identify {:.1%}".format(
        len(samples), 1.0 / len(samples) if samples else float("nan")
    ))

    model, tokenizer = model_module.load(args.model)
    model_module.check_layers(model, args.layers)
    generator = generation.Generator(model, tokenizer, batch_size=args.batch_size)

    for condition in todo:
        surface = condition["surface"]
        pairs = texts_for(samples, surface)
        texts = [text for text, _ in pairs]
        prompts = [prompt for _, prompt in pairs]
        chats = [generator.build_chat(prompt, thinking=True) for prompt in prompts]

        before = hooks.capture_residuals(
            model, tokenizer, chats, args.layers,
            batch_size=args.act_batch_size,
            spans=texts,
            progress="{}: before reasoning".format(surface),
        )

        model_module.set_seed(args.seed)
        completions = generator.generate_budgeted(
            prompts,
            condition["budget"],
            max_answer_tokens=args.answer_tokens,
            progress="{}: reasoning".format(surface),
        )

        # The text the model has in front of it the instant it starts writing
        # its answer: the prompt, its own reasoning, and the closing tag.
        after_texts = [
            chat + completion.thinking + generation.THINK_END + "\n\n"
            for chat, completion in zip(chats, completions)
        ]
        after = hooks.capture_residuals(
            model, tokenizer, after_texts, args.layers,
            batch_size=args.act_batch_size,
            spans=[completion.thinking for completion in completions],
            progress="{}: after reasoning".format(surface),
        )

        torch.save(
            {
                "problem_end": before["span_last"],
                "prompt_end": before["last"],
                "reasoning_end": after["span_last"],
                "reasoning_mean": after["span_mean"],
                "after_reasoning": after["last"],
                "pair_ids": [s["pair_id"] for s in samples],
                "surface": surface,
                "layers": list(args.layers),
            },
            activations_path(run, condition.name),
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
                            **completion.as_dict()
                        ),
                    )
                )
        rows = run.read(condition.name)
        print(
            "  {:<10} accuracy {}  thinking {:.0f} tokens  cut off {}".format(
                surface,
                grading.correct_rate(rows),
                _mean(r["think_tokens"] for r in rows),
                stats.rate_of(not r["closed_naturally"] for r in rows),
            )
        )


def pending(
    run: runs.RunDirectory, conditions: List[runs.Condition], force: bool
) -> List[runs.Condition]:
    """Conditions still to run, counting a missing activation file as unfinished.

    The shared helper only knows about records; here a condition is not done
    until both halves of its output exist, and a half-finished one is dropped
    so it is redone rather than silently analysed with stale activations.
    """
    todo = []
    for condition in conditions:
        complete = run.has(condition.name) and activations_path(run, condition.name).exists()
        if force or not complete:
            run.drop(condition.name)
            path = activations_path(run, condition.name)
            if path.exists():
                path.unlink()
            todo.append(condition)
    return todo


# -------------------------------------------------------------------- analysis


def analyze(run: runs.RunDirectory, conditions: List[runs.Condition]) -> dict:
    import torch

    finished = [
        c for c in conditions
        if run.has(c.name) and activations_path(run, c.name).exists()
    ]
    if len(finished) < 2:
        raise SystemExit(
            "need at least two surface forms with results in {}, since a match "
            "is only meaningful across forms".format(run.path)
        )
    if len(finished) < len(conditions):
        print(
            "WARNING: analysing {} of {} surfaces; the rest have not "
            "run".format(len(finished), len(conditions))
        )

    loaded = {c.name: torch.load(activations_path(run, c.name)) for c in finished}
    records = {c.name: {r["pair_id"]: r for r in run.read(c.name)} for c in finished}
    layers = loaded[finished[0].name]["layers"]
    pair_ids = loaded[finished[0].name]["pair_ids"]

    surfaces = [c.name for c in finished]
    labels: List[str] = []
    origin: List[str] = []
    for name in surfaces:
        labels.extend(loaded[name]["pair_ids"])
        origin.extend([name] * len(loaded[name]["pair_ids"]))
    was_correct = [
        records[name][pair]["status"] == "correct"
        for name in surfaces
        for pair in loaded[name]["pair_ids"]
    ]

    summary: dict = {
        "n_laws": len(pair_ids),
        "n_surfaces": len(surfaces),
        "surfaces": surfaces,
        "chance": 1.0 / len(pair_ids),
        "position_meanings": POSITIONS,
        "accuracy_by_surface": {
            name: grading.correct_rate(list(records[name].values())).as_dict()
            for name in surfaces
        },
        "identification": {},
    }

    for position in POSITIONS:
        by_layer = {}
        for index, layer in enumerate(layers):
            stacked = torch.cat([loaded[name][position][layer] for name in surfaces])
            hits = nearest_neighbour_hits(stacked, labels, origin)
            entry = {"overall": stats.rate_of(hits).as_dict()}
            if position in POST_REASONING:
                entry["when_answer_correct"] = stats.rate_of(
                    hit for hit, ok in zip(hits, was_correct) if ok
                ).as_dict()
                entry["when_answer_wrong"] = stats.rate_of(
                    hit for hit, ok in zip(hits, was_correct) if not ok
                ).as_dict()
            by_layer[str(layer)] = entry
        summary["identification"][position] = by_layer

    summary["best_layer"] = {
        position: max(
            by_layer, key=lambda L: by_layer[L]["overall"]["rate"]
        )
        for position, by_layer in summary["identification"].items()
    }
    summary["reasoning_cost"] = reasoning_cost(summary)
    summary["thinking"] = {
        name: {
            "think_tokens_mean": _mean(
                r["think_tokens"] for r in records[name].values()
            ),
            "cut_off": stats.rate_of(
                not r["closed_naturally"] for r in records[name].values()
            ).as_dict(),
        }
        for name in surfaces
    }

    print_summary(summary)
    return summary


def reasoning_cost(summary: dict) -> dict:
    """What the reasoning trace did to identification, at a matched layer.

    Every post-reasoning position is compared at the layer that reads best
    before reasoning, so no comparison is won by picking a different layer for
    each side of it.
    """
    layer = summary["best_layer"]["problem_end"]
    before = summary["identification"]["problem_end"][layer]["overall"]
    out = {"layer": int(layer), "before": before, "after": {}}
    for position in POST_REASONING:
        cell = summary["identification"].get(position, {}).get(layer)
        if cell is None:
            continue
        out["after"][position] = dict(
            cell["overall"], delta=cell["overall"]["rate"] - before["rate"]
        )
    return out


def nearest_neighbour_hits(
    vectors, labels: Sequence[str], origin: Sequence[str]
) -> List[bool]:
    """For each text, does its closest text in another surface form share its law?

    Cosine similarity, with every candidate from the query's own surface form
    ruled out — itself included — so nothing can be matched on shared wording.
    """
    import torch

    unit = torch.nn.functional.normalize(vectors.float(), dim=-1)
    similarity = unit @ unit.T
    same_surface = torch.tensor(
        [[a == b for b in origin] for a in origin], dtype=torch.bool
    )
    similarity = similarity.masked_fill(same_surface, float("-inf"))
    nearest = similarity.argmax(dim=1).tolist()
    return [labels[query] == labels[match] for query, match in enumerate(nearest)]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------------- output


def print_summary(summary: dict) -> None:
    print(
        "\n{} laws in {} surface forms; guessing would identify {:.1%}".format(
            summary["n_laws"], summary["n_surfaces"], summary["chance"]
        )
    )
    print("\nhow often the closest text in another surface form is the same law")
    layers = sorted(summary["identification"]["problem_end"], key=int)
    print("  {:<18} {}".format("position", " ".join("{:>7}".format(L) for L in layers)))
    for position in POSITIONS:
        by_layer = summary["identification"][position]
        print(
            "  {:<18} {}".format(
                position,
                " ".join(
                    "{:>7.0%}".format(by_layer[L]["overall"]["rate"]) for L in layers
                ),
            )
        )

    for position, meaning in POSITIONS.items():
        best = summary["best_layer"][position]
        cell = summary["identification"][position][best]["overall"]
        print(
            "\n{} ({})\n  best at layer {}: {:.1%} ({}/{}, 95% CI {:.0%}-{:.0%})".format(
                position, meaning, best, cell["rate"], cell["successes"],
                cell["total"], cell["ci95"][0], cell["ci95"][1],
            )
        )
        split = summary["identification"][position][best]
        if "when_answer_correct" in split:
            print(
                "  on problems answered correctly {}, on problems answered "
                "wrongly {}".format(
                    _share(split["when_answer_correct"]),
                    _share(split["when_answer_wrong"]),
                )
            )

    cost = summary.get("reasoning_cost")
    if cost and cost.get("after"):
        print(
            "\nreading at layer {}, where the problem statement reads best "
            "({:.1%}), reasoning leaves identification at".format(
                cost["layer"], cost["before"]["rate"]
            )
        )
        for position, cell in cost["after"].items():
            print(
                "  {:<16} {:.1%}, a change of {:+.1f} points".format(
                    position, cell["rate"], 100 * cell["delta"]
                )
            )

    print("\naccuracy and reasoning length per surface form")
    for surface, accuracy in sorted(summary["accuracy_by_surface"].items()):
        thinking = summary["thinking"][surface]
        print(
            "  {:<10} {:.1%} correct ({}/{}), {:.0f} thinking tokens, "
            "{:.0%} cut off by the budget".format(
                surface, accuracy["rate"], accuracy["successes"], accuracy["total"],
                thinking["think_tokens_mean"], thinking["cut_off"]["rate"],
            )
        )


def _share(cell: dict) -> str:
    """A rate with its denominator, or a plain note when the group is empty."""
    if not cell["total"]:
        return "no such problems"
    return "{:.1%} ({}/{})".format(cell["rate"], cell["successes"], cell["total"])


def make_figure(args, summary: dict, run: runs.RunDirectory) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the figure")
        return None

    import torch

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    layers = sorted(summary["identification"]["problem_end"], key=int)
    for position in POSITIONS:
        by_layer = summary["identification"][position]
        axes[0].plot(
            [int(L) for L in layers],
            [by_layer[L]["overall"]["rate"] for L in layers],
            marker="o",
            label=position.replace("_", " "),
        )
    axes[0].axhline(summary["chance"], linestyle=":", color="grey", label="guessing")
    axes[0].set_xlabel("layer of the model")
    axes[0].set_ylabel("share matched to the same law")
    axes[0].set_title("Is the law still identifiable?")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # The two-dimensional views the earlier work used, side by side, so the
    # before-and-after difference can be seen and not only tabulated.
    surfaces = summary["surfaces"]
    loaded = {name: torch.load(activations_path(run, name)) for name in surfaces}
    reference_layer = summary["best_layer"]["problem_end"]
    # Of the three post-reasoning positions, draw whichever holds the law best,
    # so the panel shows the strongest case that anything survived.
    best_post = max(
        POST_REASONING,
        key=lambda p: summary["identification"][p][reference_layer]["overall"]["rate"],
    )
    for axis, position in zip(axes[1:], ("problem_end", best_post)):
        layer = int(reference_layer)
        stacked = torch.cat([loaded[name][position][layer] for name in surfaces])
        points, explained = principal_components(stacked)
        pair_ids = loaded[surfaces[0]]["pair_ids"]
        shown = min(args.laws_in_figure, len(pair_ids))
        colours = plt.cm.tab20(range(20))
        for surface_index, name in enumerate(surfaces):
            offset = surface_index * len(pair_ids)
            for law_index in range(shown):
                point = points[offset + law_index]
                axis.scatter(
                    point[0], point[1],
                    color=colours[law_index % 20],
                    marker=("o", "s", "^", "v", "D")[surface_index % 5],
                    s=28, alpha=0.85,
                )
        axis.set_title(
            "{} at layer {}\n{} laws, colour is the law, shape is the "
            "surface form".format(position.replace("_", " "), layer, shown)
        )
        axis.set_xlabel("first principal direction ({:.0%} of the spread)".format(explained[0]))
        axis.set_ylabel("second ({:.0%})".format(explained[1]))
        axis.grid(alpha=0.3)

    fig.tight_layout()
    path = run.figure_path("{}.png".format(EXPERIMENT))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def principal_components(vectors, k: int = 2):
    """Project onto the two directions of greatest spread, with their shares."""
    import torch

    centred = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(centred, full_matrices=False)
    variance = (s ** 2) / max(1, centred.shape[0] - 1)
    explained = (variance / variance.sum()).tolist()
    return (u[:, :k] * s[:k]).tolist(), explained[:k]


# ------------------------------------------------------------------------ main


def main(argv: Optional[List[str]] = None) -> int:
    cli = runs.base_parser(__doc__.split("\n")[0])
    cli.add_argument(
        "--surfaces",
        type=lambda text: tuple(s.strip() for s in text.split(",")),
        default=DEFAULT_SURFACES,
        help="surface forms to render each law in (default: %(default)s)",
    )
    cli.add_argument(
        "--layers",
        type=lambda text: tuple(int(L) for L in text.split(",")),
        default=DEFAULT_LAYERS,
        help="layers to read the residual stream at (default: %(default)s)",
    )
    cli.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="thinking-token budget; reading after reasoning needs it above "
        "zero (default: %(default)s)",
    )
    cli.add_argument(
        "--per-bin",
        type=int,
        default=DEFAULT_PER_BIN,
        help="laws sampled per operation-count bin (default: %(default)s)",
    )
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument(
        "--act-batch-size",
        type=int,
        default=4,
        help="batch size for reading activations, which holds every layer of a "
        "batch in memory at once and so has to be smaller (default: %(default)s)",
    )
    cli.add_argument(
        "--answer-tokens",
        type=int,
        default=DEFAULT_ANSWER_TOKENS,
        help="cap on the answer, after the thinking budget (default: %(default)s)",
    )
    cli.add_argument(
        "--laws-in-figure",
        type=int,
        default=10,
        help="how many laws to draw in the two-dimensional views; all of them "
        "is unreadable (default: %(default)s)",
    )
    args = cli.parse_args(argv)

    if args.quick:
        args.per_bin = 1
        args.budget = 32
        args.layers = (12, 24)
        args.surfaces = args.surfaces[:2]
        # Enough room for an answer to finish; below roughly 256 tokens every
        # row grades as unparseable and the smoke run stops testing grading.
        args.answer_tokens = min(args.answer_tokens, 256)

    if args.budget <= 0:
        raise SystemExit(
            "a budget of 0 means no reasoning trace, and there would be nothing "
            "to read after"
        )
    unknown = set(args.surfaces) - set(dataset.STORY_THEMES) - {"literal", "rg"}
    if unknown:
        raise SystemExit("unknown surface forms: {}".format(sorted(unknown)))

    conditions = build_conditions(args.surfaces, args.budget, args.per_bin)
    out_dir = runs.resolve_out_dir(args, __file__)
    run = runs.RunDirectory.open(
        out_dir,
        EXPERIMENT,
        {
            "model": args.model,
            "seed": args.seed,
            "surfaces": list(args.surfaces),
            "layers": list(args.layers),
            "budget": args.budget,
            "per_bin": args.per_bin,
            "answer_tokens": args.answer_tokens,
            "conditions": [c.as_dict() for c in conditions],
        },
        force=args.force,
    )

    if not args.analyze_only:
        run_experiment(args, run, conditions)

    summary = analyze(run, conditions)
    path = run.write_summary(summary)
    print("\nsummary -> {}".format(path))
    figure = make_figure(args, summary, run)
    if figure:
        print("figure  -> {}".format(figure))
    return 0


if __name__ == "__main__":
    sys.exit(main())
