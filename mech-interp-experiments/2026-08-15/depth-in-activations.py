#!/usr/bin/env python3
"""Do the model's internal activations order by depth, once length is constant?

QUESTION
--------
The companion experiment, depth-at-fixed-length.py, established that depth
changes what the model *does*: accuracy fell from 91.5% to 70.0% across the
four depths of one family and from 87.5% to 54.5% across the other, 200
problems per depth, while the number of operations, the number of variables,
the number of steps and the prompt length in tokens (1467 and 1525) were held
identical. This asks whether depth changes what the model *is*: does the
representation it builds while reading the problem carry the depth, and if so,
where in the network.

The earlier activation-structure work found that representations separate by
complexity, but complexity there meant operation count, which is the same
variable as text length in this dataset. That finding could not distinguish
"the model reresents how hard the problem is" from "the model represents how
long the input is". Here the texts are exactly the same length, so that
alternative is unavailable.

Three things are measured, at each layer and each read position:

1. **Ladder.** Take the direction from the shallowest cell's mean activation to
   the deepest cell's, and project every cell's mean onto it. If depth is
   represented as a graded quantity, the intermediate depths land in order and
   between the two ends. This needs no fitting and has no free parameters.
2. **Decoding.** Can depth be read off a single problem's activation? Nearest
   centroid, five-fold cross-validated, against the 25% chance rate of four
   equally sized depth classes.
3. **Failure.** Does the representation predict which individual problems this
   model gets wrong, beyond knowing their depth? Uses the graded answers from
   the behavioural run, matched by problem id.

The figure also shows the activations projected onto their first two principal
components — the two directions they vary along most — coloured by depth, which
is the earlier surface-form clustering picture redrawn with length held fixed.
That picture is an illustration and not one of the three measurements, for a
reason that has to be kept in view: principal components are found without
being told what depth is, so a direction can carry depth perfectly and still
hold too little of the variance to show up in it. Every panel therefore states
what share of the shallow-to-deep direction actually lies in the plane being
plotted. If that share is small, the scatter showing four overlapping clouds
says nothing, and the decoding number is what to read.

WHAT EACH OUTCOME WOULD MEAN
----------------------------
- Depth decodes well above the word-count floor, and the ladder is ordered:
  the model builds a graded representation of the problem's serial structure,
  and it is not a restatement of the input's surface. This is the result the
  earlier complexity finding wanted to claim and could not.
- Depth decodes no better than the word-count floor: whatever orders the
  activations is already in the text, and no internal construction needs to be
  invoked to explain it.
- Depth decodes but the ladder is not ordered: depth is distinguishable but not
  represented as a magnitude, i.e. the cells differ without lying on an axis.
  Worth knowing, and a weaker claim than a difficulty dimension.
- Failure is predictable above what depth alone predicts: the representation
  carries something about this particular problem's difficulty, not just its
  category. That would be the strongest link between the geometry and the
  behaviour.

CONTROLS AND WHAT LIMITS THIS
-----------------------------
- **The word-count floor is the control that matters.** Every measurement is
  repeated on plain word-count vectors of the same texts, and only the margin
  over that floor is evidence about the model. This is the same discipline the
  dataset-geometry experiment used.
- That floor should be near chance here, for a structural reason worth stating:
  in a description with k steps, exactly k - 1 of the intermediate results are
  consumed by a later step, whatever the tree's shape. So every problem in a
  family mentions Value 1 through Value k-1 exactly once each as an input, and
  the bag of words is the same at every depth. What differs is only which step
  refers to which. If the floor comes out well above chance, something in the
  surface does track depth after all and the margin is what to trust.
- Same cells, same seed and same surface form as the behavioural run, so the
  problems are literally the same ones and the success labels line up.
- Analysis is per family. Across families the operation count differs and with
  it the text length, which is the original confound; only within a family is
  length constant.
- This measures the representation of the *prompt*, before any answer is
  generated. It says what the model has built by the time it starts writing,
  not what it does while writing.
- **Read the layer profile, not the best cell.** Seven layers times two read
  positions times two families is 28 measurements, so a few will look good by
  chance. At the default size — 800 problems per family — both statistics sit
  within about 0.02 of their chance value when there is nothing to find
  (checked by running them on random vectors), so an isolated cell 0.05 above
  chance is noise and a run of neighbouring layers is not.

    python3 depth-in-activations.py --quick
    python3 depth-in-activations.py
    python3 depth-in-activations.py --analyze-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import dataset, runs, stats
from shared.vendor import ensure_on_path

ensure_on_path()

from benchmark import wrap_prompt  # noqa: E402

EXPERIMENT = "depth-in-activations"

# Layers to read, as indices into the hidden-state stack: 0 is the embedding
# output and 36 is the last block of Qwen3-4B. The middle of the stack is where
# the earlier work found law identity to be most decodable.
DEFAULT_LAYERS = (0, 6, 12, 18, 24, 30, 36)

# Where in the prompt to read. "last" is the final token, the one the model is
# about to generate from; "mean" averages over the whole prompt.
POSITIONS = ("last", "mean")

DEFAULT_PER_CELL = 200
DEFAULT_POOL_PER_BIN = 60000

# Both headline measurements are cross-validated, so how well chance does at
# them has to be measured rather than assumed. 200 shufflings is enough to
# separate a real effect from noise and costs a minute of processor time; it is
# analysis only, so it can be redone with --analyze-only and no GPU.
DEFAULT_PERMUTATIONS = 200


def build_conditions(
    cells: Sequence[dataset.Shape], per_cell: int, form: str
) -> List[runs.Condition]:
    """One condition per cell. No generation here, so no budget axis."""
    return [
        runs.Condition(
            "f{}-{}-{}_d{}".format(minor, major, variables, depth),
            minor=minor,
            major=major,
            variables=variables,
            depth=depth,
            per_cell=per_cell,
            form=form,
        )
        for minor, major, variables, depth in cells
    ]


def family_of(condition: runs.Condition) -> str:
    return "{}-{}-{}".format(
        condition["minor"], condition["major"], condition["variables"]
    )


def samples_for(pool: Sequence[str], condition: runs.Condition, seed: int) -> List[dict]:
    """The same problems the behavioural run used, given the same seed."""
    return dataset.sample_depth_balanced(
        pool,
        condition["per_cell"],
        seed,
        cells=((condition["minor"], condition["major"], condition["variables"],
                condition["depth"]),),
        form=condition["form"],
    )


def activations_path(run: runs.RunDirectory, condition: str) -> Path:
    directory = run.path / "activations"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "{}.npz".format(condition)


# ------------------------------------------------------------------- capture


def capture(args, run: runs.RunDirectory, conditions: List[runs.Condition]) -> None:
    from shared import generation, hooks, model as model_module

    todo = runs.pending(run, conditions, force=args.force)
    if not todo:
        print("every condition already has activations; nothing to capture")
        return

    print(runs.describe(conditions, run))
    print("\ncapturing {} of {} conditions".format(len(todo), len(conditions)))
    print("building datasets for all {} conditions".format(len(todo)))
    pool = dataset.build_pool(per_bin=args.pool_per_bin)
    problems = {c.name: samples_for(pool, c, args.seed) for c in todo}

    model, tokenizer = model_module.load(args.model)
    model_module.check_layers(model, args.layers)
    # Borrowed for its chat formatting alone; nothing here generates. Going
    # through it rather than calling the tokenizer directly is what guarantees
    # the text read here is the text the behavioural run was answering.
    generator = generation.Generator(model, tokenizer, batch_size=args.batch_size)

    for condition in todo:
        samples = problems[condition.name]
        prompts = [
            wrap_prompt(s["prompt"], "on", "", condition["form"]) for s in samples
        ]
        chats = [generator.build_chat(prompt, thinking=True) for prompt in prompts]
        prompt_tokens = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]

        captured = hooks.capture_residuals(
            model,
            tokenizer,
            chats,
            args.layers,
            batch_size=args.batch_size,
            progress=condition.name,
        )

        # Publish the activations before the records: the records file is what
        # marks a condition finished, so a crash in between simply redoes it.
        arrays = {
            "{}_L{}".format(position, layer): captured[position][layer].numpy()
            for position in POSITIONS
            for layer in args.layers
        }
        path = activations_path(run, condition.name)
        temporary = path.parent / (path.name + ".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)

        with run.writing(condition.name) as writer:
            for row, sample in enumerate(samples):
                writer.write(
                    dict(
                        condition.settings,
                        condition=condition.name,
                        pair_id=sample["pair_id"],
                        row=row,
                        prompt_tokens=prompt_tokens[row],
                        # The problem text without the instruction wrapper,
                        # which is the same for every problem. Kept so the
                        # word-count floor can be rebuilt with --analyze-only.
                        text=sample["prompt"],
                    )
                )
        print(
            "  {:<20} {} problems, {} layers x {} positions".format(
                condition.name, len(samples), len(args.layers), len(POSITIONS)
            )
        )


# ------------------------------------------------------------- numpy analysis


def ladder(vectors: np.ndarray, depths: np.ndarray) -> dict:
    """Project each depth's mean activation onto the shallow-to-deep axis.

    Ordered, evenly spaced projections mean depth is held as a magnitude rather
    than as four unrelated categories. Scaled so the shallowest is 0 and the
    deepest is 1, which makes the middle two readable at a glance.
    """
    levels = sorted(set(depths.tolist()))
    centroids = {d: vectors[depths == d].mean(axis=0) for d in levels}
    axis = centroids[levels[-1]] - centroids[levels[0]]
    norm = np.linalg.norm(axis)
    if norm == 0:
        return {"ordered": False, "projections": {}}
    axis = axis / norm
    raw = {d: float(centroids[d] @ axis) for d in levels}
    low, high = raw[levels[0]], raw[levels[-1]]
    scaled = {d: (raw[d] - low) / (high - low) for d in levels}
    values = [scaled[d] for d in levels]
    return {
        "projections": {str(d): scaled[d] for d in levels},
        "ordered": all(a < b for a, b in zip(values, values[1:])),
        "separation": float(high - low),
    }


def principal_components(vectors: np.ndarray, k: int = 3):
    """Scores on the top k principal components, and what share of variance each holds.

    Principal components are the directions the activations vary along most,
    found without being told what depth is. They are here for the picture and
    for one diagnostic — see depth_axis_in_plane — not as the test, because a
    direction can carry depth perfectly and still hold too little of the
    variance to appear in a plot of the first two components.
    """
    centered = vectors - vectors.mean(axis=0)
    _, singular, directions = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2
    return (
        centered @ directions[:k].T,
        variance[:k] / variance.sum(),
        directions[:k],
    )


def depth_axis_in_plane(vectors: np.ndarray, depths: np.ndarray, directions: np.ndarray) -> float:
    """How much of the shallow-to-deep axis lies in the plotted plane, 0 to 1.

    This is what says whether a scatter of the first two principal components
    could have shown the depth effect on its own. Near 1 and the picture is the
    evidence; near 0 and the depths can be perfectly separated in the
    activations while the picture shows four overlapping clouds, because the
    direction that separates them points out of the page.
    """
    levels = sorted(set(depths.tolist()))
    axis = vectors[depths == levels[-1]].mean(axis=0) - vectors[depths == levels[0]].mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return float("nan")
    return float((((axis / norm) @ directions.T) ** 2).sum())


def nearest_centroid_accuracy(
    vectors: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0
) -> float:
    """Cross-validated accuracy of assigning each point to the nearest class mean.

    Chosen over a fitted probe because it has no hyperparameter to tune, so
    there is nothing to tune differently between the activations and the
    word-count floor they are compared against.
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(labels))
    vectors, labels = vectors[order], labels[order]
    classes = sorted(set(labels.tolist()))
    correct = 0
    for fold in range(folds):
        test = np.arange(len(labels)) % folds == fold
        train = ~test
        centroids = []
        for c in classes:
            rows = vectors[train & (labels == c)]
            if not len(rows):
                return float("nan")
            centroids.append(rows.mean(axis=0))
        centroids = np.stack(centroids)
        distances = ((vectors[test][:, None, :] - centroids[None]) ** 2).sum(axis=2)
        predicted = np.array(classes)[distances.argmin(axis=1)]
        correct += int((predicted == labels[test]).sum())
    return correct / len(labels)


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve, from ranks. 0.5 is chance."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Ties share their average rank, or the score becomes order-dependent.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def separation_auc(
    vectors: np.ndarray, positive: np.ndarray, folds: int = 5, seed: int = 0
) -> float:
    """Cross-validated AUC of projecting onto the difference of class means."""
    if positive.all() or not positive.any():
        return float("nan")
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(positive))
    vectors, positive = vectors[order], positive[order]
    scores = np.zeros(len(positive))
    for fold in range(folds):
        test = np.arange(len(positive)) % folds == fold
        train = ~test
        if not positive[train].any() or positive[train].all():
            return float("nan")
        axis = vectors[train & positive].mean(axis=0) - vectors[train & ~positive].mean(axis=0)
        scores[test] = vectors[test] @ axis
    return auc(scores, positive)


def permutation_p(score, labels: np.ndarray, observed: float, rounds: int, seed: int = 0) -> Optional[float]:
    """How often shuffled labels score at least as well as the real ones.

    Both measurements here are cross-validated, so their null distribution is
    not the textbook one and cannot be looked up; shuffling the labels and
    refitting gives it directly. Reported as a fraction of `rounds`, so the
    smallest value it can return is 1 / (rounds + 1).
    """
    if rounds <= 0 or observed != observed:  # NaN observed
        return None
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    beaten = 0
    for _ in range(rounds):
        null = score(labels[rng.permutation(len(labels))])
        beaten += null == null and null >= observed
    return (beaten + 1) / (rounds + 1)


def word_count_matrix(texts: Sequence[str]) -> np.ndarray:
    """Plain word counts: the model-free floor every measurement is judged against."""
    vocabulary: Dict[str, int] = {}
    for text in texts:
        for word in text.split():
            vocabulary.setdefault(word, len(vocabulary))
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.split():
            matrix[row, vocabulary[word]] += 1
    return matrix


# -------------------------------------------------------------------- analysis


def analyze(
    run: runs.RunDirectory,
    conditions: List[runs.Condition],
    layers: Sequence[int],
    behaviour: Optional[Dict[str, bool]] = None,
    permutations: int = DEFAULT_PERMUTATIONS,
) -> dict:
    finished = [c for c in conditions if run.has(c.name)]
    if not finished:
        raise SystemExit(
            "no condition has activations yet in {}; run without "
            "--analyze-only first".format(run.path)
        )

    records = {c.name: run.read(c.name) for c in finished}
    loaded = {c.name: np.load(activations_path(run, c.name)) for c in finished}

    summary: dict = {
        "families": {},
        "layers": list(layers),
        "permutations": permutations,
    }
    for family in sorted({family_of(c) for c in finished}):
        cells = [c for c in finished if family_of(c) == family]
        depths = np.concatenate(
            [np.full(len(records[c.name]), c["depth"]) for c in cells]
        )
        texts = [r["text"] for c in cells for r in records[c.name]]
        prompt_tokens = sorted(
            {r["prompt_tokens"] for c in cells for r in records[c.name]}
        )

        correct = None
        if behaviour is not None:
            flags = [
                behaviour.get(r["pair_id"]) for c in cells for r in records[c.name]
            ]
            matched = sum(flag is not None for flag in flags)
            if matched == len(flags):
                correct = np.array(flags, dtype=bool)
            else:
                # The two runs drew different problems, which happens when they
                # were given different --per-cell or --seed values. Pairing the
                # ones that do match would compare a biased subset, so the
                # failure analysis is dropped rather than run on part of it.
                print(
                    "family {}: only {} of {} problems appear in the "
                    "behavioural run, so the failure analysis is skipped; the "
                    "two runs need the same --seed, --per-cell and --form "
                    "to draw the same problems".format(family, matched, len(flags))
                )

        floor = word_count_matrix(texts)
        block: dict = {
            "prompt_token_lengths": prompt_tokens,
            "prompt_length_constant": len(prompt_tokens) == 1,
            "n": int(len(depths)),
            "word_count_floor": {
                "depth_decoding": nearest_centroid_accuracy(floor, depths),
                "failure_auc": (
                    None if correct is None else separation_auc(floor, correct)
                ),
            },
            "chance_depth_decoding": 1.0 / len(set(depths.tolist())),
            "by_position": {},
        }
        if correct is not None:
            block["accuracy_by_depth"] = {
                str(c["depth"]): stats.rate_of(
                    behaviour[r["pair_id"]] for r in records[c.name]
                ).as_dict()
                for c in cells
            }
            block["depth_only_failure_auc"] = auc(-depths.astype(float), correct)

        for position in POSITIONS:
            per_layer = {}
            for layer in layers:
                key = "{}_L{}".format(position, layer)
                vectors = np.concatenate(
                    [loaded[c.name][key] for c in cells]
                ).astype(np.float64)
                decoding = nearest_centroid_accuracy(vectors, depths)
                _, explained, directions = principal_components(vectors, k=2)
                entry = {
                    "ladder": ladder(vectors, depths),
                    "pca_variance_explained": explained.tolist(),
                    "depth_axis_in_pca_plane": depth_axis_in_plane(
                        vectors, depths, directions
                    ),
                    "depth_decoding": decoding,
                    "depth_decoding_p": permutation_p(
                        lambda y: nearest_centroid_accuracy(vectors, y),
                        depths,
                        decoding,
                        permutations,
                    ),
                }
                if correct is not None:
                    value = separation_auc(vectors, correct)
                    entry["failure_auc"] = value
                    entry["failure_auc_p"] = permutation_p(
                        lambda y: separation_auc(vectors, y),
                        correct,
                        value,
                        permutations,
                    )
                per_layer[str(layer)] = entry
            block["by_position"][position] = per_layer
        summary["families"][family] = block

    print_summary(summary, layers)
    return summary


def load_behaviour(path: Optional[Path]) -> Optional[Dict[str, bool]]:
    """Per-problem success from the behavioural run, keyed by problem id.

    That run may have swept several thinking budgets, in which case one problem
    has several verdicts. Only one can be used, and the largest budget is the
    one that says whether the model can do the problem at all, so the others
    are dropped and the choice is printed.
    """
    if path is None:
        return None
    directory = Path(path)
    records = directory / "records"
    if not records.is_dir():
        raise SystemExit(
            "{} has no records/ directory; point --behaviour-run at a run "
            "directory of depth-at-fixed-length.py".format(directory)
        )

    rows = [
        json.loads(line)
        for file in sorted(records.glob("*.jsonl"))
        for line in file.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("no graded records found under {}".format(records))

    budgets = sorted({row.get("budget") for row in rows})
    if len(budgets) > 1:
        rows = [row for row in rows if row.get("budget") == budgets[-1]]
        print(
            "{} swept thinking budgets {}; using the largest, {} tokens".format(
                directory, budgets, budgets[-1]
            )
        )
    out = {row["pair_id"]: row["status"] == "correct" for row in rows}
    print(
        "loaded {} graded answers from {} ({:.1%} correct)".format(
            len(out), directory, sum(out.values()) / len(out)
        )
    )
    return out


def print_summary(summary: dict, layers: Sequence[int]) -> None:
    for family, block in sorted(summary["families"].items()):
        print("\nfamily {} ({} problems)".format(family, block["n"]))
        lengths = block["prompt_token_lengths"]
        print(
            "  prompt length in tokens: {}{}".format(
                lengths,
                "" if block["prompt_length_constant"] else "  <-- NOT CONSTANT",
            )
        )
        floor = block["word_count_floor"]
        print(
            "  word-count floor: depth decoding {:.1%} (chance {:.0%})".format(
                floor["depth_decoding"], block["chance_depth_decoding"]
            )
            + (
                ""
                if floor["failure_auc"] is None
                else ", failure AUC {:.3f}".format(floor["failure_auc"])
            )
        )
        if "depth_only_failure_auc" in block:
            print(
                "  knowing depth alone predicts failure at AUC {:.3f}".format(
                    block["depth_only_failure_auc"]
                )
            )

        row = "    {:>5} {:>9} {:>9} {:>6}  {:<32} {:>9} {:>6} {:>8}"
        for position, per_layer in sorted(block["by_position"].items()):
            print("\n  read position: {}".format(position))
            print(
                row.format(
                    "layer",
                    "depth",
                    "vs floor",
                    "p",
                    "ladder rungs, shallow to deep",
                    "fail AUC",
                    "p",
                    "in plot",
                )
            )
            for layer in layers:
                entry = per_layer[str(layer)]
                rungs = entry["ladder"]["projections"]
                shown = " ".join(
                    "{:.2f}".format(rungs[d]) for d in sorted(rungs, key=int)
                )
                if not entry["ladder"]["ordered"]:
                    shown += "  out of order"
                print(
                    row.format(
                        layer,
                        "{:.1%}".format(entry["depth_decoding"]),
                        "{:+.1f}".format(
                            100 * (entry["depth_decoding"] - floor["depth_decoding"])
                        ),
                        _p(entry.get("depth_decoding_p")),
                        shown,
                        "n/a"
                        if entry.get("failure_auc") is None
                        else "{:.3f}".format(entry["failure_auc"]),
                        _p(entry.get("failure_auc_p")),
                        "{:.1%}".format(entry["depth_axis_in_pca_plane"]),
                    )
                )
        if summary["permutations"]:
            print(
                "\n  p is the share of {} label shufflings that scored at least "
                "as well as the real labels".format(summary["permutations"])
            )
        print(
            "  'in plot' is how much of the shallow-to-deep direction lies in "
            "the plane of the\n  first two principal components, i.e. how much "
            "of the effect a scatter plot of them\n  could show; the rest points "
            "out of the page"
        )


def _p(value: Optional[float]) -> str:
    return "-" if value is None else "{:.3f}".format(value)


def make_figure(
    summary: dict,
    run: runs.RunDirectory,
    conditions: List[runs.Condition],
    layers: Sequence[int],
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the figure")
        return None

    families = sorted(summary["families"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    scatter_axes, (decode_ax, failure_ax) = axes[0], axes[1]

    # Top row: the picture, one problem set each, at the layer where depth is
    # most decodable. Reads the saved activations rather than the summary, so a
    # redraw needs the run directory but no model.
    for panel, family in zip(scatter_axes, families):
        block = summary["families"][family]
        position, layer = best_cell(block, layers)
        vectors, depths = family_vectors(run, conditions, family, position, layer)
        scores, explained, _ = principal_components(vectors, k=2)
        for depth in sorted(set(depths.tolist())):
            rows = depths == depth
            panel.scatter(
                scores[rows, 0],
                scores[rows, 1],
                s=9,
                alpha=0.65,
                label="depth {}".format(depth),
            )
        entry = block["by_position"][position][str(layer)]
        panel.set_title(
            "{}: layer {}, {} token\n"
            "depth reads off at {:.0%} against a {:.0%} word-count floor; "
            "{:.0%} of that\ndirection lies in this plane".format(
                family,
                layer,
                position,
                entry["depth_decoding"],
                block["word_count_floor"]["depth_decoding"],
                entry["depth_axis_in_pca_plane"],
            ),
            fontsize=8,
        )
        panel.set_xlabel(
            "1st principal component ({:.0%} of variance)".format(explained[0])
        )
        panel.set_ylabel(
            "2nd principal component ({:.0%})".format(explained[1])
        )
        panel.legend(fontsize=7)
        panel.grid(alpha=0.2)
    for panel in scatter_axes[len(families):]:
        panel.axis("off")

    for family in families:
        block = summary["families"][family]
        for position in POSITIONS:
            per_layer = block["by_position"][position]
            decode_ax.plot(
                list(layers),
                [per_layer[str(L)]["depth_decoding"] for L in layers],
                marker="o",
                label="{} ({})".format(family, position),
            )
            if per_layer[str(layers[0])].get("failure_auc") is not None:
                failure_ax.plot(
                    list(layers),
                    [per_layer[str(L)]["failure_auc"] for L in layers],
                    marker="o",
                    label="{} ({})".format(family, position),
                )
        decode_ax.axhline(
            block["word_count_floor"]["depth_decoding"],
            linestyle="--",
            alpha=0.5,
            label="{} word-count floor".format(family),
        )

    chance = summary["families"][families[0]]["chance_depth_decoding"]
    decode_ax.axhline(chance, color="grey", linestyle=":", label="chance")
    decode_ax.set_xlabel("layer")
    decode_ax.set_ylabel("depth read off one problem's activations")
    decode_ax.set_title("Can depth be decoded, and does it beat the text?", fontsize=9)
    decode_ax.set_ylim(0, 1)
    decode_ax.grid(alpha=0.3)
    decode_ax.legend(fontsize=7)

    failure_ax.axhline(0.5, color="grey", linestyle=":", label="chance")
    failure_ax.set_xlabel("layer")
    failure_ax.set_ylabel("predicting which problems fail (AUC)")
    failure_ax.set_title("Does the representation know it will fail?", fontsize=9)
    failure_ax.grid(alpha=0.3)
    failure_ax.legend(fontsize=7)

    fig.tight_layout()
    path = run.figure_path("{}.png".format(EXPERIMENT))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def best_cell(block: dict, layers: Sequence[int]):
    """The read position and layer where depth decodes best, for the picture."""
    return max(
        ((position, layer) for position in POSITIONS for layer in layers),
        key=lambda pair: block["by_position"][pair[0]][str(pair[1])]["depth_decoding"],
    )


def family_vectors(
    run: runs.RunDirectory,
    conditions: List[runs.Condition],
    family: str,
    position: str,
    layer: int,
):
    """One problem set's activations at one layer, with each row's depth."""
    cells = [c for c in conditions if family_of(c) == family and run.has(c.name)]
    key = "{}_L{}".format(position, layer)
    vectors, depths = [], []
    for cell in cells:
        rows = np.load(activations_path(run, cell.name))[key]
        vectors.append(rows)
        depths.append(np.full(len(rows), cell["depth"]))
    return np.concatenate(vectors).astype(np.float64), np.concatenate(depths)


# ------------------------------------------------------------------------ main


def main(argv: Optional[List[str]] = None) -> int:
    cli = runs.base_parser(__doc__.split("\n")[0])
    cli.add_argument(
        "--layers",
        type=lambda text: tuple(int(L) for L in text.split(",")),
        default=DEFAULT_LAYERS,
        help="hidden-state indices to read (default: %(default)s)",
    )
    cli.add_argument("--per-cell", type=int, default=DEFAULT_PER_CELL)
    cli.add_argument("--form", default="literal", choices=("literal", "story"))
    cli.add_argument(
        "--cells",
        type=dataset.parse_cells,
        default=dataset.parse_cells(dataset.DEFAULT_CELLS),
        metavar="MINOR:MAJOR:VARS:DEPTH,...",
    )
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--pool-per-bin", type=int, default=DEFAULT_POOL_PER_BIN)
    cli.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="label shufflings used to measure how well chance does at each "
        "measurement; 0 skips them (default: %(default)s)",
    )
    cli.add_argument(
        "--behaviour-run",
        type=Path,
        default=None,
        help="run directory of depth-at-fixed-length.py, to link each "
        "problem's activations to whether the model answered it correctly "
        "(default: the sibling run of that experiment, if present)",
    )
    args = cli.parse_args(argv)

    if args.quick:
        args.per_cell = min(args.per_cell, 8)
        args.layers = (0, 18, 36)
        args.pool_per_bin = min(args.pool_per_bin, 4000)
        args.permutations = min(args.permutations, 20)

    conditions = build_conditions(args.cells, args.per_cell, args.form)
    out_dir = runs.resolve_out_dir(args, __file__)
    run = runs.RunDirectory.open(
        out_dir,
        EXPERIMENT,
        {
            "model": args.model,
            "seed": args.seed,
            "per_cell": args.per_cell,
            "form": args.form,
            "layers": list(args.layers),
            "pool_per_bin": args.pool_per_bin,
            "cells": [list(cell) for cell in args.cells],
            "conditions": [c.as_dict() for c in conditions],
        },
        force=args.force,
    )

    if not args.analyze_only:
        capture(args, run, conditions)

    behaviour_dir = args.behaviour_run
    if behaviour_dir is None:
        sibling = out_dir / "runs" / "depth-at-fixed-length"
        behaviour_dir = sibling if (sibling / "records").is_dir() else None
        if behaviour_dir is None:
            print(
                "no behavioural run found at {}; skipping the failure-"
                "prediction analysis (pass --behaviour-run to point at "
                "one)".format(sibling)
            )
    behaviour = load_behaviour(behaviour_dir)

    summary = analyze(run, conditions, args.layers, behaviour, args.permutations)
    path = run.write_summary(summary)
    print("\nsummary -> {}".format(path))
    figure = make_figure(summary, run, conditions, args.layers)
    if figure:
        print("figure  -> {}".format(figure))
    return 0


if __name__ == "__main__":
    sys.exit(main())
