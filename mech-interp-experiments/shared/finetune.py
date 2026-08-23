#!/usr/bin/env python3
"""The fine-tuning checkpoints, and which laws they were trained on.

The checkpoints live at the Hugging Face repository
`SoHarshh/mars-v-ft-checkpoints`. They are LoRA adapters: the base model's
weights are frozen and training adds only a small low-rank correction, saved
every so often. Because the base never changes, activations read from two
different checkpoints are in the same coordinate system and can be compared
directly — which is what makes a single frozen set of axes meaningful across a
whole training run, and is not true of two independently trained models.

Two recipes exist for each of two base models. They share the base, the adapter
settings, the seed and the schedule, and differ only in training data:

- `task-pairs`   trained on themed story paired with the answer notation.
                 Learns the task; on the 32-billion-parameter base, accuracy
                 goes from 34% to 99.7%.
- `grammar-only` trained on the bare answer notation with no story attached.
                 Learns to produce flawless notation that says the wrong thing;
                 on the same base, accuracy goes from 34% to 3%.

The second is the control any across-checkpoint measurement needs. Both runs
perturb every layer, so "the activations moved" on its own says only that the
weights moved. Only the contrast between a run that learned the task and a run
that did not separates the two.

Which laws were trained on
--------------------------

The training corpus is committed on the `harsh/experiments` branch of this
repository at `ft-experiments/train_v2/train.jsonl`, 2,772 rows. Any experiment
that reads activations for a law the model was fine-tuned on is measuring
memorization as much as learning, so laws appearing in that corpus have to be
excluded rather than assumed absent. They are not rare: of the 100 laws in the
three-operation cell this project samples, 44 appear in training.

Exclusion is by *law class*, not by equation number. `law_class_key` folds
together the four readings a law has that the grader treats as the same thing —
the equation, its two sides swapped, its dual (every operation's arguments
mirrored), and both — so a law cannot slip through by arriving swapped or
dualized. This is a port of `law_class_key` in the branch's
`ft-experiments/data-gen/ftlib.py`, and it is verified rather than assumed: it
reproduces all 5,544 `law_hash_e` / `law_hash_f` values already recorded in the
corpus, exactly.

    python3 -m shared.finetune --train-jsonl PATH --out training-laws.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .vendor import ensure_on_path

ensure_on_path()

from checkform import dual  # noqa: E402
from storyform import ParseError, Term, canonical, parse_equation  # noqa: E402

HF_REPO = "SoHarshh/mars-v-ft-checkpoints"

# The themes the story renderer offers, sorted, with the last held out of
# training entirely so the fine-tune's own generalization to an unseen theme
# can be measured. Recorded here because an experiment that reads a story
# activation needs to know whether that theme was trained on.
TRAINED_THEMES = ("graft", "paint", "signal")
HELDOUT_THEME = "tea"


class Run:
    """One training trajectory: a base model, a recipe, and its checkpoints."""

    def __init__(self, name: str, base_model: str, steps: Sequence[int]):
        self.name = name
        self.base_model = base_model
        self.steps = tuple(steps)

    def subfolder(self, step: int) -> str:
        """Where this checkpoint's adapter sits inside the Hugging Face repo."""
        if step not in self.steps:
            raise ValueError(
                "{} has no step {}; it has {}".format(self.name, step, self.steps)
            )
        return "{}/step-{}".format(self.name, step)

    def __repr__(self) -> str:
        return "Run({!r}, {} checkpoints)".format(self.name, len(self.steps))


# Step 0 of every run is the freshly initialized, untrained adapter, so each
# trajectory has its own exact origin and the first checkpoint is the baseline.
RUNS: Dict[str, Run] = {
    run.name: run
    for run in (
        Run(
            "llama-3.1-8b_grammar-only",
            "meta-llama/Llama-3.1-8B-Instruct",
            (0, 10, 20, 30, 40, 50, 60, 70, 75),
        ),
        Run(
            "llama-3.1-8b_task-pairs",
            "meta-llama/Llama-3.1-8B-Instruct",
            (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1041),
        ),
        Run(
            "ministral-3-14b_task-pairs",
            "mistralai/Ministral-3-14B-Instruct-2512",
            (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1041),
        ),
        Run(
            "qwen3-32b_grammar-only",
            "Qwen/Qwen3-32B",
            (0, 10, 20, 30, 40, 50, 60, 70, 75),
        ),
        # Stops at 900 rather than a full three epochs: the training container
        # was lost to a network failure at about 2.6 epochs. Its accuracy curve
        # is flat from step 500, so little was left to change.
        Run(
            "qwen3-32b_task-pairs",
            "Qwen/Qwen3-32B",
            (0, 100, 200, 300, 400, 500, 600, 700, 800, 900),
        ),
    )
}


def paired_runs(base_prefix: str) -> Tuple[Run, Run]:
    """The task-pairs run and its grammar-only control for one base model.

    The shared base is checked rather than assumed. The control only controls
    for anything if both runs started from the same weights: activations from
    two different bases are in unrelated coordinate systems, so projecting them
    into one frame would produce a difference between the recipes that is
    really just a difference between two models, and nothing downstream could
    tell the two apart.
    """
    try:
        task = RUNS["{}_task-pairs".format(base_prefix)]
        grammar = RUNS["{}_grammar-only".format(base_prefix)]
    except KeyError:
        raise SystemExit(
            "no task-pairs/grammar-only pair for {!r}; have {}".format(
                base_prefix, sorted(RUNS)
            )
        )
    if task.base_model != grammar.base_model:
        raise SystemExit(
            "{} and {} start from different base models ({} and {}), so they "
            "cannot be compared in one set of axes".format(
                task.name, grammar.name, task.base_model, grammar.base_model
            )
        )
    return task, grammar


# ------------------------------------------------------- Law-class identity


def law_class_key(lhs: Term, rhs: Term) -> str:
    """The identity of a law under every reading the grader accepts as the same.

    An equation, its two sides swapped, its dual, and its dual swapped all
    describe the same law as far as grading is concerned, so all four must
    collapse to one key. Taking the smallest of the four canonical strings does
    that without needing to pick a preferred reading.
    """
    return min(
        canonical(lhs, rhs),
        canonical(rhs, lhs),
        canonical(dual(lhs), dual(rhs)),
        canonical(dual(rhs), dual(lhs)),
    )


def law_hash(lhs: Term, rhs: Term) -> str:
    """The law class as a short digest, matching the fine-tuning corpus."""
    return hashlib.sha256(law_class_key(lhs, rhs).encode("utf-8")).hexdigest()[:16]


def equation_law_hash(equation_text: str) -> Optional[str]:
    """One equation's law hash, or None if it does not parse."""
    try:
        return law_hash(*parse_equation(equation_text))
    except (ParseError, ValueError):
        return None


def load_trained_law_hashes(path: Path) -> Set[str]:
    """The set of law hashes the fine-tune was trained on."""
    blob = json.loads(Path(path).read_text())
    return set(blob["law_hashes"])


def untouched_equation_numbers(
    equations: Sequence[str], trained: Set[str], numbers: Iterable[int]
) -> List[int]:
    """Those equation numbers whose law never appears in the training corpus.

    Numbers are one-based, matching the published list's own numbering, so
    `equations[n - 1]` is equation n.
    """
    keep = []
    for number in numbers:
        digest = equation_law_hash(equations[number - 1])
        if digest is not None and digest not in trained:
            keep.append(number)
    return keep


# ------------------------------------------------------------------ Rebuild


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument(
        "--train-jsonl",
        type=Path,
        required=True,
        help="ft-experiments/train_v2/train.jsonl from the harsh/experiments branch",
    )
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args(argv)

    rows = [
        json.loads(line)
        for line in Path(args.train_jsonl).read_text().splitlines()
        if line.strip()
    ]

    # Recompute every hash from the canonical equations rather than trusting the
    # recorded ones. Agreement proves this port and the corpus mean the same
    # thing by "the same law"; a silent divergence here would let trained laws
    # through the screen while the run looked clean.
    checked = mismatched = 0
    for row in rows:
        for side in ("e", "f"):
            digest = equation_law_hash(row["canonical_{}".format(side)])
            checked += 1
            if digest != row["law_hash_{}".format(side)]:
                mismatched += 1
    if mismatched:
        raise SystemExit(
            "law_hash port disagrees with the corpus on {} of {} laws".format(
                mismatched, checked
            )
        )

    hashes = sorted(
        {row["law_hash_e"] for row in rows} | {row["law_hash_f"] for row in rows}
    )
    blob = {
        "source": "ft-experiments/train_v2/train.jsonl (branch harsh/experiments)",
        "n_rows": len(rows),
        "n_law_hashes": len(hashes),
        "verified_against_recorded_hashes": checked,
        "trained_themes": list(TRAINED_THEMES),
        "heldout_theme": HELDOUT_THEME,
        "law_hashes": hashes,
    }
    Path(args.out).write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    print(
        "{} rows -> {} distinct law classes, {} hashes verified -> {}".format(
            len(rows), len(hashes), checked, args.out
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
