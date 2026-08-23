#!/usr/bin/env python3
"""Build the law pairs an experiment runs on.

An implication pair is two algebraic laws, E and F, rendered into text that
asks "if a world always obeys rule E, must it also obey rule F?". The renderers
live in the vendored checkout; what this module adds is control over *which*
pairs get drawn, and in particular the depth grid described below.

Why the depth grid exists
-------------------------

The dataset-geometry experiment found that word count correlates with the
number of operations in a law at 0.93 to 1.00 within every surface form, so any
apparent "complexity gradient" in the model's activations is a text-length
gradient until shown otherwise. That correlation cannot be sampled away: every
renderer emits one fixed-length step per operation, so a law's word count is
essentially affine in its operation count, and no long two-operation law exists
to select as a counterweight.

Depth — the height of the term tree, i.e. the length of the chain of steps that
must be evaluated one after another — is not constrained that way. At a fixed
operation count the number of steps is fixed too, but the tree can be a spine
(each step consuming the previous result) or bushy (independent branches). So
sampling laws that agree on everything a renderer's length depends on, and
differ only in tree shape, gives an ordered complexity variable with text
length held fixed exactly rather than approximately.

A *cell* is (minor side operations, major side operations, variables, depth),
and a *family* is a set of cells differing only in depth. All three pinned
quantities have to be in the key:

- operation count, because length is affine in it;
- the side split, because depth is a property of one *side* of the equation, so
  a k-operation equation only reaches depth k by putting all k operations on
  one side. Sampling on depth alone drifts toward bare-sided laws as depth
  rises, and a bare side renders shorter;
- variable count, because it sets the length of every quantifier clause, and
  the pool's variable mix shifts slightly with depth.

Depth is bounded by the major side, so useful families need more operations
than the published ETP law list's cap of four; build_pool synthesizes them.

    python3 -m shared.dataset --list-cells
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .vendor import ensure_on_path

ensure_on_path()

from benchmark import load_equations, make_sample, sample_pairs_stratified  # noqa: E402
from filter_vacuous import is_vacuous  # noqa: E402
from genform import generate_corpus  # noqa: E402
from literalform import render_description  # noqa: E402
from storyform import (  # noqa: E402
    ParseError,
    Term,
    Var,
    parse_equation,
    render_story,
    variables_in_order,
)

Shape = Tuple[int, int, int, int]  # minor ops, major ops, variables, depth

# The arguments behind the default pool: 8000 laws at 6 and 7 operations, the
# counts the two default families need. Generation is a pure function of these,
# so the corpus never needs committing or uploading. Written out by
# `genform.py --seed 0 --bins 6:7 --per-bin 4000` it is sha256
# 487a66390266a3d3870f721e7a26195e31f814dd189fb0e824b75e0b52a262ac.
POOL_SEED = 0
POOL_BINS = range(6, 8)
POOL_PER_BIN = 4000

# Synthetic laws must never be labelled as though they carried ETP numbering.
LABEL_PREFIX = "G"

# The two families the depth experiments use, four depths each.
DEFAULT_CELLS = "0:6:3:3-6,0:7:4:4-7"

# The surface forms the geometry experiments compare: four story themes, the
# literal description, and Rigid Grammar (the terse formal rendering).
STORY_THEMES = ("graft", "paint", "signal", "tea")


def build_pool(
    seed: int = POOL_SEED,
    bins: range = POOL_BINS,
    per_bin: int = POOL_PER_BIN,
) -> List[str]:
    """The law list to sample from, generated rather than read from disk."""
    return generate_corpus(seed, bins, per_bin)


# --------------------------------------------------------------------- Shapes


def _term_ops(term: Term) -> int:
    if isinstance(term, Var):
        return 0
    return 1 + _term_ops(term.left) + _term_ops(term.right)


def _term_depth(term: Term) -> int:
    if isinstance(term, Var):
        return 0
    return 1 + max(_term_depth(term.left), _term_depth(term.right))


def complexity_tags(metadata: dict) -> dict:
    """The per-pair complexity fields the vendored benchmark records.

    Kept identical so rows from here stay comparable with rows from any run
    driven by the checkout's own command line.
    """
    e_lhs, e_rhs = parse_equation(metadata["equation_e"])
    f_lhs, f_rhs = parse_equation(metadata["equation_f"])
    ops_e = _term_ops(e_lhs) + _term_ops(e_rhs)
    ops_f = _term_ops(f_lhs) + _term_ops(f_rhs)
    return {
        "ops_e": ops_e,
        "ops_f": ops_f,
        "ops_total": ops_e + ops_f,
        "depth": max(map(_term_depth, (e_lhs, e_rhs, f_lhs, f_rhs))),
    }


def equation_shape(text: str) -> Optional[Shape]:
    """One equation's cell, or None if it cannot be parsed.

    Depth is the deeper side's nesting, matching what complexity_tags reports
    for a pair.
    """
    try:
        lhs, rhs = parse_equation(text)
    except ParseError:
        return None
    sides = sorted((_term_ops(lhs), _term_ops(rhs)))
    return (
        sides[0],
        sides[1],
        len(variables_in_order(lhs, rhs)),
        max(_term_depth(lhs), _term_depth(rhs)),
    )


def index_by_shape(equations: Sequence[str]) -> Dict[Shape, List[int]]:
    """Equation numbers grouped by cell, ascending within a cell."""
    by_shape: Dict[Shape, List[int]] = {}
    for number, text in enumerate(equations, start=1):
        shape = equation_shape(text)
        if shape is not None:
            by_shape.setdefault(shape, []).append(number)
    return by_shape


def orderable(pool: Sequence[int]) -> int:
    """How many ordered (E, F) pairs a pool of distinct equations supplies."""
    return len(pool) * (len(pool) - 1)


def format_cell_inventory(equations: Sequence[str]) -> str:
    """The cells a law list can fill, with pool and ordered-pair counts.

    Depth is bounded by the major side, so most cells are empty. The widest
    depth spread sits on the minor = 0 rows, where one side is a bare variable
    and the other carries every operation.
    """
    header = ("minor", "major", "vars", "depth", "laws", "pairs")
    lines = [" ".join("{:>6}".format(name) for name in header)]
    by_shape = index_by_shape(equations)
    for shape in sorted(by_shape):
        pool = by_shape[shape]
        fields = shape + (len(pool), orderable(pool))
        lines.append(" ".join("{:>6}".format(value) for value in fields))
    return "\n".join(lines)


def parse_cells(text: str) -> Tuple[Shape, ...]:
    """Parse a cell spec: comma-separated MINOR:MAJOR:VARS:DEPTH items.

    The depth field may be a range, so '0:7:4:4-7' covers depths 4 through 7 of
    the seven-operation, four-variable, bare-minor-side family.
    """
    cells: List[Shape] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) != 4:
            raise SystemExit("expected MINOR:MAJOR:VARS:DEPTH, got {!r}".format(item))
        low_text, dash, high_text = fields[3].partition("-")
        try:
            minor, major, variables = (int(field) for field in fields[:3])
            low = int(low_text)
            high = int(high_text) if dash else low
        except ValueError:
            raise SystemExit("expected MINOR:MAJOR:VARS:DEPTH, got {!r}".format(item))
        if minor < 0 or major < minor or variables < 1 or low < 1 or high < low:
            raise SystemExit("empty range: {!r}".format(item))
        cells.extend(
            (minor, major, variables, depth) for depth in range(low, high + 1)
        )
    if not cells:
        raise SystemExit("no cells given")
    return tuple(dict.fromkeys(cells))


# ------------------------------------------------------------------- Sampling


def sample_depth_balanced(
    equations: Sequence[str],
    per_cell: int,
    seed: int,
    cells: Optional[Tuple[Shape, ...]] = None,
    form: str = "story",
    template_path: Optional[Path] = None,
    label_prefix: str = LABEL_PREFIX,
    allowed: Optional[Set[int]] = None,
) -> List[dict]:
    """Draw per_cell pairs from each cell, both laws taken from that cell.

    `allowed`, if given, restricts the draw to those equation numbers. It
    exists so an experiment can screen laws out before sampling rather than
    filtering afterwards — dropping pairs after the fact leaves a cell short
    and makes the sample size depend on which pairs happened to be drawn. The
    screen this was added for is excluding laws a model was fine-tuned on,
    where reading activations for a memorized law measures something other than
    what the experiment is asking about.

    Every pair therefore has ops_total = 2 * (minor + major) and exactly the
    cell's depth, and pairs within a family render to identical length. Cells
    default to every cell the list can fill, ascending; pass them explicitly to
    pin an experiment's grid. Rows carry the vendored complexity tags plus a
    "shape" field, and the same seed always yields the same pairs.

    Vacuous laws (x = x, y = x) are skipped rather than filtered afterwards, so
    a cell always comes back full. They cannot occur in a cell with a nonzero
    operation count, which every useful cell has; the check is here so that
    reusing this function on the published law list stays safe.
    """
    rng = random.Random(seed)
    by_shape = index_by_shape(equations)
    if allowed is not None:
        by_shape = {
            shape: [n for n in pool if n in allowed]
            for shape, pool in by_shape.items()
        }
    if cells is None:
        cells = tuple(
            sorted(
                shape
                for shape, pool in by_shape.items()
                if orderable(pool) >= per_cell
            )
        )
        if not cells:
            raise SystemExit(
                "no cell supports {} ordered pairs".format(per_cell)
            )

    samples: List[dict] = []
    chosen = set()
    for cell in cells:
        minor, major, variables, depth = cell
        pool = by_shape.get(cell, [])
        label = "cell minor={} major={} vars={} depth={}".format(
            minor, major, variables, depth
        )
        if orderable(pool) < per_cell:
            raise SystemExit(
                "{}: {} laws supply only {} ordered pairs, need {}".format(
                    label, len(pool), orderable(pool), per_cell
                )
            )
        got = 0
        attempts = 0
        while got < per_cell:
            attempts += 1
            if attempts > 1000 * per_cell:
                raise SystemExit("could not fill {}".format(label))
            e_num = pool[rng.randrange(len(pool))]
            f_num = pool[rng.randrange(len(pool))]
            if e_num == f_num or (e_num, f_num) in chosen:
                continue
            chosen.add((e_num, f_num))
            sample = make_sample(
                equations, e_num, f_num, form, template_path, label_prefix
            )
            if sample is None or is_vacuous(sample):
                continue
            sample.update(complexity_tags(sample["metadata"]))
            sample["shape"] = [minor, major, variables, depth]
            samples.append(sample)
            got += 1
    return samples


# ------------------------------------------------- The published ETP law list

# Bin 1 of the published list is made entirely of pairs containing a vacuous
# law, so the useful stratified range starts at 2.
ETP_BINS = tuple(range(2, 9))


def sample_etp_stratified(
    per_bin: int,
    seed: int,
    bins: Sequence[int] = ETP_BINS,
    form: str = "story",
    template_path: Optional[Path] = None,
) -> List[dict]:
    """Draw pairs from the published ETP law list, stratified by operation count.

    The counterpart of sample_depth_balanced for experiments that continue the
    earlier steering and activation work: same sampler, same seed and same law
    list those runs used, so their baselines remain the right comparison.

    Sampling never consults `form`, so one seed gives the identical pair set in
    every surface form and two arms can be compared example by example.
    """
    equations, _ = load_equations()
    samples = sample_pairs_stratified(
        equations, per_bin, seed, tuple(bins), form=form, template_path=template_path
    )
    return [sample for sample in samples if not is_vacuous(sample)]


def sample_etp_matched(
    per_bin: int,
    seed: int,
    forms: Sequence[str] = ("story", "literal"),
    bins: Sequence[int] = ETP_BINS,
) -> Dict[str, List[dict]]:
    """The same pairs rendered in each of `forms`, aligned position by position."""
    arms = {form: sample_etp_stratified(per_bin, seed, bins, form) for form in forms}
    reference = [s["pair_id"] for s in arms[forms[0]]]
    for form in forms[1:]:
        if [s["pair_id"] for s in arms[form]] != reference:
            raise SystemExit(
                "the {!r} arm drew a different pair set than {!r}; the sampler "
                "is meant to ignore the form".format(form, forms[0])
            )
    return arms


def split_alternating(samples: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    """Split into a fit half and a held-out half, taking every other pair.

    Alternating rather than cutting in the middle because the sampler emits
    pairs grouped by complexity bin, so a contiguous split would put the easy
    problems in one half and the hard ones in the other.
    """
    return list(samples[0::2]), list(samples[1::2])


def render_all_forms(metadata: dict) -> Dict[str, str]:
    """The six surface forms of one pair, keyed by form name.

    Four themed stories, the literal description, and Rigid Grammar — the terse
    formal rendering, produced here by fabricating the correct answer from the
    record's own canonical equations.
    """
    from benchmark import synthesize_response

    e_text, f_text = metadata["equation_e"], metadata["equation_f"]
    texts = {
        theme: render_story(e_text, f_text, theme_key=theme)[0]
        for theme in STORY_THEMES
    }
    texts["literal"] = render_description(e_text, f_text)[0]
    texts["rg"] = synthesize_response(metadata)
    return texts


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument(
        "--list-cells",
        action="store_true",
        help="print the cells the default pool can fill, then exit",
    )
    cli.add_argument("--per-bin", type=int, default=POOL_PER_BIN)
    cli.add_argument("--seed", type=int, default=POOL_SEED)
    args = cli.parse_args(argv)

    if not args.list_cells:
        cli.error("nothing to do; pass --list-cells")
    print(format_cell_inventory(build_pool(args.seed, POOL_BINS, args.per_bin)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
