#!/usr/bin/env python3
"""Rates with intervals, and paired comparisons between conditions.

Two rules from the project's conventions are enforced by these helpers rather
than left to each experiment: every rate is reported with its denominator and a
confidence interval, and conditions are compared *paired* — on the same
problems — because at the sample sizes used here (roughly 50 to 300 pairs per
cell) an unpaired comparison cannot resolve a ten-point shift that a paired one
can.

Pure standard library, so analysis runs anywhere.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# 1.959963985 is the two-sided normal quantile at 95%.
Z95 = 1.959963984540054


class Rate:
    """A proportion, its denominator, and a 95% confidence interval."""

    __slots__ = ("successes", "total", "low", "high")

    def __init__(self, successes: int, total: int):
        self.successes = successes
        self.total = total
        self.low, self.high = wilson_interval(successes, total)

    @property
    def value(self) -> float:
        return self.successes / self.total if self.total else float("nan")

    def as_dict(self) -> dict:
        return {
            "rate": self.value,
            "successes": self.successes,
            "total": self.total,
            "ci95": [self.low, self.high],
        }

    def __str__(self) -> str:
        if not self.total:
            return "n/a (0 examples)"
        return "{:.1%} ({}/{}, 95% CI {:.1%}-{:.1%})".format(
            self.value, self.successes, self.total, self.low, self.high
        )

    def __repr__(self) -> str:
        return "Rate({}, {})".format(self.successes, self.total)


def wilson_interval(successes: int, total: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the textbook normal interval because it stays inside [0, 1]
    and behaves sensibly at rates near 0 or 1, which is where accuracy sits in
    the harder cells here.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    spread /= denominator
    # The interval contains the observed proportion by construction, but at 0
    # and 1 the two terms cancel only up to rounding and the bound can land a
    # few 1e-17 the wrong side of it. Clamping to the proportion keeps the
    # distance from rate to bound non-negative, which is what error bars need.
    return (
        max(0.0, min(p, centre - spread)),
        min(1.0, max(p, centre + spread)),
    )


def rate(successes: int, total: int) -> Rate:
    return Rate(successes, total)


def rate_of(flags: Iterable[bool]) -> Rate:
    flags = list(flags)
    return Rate(sum(1 for flag in flags if flag), len(flags))


class Paired:
    """The result of comparing two conditions on the same problems.

    `gained` counts problems the second condition got right and the first got
    wrong; `lost` counts the reverse. McNemar's test uses only those two counts,
    because problems both conditions agree on carry no information about which
    is better.
    """

    __slots__ = ("n", "gained", "lost", "both", "neither", "p_value")

    def __init__(self, n: int, gained: int, lost: int, both: int, neither: int):
        self.n = n
        self.gained = gained
        self.lost = lost
        self.both = both
        self.neither = neither
        self.p_value = mcnemar_exact(gained, lost)

    @property
    def delta(self) -> float:
        """Second condition's rate minus the first's, in proportion points."""
        return (self.gained - self.lost) / self.n if self.n else float("nan")

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "gained": self.gained,
            "lost": self.lost,
            "both_correct": self.both,
            "neither_correct": self.neither,
            "delta": self.delta,
            "mcnemar_p": self.p_value,
        }

    def __str__(self) -> str:
        if not self.n:
            return "no shared problems"
        return (
            "{:+.1f} points on {} shared problems "
            "({} gained, {} lost, McNemar p = {:.3g})".format(
                100 * self.delta, self.n, self.gained, self.lost, self.p_value
            )
        )


class Unpaired:
    """A difference between two rates measured on *different* problems.

    Needed where pairing is impossible rather than merely inconvenient. Two
    depth levels, for instance, are made of different laws by construction —
    a law cannot be both depth 4 and depth 7 — so no problem is shared and
    McNemar's test does not apply. The interval is Newcombe's, built from the
    two Wilson intervals; it is wider than a paired one would be, which is the
    honest price of the design.
    """

    __slots__ = ("first", "second", "low", "high")

    def __init__(self, first: Rate, second: Rate):
        self.first = first
        self.second = second
        d = second.value - first.value
        self.low = d - math.sqrt(
            (second.value - second.low) ** 2 + (first.high - first.value) ** 2
        )
        self.high = d + math.sqrt(
            (second.high - second.value) ** 2 + (first.value - first.low) ** 2
        )

    @property
    def delta(self) -> float:
        return self.second.value - self.first.value

    @property
    def distinguishable(self) -> bool:
        """True when the 95% interval for the difference excludes zero."""
        return self.low > 0 or self.high < 0

    def as_dict(self) -> dict:
        return {
            "first": self.first.as_dict(),
            "second": self.second.as_dict(),
            "delta": self.delta,
            "ci95": [self.low, self.high],
            "excludes_zero": self.distinguishable,
        }

    def __str__(self) -> str:
        return "{:+.1f} points ({} vs {}), 95% CI {:+.1f} to {:+.1f}".format(
            100 * self.delta,
            "{}/{}".format(self.second.successes, self.second.total),
            "{}/{}".format(self.first.successes, self.first.total),
            100 * self.low,
            100 * self.high,
        )


def unpaired_difference(first: Rate, second: Rate) -> Unpaired:
    """`second` minus `first`, with an interval, for unshared problems."""
    return Unpaired(first, second)


def mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    The exact binomial form rather than the chi-square approximation, since the
    discordant count is often small enough here that the approximation is not
    trustworthy.
    """
    n = gained + lost
    if n == 0:
        return 1.0
    smaller = min(gained, lost)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def pair_records(
    baseline: Sequence[dict],
    variant: Sequence[dict],
    key: str = "pair_id",
    correct: str = "correct",
    status: str = "status",
) -> Paired:
    """Compare two sets of graded records on the problems they share.

    Records are matched by `key` rather than by position, so a condition that
    skipped or reordered problems still compares correctly. A key appearing
    more than once in either set is an error, since it would make the pairing
    ambiguous.
    """
    left = _index(baseline, key)
    right = _index(variant, key)
    shared = [k for k in left if k in right]

    gained = lost = both = neither = 0
    for k in shared:
        a = left[k].get(status) == correct
        b = right[k].get(status) == correct
        if a and b:
            both += 1
        elif not a and not b:
            neither += 1
        elif b:
            gained += 1
        else:
            lost += 1
    return Paired(len(shared), gained, lost, both, neither)


def _index(records: Sequence[dict], key: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for record in records:
        value = record[key]
        if value in out:
            raise ValueError(
                "duplicate {} {!r}: cannot pair records unambiguously".format(
                    key, value
                )
            )
        out[value] = record
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rank correlation, or None when either variable is constant.

    Used for "does accuracy fall monotonically as depth rises", where the
    relationship is expected to be ordered but not necessarily linear.
    """
    if len(xs) != len(ys):
        raise ValueError("spearman needs two equal-length sequences")
    return pearson(_ranks(xs), _ranks(ys))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Linear correlation, or None when either variable is constant."""
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    var_x, var_y = sum(d * d for d in dx), sum(d * d for d in dy)
    if var_x <= 0 or var_y <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(var_x * var_y)


def _ranks(values: Sequence[float]) -> List[float]:
    """Ranks, with ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
