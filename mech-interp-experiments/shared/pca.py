#!/usr/bin/env python3
"""Principal components, with fitting and projecting kept apart.

Principal component analysis finds the directions along which a set of vectors
varies most, so a very high-dimensional cloud can be drawn in two dimensions.
It is two separable operations, and every earlier copy in this project fused
them into one call that returns coordinates:

    mech-interp-experiments/2026-08-15/pca-probe.py
    mech-interp-experiments/2026-08-15/law-representation-after-reasoning.py
    mech-interp-experiments/2026-08-15/depth-in-activations.py

Fusing them is fine when one cloud is fitted and drawn once. It makes two
things impossible, and this project needs both:

- **Comparing across models.** Components carry an arbitrary sign, and two of
  them explaining similar amounts of variance can rotate freely into each
  other. A basis refitted per model produces pictures that differ for reasons
  having nothing to do with the models. Fitting once and reusing the basis is
  the only way the difference between two pictures means anything.
- **Showing a group without letting it choose the axes.** Fitting maximizes
  the variance of whatever it is given, so an outlying group captures the
  leading component and spends it on a distinction that was already known. Fit
  without that group, then project it in: it appears in the picture, but had no
  vote in where the axes point.

So `fit` returns a `Basis` — a centre and a set of axes — and `Basis.project`
maps any vectors at all into it, whether or not they were in the fit. The
centre matters as much as the axes: projecting a group after re-centring it on
its own mean silently moves it to the origin and destroys the comparison, so
`project` always subtracts the basis's own centre.

Everything here is numpy, so analysis runs with no GPU and no PyTorch. Torch
tensors on the CPU convert without a copy through `np.asarray`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


def _as_matrix(vectors) -> np.ndarray:
    """An (n, d) float array from tensors, arrays, or sequences of vectors."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            "expected an (n, d) block of vectors, got shape {}".format(matrix.shape)
        )
    return matrix


@dataclass(frozen=True)
class Basis:
    """A frozen frame: where the origin is, and which way the axes point.

    `explained` describes the set the basis was fitted on and nothing else. For
    anything projected in afterwards, ask `residual_share`, which is computed
    against that group's own spread.
    """

    center: np.ndarray      # (d,) the fit set's mean
    axes: np.ndarray        # (k, d) orthonormal rows, most variance first
    explained: np.ndarray   # (k,) share of the fit set's variance, each axis
    n_fit: int              # how many vectors were fitted
    labels_fit: Optional[Sequence[str]] = None  # what the fit set was, for the caption

    @property
    def k(self) -> int:
        return int(self.axes.shape[0])

    def project(self, vectors) -> np.ndarray:
        """Coordinates in this frame, for vectors fitted or not.

        Subtracts the basis's own centre, never the incoming group's mean.
        """
        centred = _as_matrix(vectors) - self.center
        return centred @ self.axes.T

    def residual_share(self, vectors) -> float:
        """The share of these vectors' spread that the axes do not capture, 0 to 1.

        This is the honest companion to any picture drawn in a frozen frame. If
        a group has moved into directions the frame was not built from, its
        coordinates still plot, but the plot is no longer where most of its
        variation lives — and a figure that does not say so is misleading.
        Spread is measured about the basis's centre, so a group that has simply
        translated away counts as displaced rather than as unexplained.
        """
        centred = _as_matrix(vectors) - self.center
        total = float((centred ** 2).sum())
        if total == 0.0:
            return float("nan")
        captured = float(((centred @ self.axes.T) ** 2).sum())
        return 1.0 - captured / total

    def to_json(self) -> dict:
        """Everything needed to rebuild this basis in another process."""
        return {
            "center": self.center.tolist(),
            "axes": self.axes.tolist(),
            "explained": self.explained.tolist(),
            "n_fit": self.n_fit,
            "labels_fit": list(self.labels_fit) if self.labels_fit else None,
        }

    @classmethod
    def from_json(cls, blob: dict) -> "Basis":
        return cls(
            center=np.asarray(blob["center"], dtype=np.float64),
            axes=np.asarray(blob["axes"], dtype=np.float64),
            explained=np.asarray(blob["explained"], dtype=np.float64),
            n_fit=int(blob["n_fit"]),
            labels_fit=blob.get("labels_fit"),
        )


def fit(vectors, k: int = 2, labels: Optional[Sequence[str]] = None) -> Basis:
    """Find the k directions of greatest variance in these vectors.

    `labels` is carried through only so a figure can say what the axes were
    built from, which is the first thing a reader of a frozen-frame plot needs
    to know.
    """
    matrix = _as_matrix(vectors)
    if matrix.shape[0] < 2:
        raise ValueError("need at least two vectors to fit a basis")
    k = min(k, *matrix.shape)

    center = matrix.mean(axis=0)
    centred = matrix - center
    _, singular, directions = np.linalg.svd(centred, full_matrices=False)
    variance = singular ** 2
    total = variance.sum()
    explained = variance[:k] / total if total > 0 else np.zeros(k)

    return Basis(
        center=center,
        axes=directions[:k],
        explained=np.asarray(explained, dtype=np.float64),
        n_fit=int(matrix.shape[0]),
        labels_fit=list(labels) if labels is not None else None,
    )


def center_within_groups(vectors, groups: Sequence) -> np.ndarray:
    """Subtract each group's own mean, removing the between-group difference.

    Used when the between-group split is the largest thing in the data but is
    not what is being asked about. Subtracting per-group means leaves only the
    variation *within* groups, so a basis fitted afterwards describes that
    instead of rediscovering the split. Here that is how the law geometry is
    reached: the difference between surface forms is far larger than the
    difference between laws, so without this step the components would only
    ever separate the forms.
    """
    matrix = _as_matrix(vectors)
    groups = list(groups)
    if len(groups) != matrix.shape[0]:
        raise ValueError(
            "got {} vectors but {} group labels".format(matrix.shape[0], len(groups))
        )
    out = matrix.copy()
    for group in dict.fromkeys(groups):
        rows = np.array([i for i, g in enumerate(groups) if g == group])
        out[rows] -= matrix[rows].mean(axis=0)
    return out


def centroids(coordinates, groups: Sequence) -> Dict[object, np.ndarray]:
    """Each group's mean position, in whatever frame the coordinates are in."""
    matrix = _as_matrix(coordinates)
    groups = list(groups)
    if len(groups) != matrix.shape[0]:
        raise ValueError(
            "got {} points but {} group labels".format(matrix.shape[0], len(groups))
        )
    return {
        group: matrix[[i for i, g in enumerate(groups) if g == group]].mean(axis=0)
        for group in dict.fromkeys(groups)
    }


def spread_ratio(vectors, groups: Sequence) -> float:
    """Mean within-group spread divided by overall spread.

    One number for "do things that should go together, go together". Near 0
    means each group is tight relative to how far apart the groups are; near 1
    means the grouping accounts for nothing. Distances are root-mean-square
    about the relevant mean, so the ratio is unitless and comparable across
    layers and across models with different activation scales.
    """
    matrix = _as_matrix(vectors)
    groups = list(groups)
    overall = float(np.sqrt((((matrix - matrix.mean(axis=0)) ** 2).sum(axis=1)).mean()))
    if overall == 0.0:
        return float("nan")

    within = []
    for group in dict.fromkeys(groups):
        rows = np.array([i for i, g in enumerate(groups) if g == group])
        if rows.size < 2:
            continue
        block = matrix[rows]
        within.append(float(np.sqrt((((block - block.mean(axis=0)) ** 2).sum(axis=1)).mean())))
    if not within:
        return float("nan")
    return float(np.mean(within) / overall)
