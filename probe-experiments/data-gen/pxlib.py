#!/usr/bin/env python3
"""Shared helpers for probe-experiments plus the AST perturbation engine.

Serialization, canonicalization, hashing, and grading are NOT reimplemented:
ft-experiments' ftlib is loaded by path (which in turn defers to the repo's
storyform/checkform), so contrast_v1 rows hash and grade exactly as every
other artifact in this repo does.

The perturbation engine produces wrong answers as single surgical AST edits.
An edit is only accepted if checkform grades the result "wrong" — edits that
land inside the grader's accepted symmetry orbit (renaming, side swap,
uniform dualization) are counted and retried, never emitted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

STAGE = Path(__file__).resolve().parent
PX_ROOT = STAGE.parents[0]
REPO_ROOT = STAGE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ftlib = _load("px_ftlib", REPO_ROOT / "ft-experiments" / "data-gen" / "ftlib.py")
pxc = _load("px_root_config", PX_ROOT / "config.py")

Term = ftlib.Term
Var = ftlib.Var
Op = ftlib.Op
Equation = Tuple[Term, Term]

# ------------------------------------------------------- node addressing


def _paths(term: Term, prefix: tuple = ()) -> Iterator[Tuple[tuple, Term]]:
    yield prefix, term
    if isinstance(term, Op):
        yield from _paths(term.left, prefix + ("L",))
        yield from _paths(term.right, prefix + ("R",))


def _replace(term: Term, path: tuple, new: Term) -> Term:
    if not path:
        return new
    if path[0] == "L":
        return Op(_replace(term.left, path[1:], new), term.right)
    return Op(term.left, _replace(term.right, path[1:], new))


def eq_sites(eq: Equation) -> List[Tuple[str, tuple, Term]]:
    return [("lhs", p, n) for p, n in _paths(eq[0])] + [
        ("rhs", p, n) for p, n in _paths(eq[1])
    ]


def eq_replace(eq: Equation, side: str, path: tuple, new: Term) -> Equation:
    if side == "lhs":
        return (_replace(eq[0], path, new), eq[1])
    return (eq[0], _replace(eq[1], path, new))


# ---------------------------------------------------------- perturbations
# Each returns a perturbed equation or None when inapplicable. Callers must
# still gate on checkform's verdict; None only means "no site exists".


def perturb_arg_swap(rng, eq: Equation, max_vars: int) -> Optional[Equation]:
    """Swap the two children of one op node (skips symmetric nodes)."""
    sites = [
        (s, p, n) for s, p, n in eq_sites(eq) if isinstance(n, Op) and n.left != n.right
    ]
    if not sites:
        return None
    side, path, node = sites[rng.randrange(len(sites))]
    return eq_replace(eq, side, path, Op(node.right, node.left))


def perturb_var_sub(rng, eq: Equation, max_vars: int) -> Optional[Equation]:
    """Rewrite ONE variable occurrence to a different letter."""
    names = ftlib.variables_in_order(*eq)
    sites = [(s, p, n) for s, p, n in eq_sites(eq) if isinstance(n, Var)]
    if not sites:
        return None
    side, path, node = sites[rng.randrange(len(sites))]
    choices = [v for v in names if v != node.name]
    if len(names) < max_vars:
        fresh = next((l for l in ftlib.RG_LETTERS if l not in names), None)
        if fresh:
            choices.append(fresh)
    if not choices:
        return None
    return eq_replace(eq, side, path, Var(choices[rng.randrange(len(choices))]))


def perturb_prune(rng, eq: Equation, max_vars: int) -> Optional[Equation]:
    """Replace one op node by one of its children (drops a subtree).
    Never returns a zero-op equation: that would be a surface tell."""
    sites = [(s, p, n) for s, p, n in eq_sites(eq) if isinstance(n, Op)]
    rng.shuffle(sites)
    for side, path, node in sites:
        child = node.left if rng.random() < 0.5 else node.right
        candidate = eq_replace(eq, side, path, child)
        if ftlib.equation_ops(candidate) >= 1:
            return candidate
    return None


def perturb_grow(rng, eq: Equation, max_vars: int) -> Optional[Equation]:
    """Replace one variable occurrence v by op(v, w) or op(w, v)."""
    names = ftlib.variables_in_order(*eq)
    sites = [(s, p, n) for s, p, n in eq_sites(eq) if isinstance(n, Var)]
    if not sites:
        return None
    side, path, node = sites[rng.randrange(len(sites))]
    other = Var(names[rng.randrange(len(names))])
    new = Op(node, other) if rng.random() < 0.5 else Op(other, node)
    return eq_replace(eq, side, path, new)


PERTURBATIONS = {
    "arg_swap": perturb_arg_swap,
    "var_sub": perturb_var_sub,
    "prune": perturb_prune,
    "grow": perturb_grow,
}


# ------------------------------------------------------------- union-find
# Law connected components: pairs sharing any law class land in the same
# component, giving fully law-disjoint CV groups for the probe stage.


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a: str) -> str:
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
