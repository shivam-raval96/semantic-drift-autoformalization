"""
ETP implication-judgment task for causal abstraction experiments.

A premise law and a conclusion law over a magma are rendered in one of
several registers (formal equation, explicit instance prose, paraphrase,
or named); the model answers True/False whether the premise implies the
conclusion. Ground truth is a certified lookup (construction-known or
substitution-instance implications; finite countermodels for
non-implications) — never a judgment.
"""

from .causal_models import CAUSAL_MODEL, TARGET_VARIABLE
from .counterfactuals import COUNTERFACTUAL_GENERATORS

__all__ = ["CAUSAL_MODEL", "COUNTERFACTUAL_GENERATORS", "TARGET_VARIABLE"]
