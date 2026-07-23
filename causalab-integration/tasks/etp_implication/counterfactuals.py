"""
Counterfactual generators for the ETP implication task.

Every sampled pair is certified, and label balance is enforced at
sampling time (the raw table is ~7 True / ~362 False — the certified
implications among non-degenerate laws are few, so balanced sampling
reuses the True pairs across registers rather than letting the base
rate leak into the dataset).

Generators map to the project's experiments:
- random_counterfactual: baseline / locate.
- flip_premise / flip_conclusion: swap one law so the certified answer
  flips — the causal test that a located representation carries law
  identity into the implication computation.
- cross_register: same law pair, different register — interchange on
  the premise span tests representation invariance across surface
  forms (the story/literal/formal question in miniature).
"""

import random

from causalab.causal.counterfactual_dataset import CounterfactualExample

from .causal_models import CAUSAL_MODEL
from .config import (
    CERTIFIED_FALSE,
    CERTIFIED_TRUE,
    DEFAULT_REGISTERS,
    OPS_BINS,
    PAIRS,
    ops_bin,
    pair_key,
    registers_for,
)

# conclusion -> premises with a certified True / False pair, for flip sampling
_BY_CONCLUSION: dict = {}
for _p, _c in CERTIFIED_TRUE:
    _BY_CONCLUSION.setdefault(_c, {True: [], False: []})[True].append(_p)
for _p, _c in CERTIFIED_FALSE:
    _BY_CONCLUSION.setdefault(_c, {True: [], False: []})[False].append(_p)
_FLIPPABLE_CONCLUSIONS = [
    c for c, d in _BY_CONCLUSION.items() if d[True] and d[False]
]

_BY_PREMISE: dict = {}
for _p, _c in CERTIFIED_TRUE:
    _BY_PREMISE.setdefault(_p, {True: [], False: []})[True].append(_c)
for _p, _c in CERTIFIED_FALSE:
    _BY_PREMISE.setdefault(_p, {True: [], False: []})[False].append(_c)
_FLIPPABLE_PREMISES = [p for p, d in _BY_PREMISE.items() if d[True] and d[False]]


def _common_registers(p, c, registers=None):
    allowed = registers or DEFAULT_REGISTERS
    return [r for r in allowed if r in registers_for(p) and r in registers_for(c)]


def _trace(p, c, register):
    return CAUSAL_MODEL.new_trace(
        {"template": register, "premise_law": p, "conclusion_law": c}
    )


def sample_balanced_input(registers=None):
    """Sample a certified pair, True/False balanced, in a shared register."""
    label = random.random() < 0.5
    pool = CERTIFIED_TRUE if label else CERTIFIED_FALSE
    while True:
        p, c = random.choice(pool)
        regs = _common_registers(p, c, registers)
        if regs:
            return _trace(p, c, random.choice(regs))


def random_counterfactual():
    return CounterfactualExample(
        input=sample_balanced_input(),
        counterfactual_inputs=[sample_balanced_input()],
    )


def flip_premise_counterfactual():
    """Hold conclusion and register; swap premise so the certified label flips."""
    while True:
        c = random.choice(_FLIPPABLE_CONCLUSIONS)
        label = random.random() < 0.5
        p = random.choice(_BY_CONCLUSION[c][label])
        p2 = random.choice(_BY_CONCLUSION[c][not label])
        regs = [
            r
            for r in _common_registers(p, c)
            if r in _common_registers(p2, c)
        ]
        if regs:
            reg = random.choice(regs)
            return CounterfactualExample(
                input=_trace(p, c, reg), counterfactual_inputs=[_trace(p2, c, reg)]
            )


def flip_conclusion_counterfactual():
    """Hold premise and register; swap conclusion so the certified label flips."""
    while True:
        p = random.choice(_FLIPPABLE_PREMISES)
        label = random.random() < 0.5
        c = random.choice(_BY_PREMISE[p][label])
        c2 = random.choice(_BY_PREMISE[p][not label])
        regs = [
            r
            for r in _common_registers(p, c)
            if r in _common_registers(p, c2)
        ]
        if regs:
            reg = random.choice(regs)
            return CounterfactualExample(
                input=_trace(p, c, reg), counterfactual_inputs=[_trace(p, c2, reg)]
            )


def cross_register_counterfactual():
    """Same certified pair, different register: label identical by construction."""
    while True:
        label = random.random() < 0.5
        pool = CERTIFIED_TRUE if label else CERTIFIED_FALSE
        p, c = random.choice(pool)
        regs = _common_registers(p, c, registers=None)
        both = [r for r in regs]
        if len(both) >= 2:
            r1, r2 = random.sample(both, 2)
            return CounterfactualExample(
                input=_trace(p, c, r1), counterfactual_inputs=[_trace(p, c, r2)]
            )


COUNTERFACTUAL_GENERATORS = {
    "random_counterfactual": random_counterfactual,
    "flip_premise": flip_premise_counterfactual,
    "flip_conclusion": flip_conclusion_counterfactual,
    "cross_register": cross_register_counterfactual,
}


def _unique_certified_items(registers=None):
    """All unique (premise, conclusion, register, label) combinations."""
    items = []
    for label, pool in ((True, CERTIFIED_TRUE), (False, CERTIFIED_FALSE)):
        for p, c in pool:
            for r in _common_registers(p, c, registers):
                items.append((p, c, r, label))
    return items


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Exactly balanced True/False, sampled WITHOUT replacement over unique
    prompts, so the set that survives causalab's dedup is still balanced.
    Stratified by complexity bin (OPS_BINS over the pair's total op count)
    and exactly label-balanced within every bin; per-cell capacity caps the
    size, so balance and stratification win over requested n.
    Counterfactuals rotate within the same balanced set, so they are
    balanced and duplicate-free too."""
    rng = random.Random(seed)
    items = _unique_certified_items()
    # stratify by complexity bin (total ops of the pair), balanced within
    # each bin: per (bin, label) k = min(n / (2*bins), available). The
    # returned set is exactly label-balanced within every complexity bin.
    groups: dict = {}
    for it in items:
        groups.setdefault((ops_bin(it[0], it[1]), it[3]), []).append(it)
    bins = sorted({b for b, _ in groups})
    per_cell = max(1, n // (2 * len(bins)))
    chosen = []
    for b in bins:
        t = groups.get((b, True), [])
        f = groups.get((b, False), [])
        k = min(per_cell, len(t), len(f))
        chosen += rng.sample(t, k) + rng.sample(f, k)
    rng.shuffle(chosen)
    examples: list[CounterfactualExample] = []
    m = len(chosen)
    for idx, (p, c, r, label) in enumerate(chosen):
        p2, c2, r2, _ = chosen[(idx + m // 2) % m]  # opposite-half partner
        examples.append(
            {
                "input": _trace(p, c, r),
                "counterfactual_inputs": [_trace(p2, c2, r2)],
            }
        )
    return examples
