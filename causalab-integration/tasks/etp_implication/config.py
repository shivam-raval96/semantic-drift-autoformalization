"""
Task-specific configuration and constants.

Data provenance: data/etp_pairs.json is generated from the
semantic-drift-autoformalization repo (branch certificate-pipeline,
pipeline/laws.py + pipeline/magma.py + the compositional renderer from
etp_items.py). Labels come only from certified routes: construction-known
or substitution-instance implications (True), finite countermodels (False).
Uncertified pairs, degenerate (vacuous) laws, and self-pairs are excluded.
Regenerate with scratchpad script export_task_data.py against that branch;
the JSON embeds its own provenance string.
"""

import json
from pathlib import Path

TASK_NAME = "etp_implication"

_DATA = json.loads((Path(__file__).parent / "data" / "etp_pairs.json").read_text())

LAWS: dict = _DATA["laws"]
LAW_IDS: list = sorted(LAWS.keys())

# "premise|conclusion" -> bool (certified implication truth)
PAIRS: dict = {k: v for k, v in _DATA["pairs"].items()}

# Registers: formal and instance exist for every law; paraphrase and named
# only for the hand-templated subset (the "named" register names the law and
# carries a memorization/fame confound — keep it out of default sampling).
REGISTERS = ["formal", "instance", "paraphrase", "named"]
DEFAULT_REGISTERS = ["formal", "instance"]


def registers_for(lid: str) -> list:
    return [r for r in REGISTERS if r in LAWS[lid]]


def pair_key(premise: str, conclusion: str) -> str:
    return f"{premise}|{conclusion}"


CERTIFIED_TRUE = [tuple(k.split("|")) for k, v in PAIRS.items() if v]
CERTIFIED_FALSE = [tuple(k.split("|")) for k, v in PAIRS.items() if not v]

MAX_TASK_TOKENS = 512
MAX_NEW_TOKENS = 1
