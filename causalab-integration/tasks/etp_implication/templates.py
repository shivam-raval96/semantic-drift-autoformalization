"""
Template definitions and fill function.

The "template" causal variable IS the register (formal / instance /
paraphrase / named): one template value per surface form, identical
boilerplate across registers so that a register change alters only the
premise and conclusion spans. Cross-register interventions are therefore
template interventions, with the instruction text held fixed.
"""

from .config import LAWS, REGISTERS

TEMPLATES = list(REGISTERS)

_PROMPT = (
    "Consider a set with one binary operation. An equation that always "
    "holds is stated, then a question is asked.\n"
    "Rule: {premise}\n"
    "Does it follow that the next statement also always holds?\n"
    "Statement: {conclusion}\n"
    "Answer with True or False.\nAnswer:"
)


def surface_for(lid: str, register: str) -> str:
    law = LAWS[lid]
    if register not in law:
        raise KeyError(f"law {lid} has no register {register!r}")
    return law[register]


def fill_template(register: str, premise_lid: str, conclusion_lid: str) -> str:
    return _PROMPT.format(
        premise=surface_for(premise_lid, register),
        conclusion=surface_for(conclusion_lid, register),
    )
