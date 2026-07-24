"""
Causal model for the ETP implication-judgment task.

DAG: (premise_law, conclusion_law) -> implication -> raw_output
     (template, premise_law, conclusion_law) -> raw_input

The implication mechanism is a lookup into the certified table: every
(premise, conclusion) value the counterfactual generators emit carries a
Lean-verified-adjacent certificate (construction-known / substitution
instance for True, finite countermodel for False). Uncertified pairs are
not in the table and raise KeyError by design — sampling must never
reach them ("certify or exclude").
"""

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var

from .config import LAW_IDS, PAIRS, TASK_NAME, law_fingerprint_coords, pair_key
from .templates import TEMPLATES, fill_template

values = {
    "template": TEMPLATES,
    "premise_law": LAW_IDS,
    "conclusion_law": LAW_IDS,
    "implication": [True, False],
    "raw_input": None,
    "raw_output": None,
}

mechanisms = {
    "template": input_var(TEMPLATES),
    "premise_law": input_var(LAW_IDS),
    "conclusion_law": input_var(LAW_IDS),
    "implication": Mechanism(
        parents=["premise_law", "conclusion_law"],
        compute=lambda t: PAIRS[pair_key(t["premise_law"], t["conclusion_law"])],
    ),
    "raw_input": Mechanism(
        parents=["template", "premise_law", "conclusion_law"],
        compute=lambda t: fill_template(
            t["template"], t["premise_law"], t["conclusion_law"]
        ),
    ),
    "raw_output": Mechanism(
        parents=["implication"],
        compute=lambda t: "True" if t["implication"] else "False",
    ),
}

_ANSWER_FORMS: dict[object, list[str]] = {
    True: [" True", "True"],
    False: [" False", "False"],
}

CAUSAL_MODEL = CausalModel(
    mechanisms,
    values,
    id=TASK_NAME,
    output_tokens={"implication": dict(_ANSWER_FORMS)},
    match_modes={"implication": "prefix"},
)

TARGET_VARIABLE = "implication"

# Coordinate embeddings for manifold analyses: laws embed at their certified
# implication-fingerprint PCA coordinates (3D, ~61% of table variance), so a
# manifold fit over law identity is parameterized by Lean's own metric — a
# good fit IS the fingerprint-isometry result. The verdict embeds as +/-1.
EMBEDDINGS = {
    "premise_law": law_fingerprint_coords,
    "conclusion_law": law_fingerprint_coords,
    "implication": lambda v: [1.0 if v else -1.0],
}
