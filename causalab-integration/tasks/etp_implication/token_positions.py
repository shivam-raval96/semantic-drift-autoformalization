"""
Token positions: the premise span, the conclusion span, and the last token.

Spans are located by substring search of the register surface inside
raw_input. The premise surface appears before the conclusion surface by
template construction; conclusion is searched after the premise match so
identical-substring collisions cannot mislocate it.
"""

from typing import Dict

from causalab.neural.pipeline import LMPipeline
from causalab.neural.token_positions import (
    TokenPosition,
    get_last_token_index,
    get_tokens_in_char_range,
    rebase_char_range,
)

from .templates import surface_for


def _span_token_indices(input_sample, pipeline, which: str):
    raw_input = input_sample["raw_input"]
    register = input_sample["template"]
    premise_surface = surface_for(input_sample["premise_law"], register)
    conclusion_surface = surface_for(input_sample["conclusion_law"], register)

    p_start = raw_input.find(premise_surface)
    if p_start < 0:
        raise ValueError("premise surface not found in raw_input")
    if which == "premise":
        char_start, char_end = p_start, p_start + len(premise_surface)
        expected = premise_surface
    else:
        c_start = raw_input.find(conclusion_surface, p_start + len(premise_surface))
        if c_start < 0:
            raise ValueError("conclusion surface not found in raw_input")
        char_start, char_end = c_start, c_start + len(conclusion_surface)
        expected = conclusion_surface

    tokenized = pipeline.load([input_sample], return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"][0]
    char_start, char_end = rebase_char_range(
        tokenized, char_start, char_end, expected, f"{which} span"
    )
    return get_tokens_in_char_range(offsets, char_start, char_end)


def create_token_positions(
    pipeline: LMPipeline, template: str | None = None
) -> Dict[str, TokenPosition]:
    return {
        "last": TokenPosition(
            lambda x, p=pipeline: get_last_token_index(x, p), pipeline, id="last"
        ),
        "premise_law": TokenPosition(
            lambda x, p=pipeline: _span_token_indices(x, p, "premise"),
            pipeline,
            id="premise_law",
        ),
        "conclusion_law": TokenPosition(
            lambda x, p=pipeline: _span_token_indices(x, p, "conclusion"),
            pipeline,
            id="conclusion_law",
        ),
    }
