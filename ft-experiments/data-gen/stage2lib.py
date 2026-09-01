#!/usr/bin/env python3
"""Shared pieces for stage 2: translation prompts + grading across the RGs.

Both train_stage2_lambda.py and eval/stage2_eval_lambda.py import this so a
story is turned into the SAME prompt at train and eval time (byte-identical),
and every grammar grades through the one seam (grammars.grade_b). Prompts are
built with the repo's own build_prompt + wrap_prompt, exactly as train_pairs
and modal_eval do. RG-1/2/3 use the frozen repo templates; RG-4 uses the
held-out template under stage2/prompts/.
"""

from __future__ import annotations

import hashlib
import sys

import stage1lib as s1
from ftlib import ftc

grammars = s1.grammars
GRAMMARS = s1.GRAMMARS
RG_LABELS = dict(s1.RG_LABELS)               # RG-N -> grammar key
GRAMMAR_TO_LABEL = dict(s1.GRAMMAR_TO_LABEL)  # grammar key -> RG-N

REPO = ftc.REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from benchmark import wrap_prompt          # noqa: E402
from checkform import build_prompt         # noqa: E402

PROMPTS = ftc.PATHS["prompts"]                    # informalizing-etp/prompts
STAGE2_PROMPTS = ftc.PATHS["stage2"] / "prompts"  # our tree (RG-4)

# grammar key -> translation prompt template
TEMPLATE_PATHS = {
    "a": PROMPTS / "formalize_prompt.md",
    "b_near": PROMPTS / "formalize_prompt_bnear.md",
    "b_far": PROMPTS / "formalize_prompt_bfar.md",
    "sexpr": STAGE2_PROMPTS / "formalize_prompt_rg4.md",
}

# frozen digests for the repo templates (config.EVAL); RG-4 is ours (unpinned)
_FROZEN_SHAS = ftc.EVAL["template_shas"]
_TEMPLATE_FILE = {
    "a": "formalize_prompt.md",
    "b_near": "formalize_prompt_bnear.md",
    "b_far": "formalize_prompt_bfar.md",
}


def key_for(label_or_key: str) -> str:
    """Accept 'RG-1'/'rg1'/'a' and return the grammar key."""
    token = label_or_key.strip()
    if token in GRAMMARS:
        return token
    up = token.upper().replace("RG", "RG-").replace("--", "-")
    if up in RG_LABELS:
        return RG_LABELS[up]
    raise KeyError(f"unknown grammar {label_or_key!r}")


def template_sha(key: str) -> str:
    return hashlib.sha256(TEMPLATE_PATHS[key].read_bytes()).hexdigest()


def assert_frozen_templates() -> None:
    """The three repo templates must match their pinned digests; RG-4 is
    ours and only recorded."""
    for key, fname in _TEMPLATE_FILE.items():
        got = template_sha(key)
        want = _FROZEN_SHAS[fname]
        assert got == want, f"frozen template changed on disk: {fname} {got}"


def build_translation_prompt(story: str, key: str, model_id: str) -> str:
    base = build_prompt({"story": story}, template_path=TEMPLATE_PATHS[key])
    return wrap_prompt(base, "off", model_id, "story")


# Bare prompt: name only the opaque label, no rules and no example. Used to
# test whether stage-1 familiarity (label -> notation, learned on statements)
# transfers to producing that notation from a story it was never trained on.
# Unwrapped on purpose: the shared "off"/"story" suffix (a two-line output
# nudge) is dropped so the model gets nothing but the label and the story.
LABEL_INSTR = "Write the following as a statement in {label}.\n\n{story}"


def build_label_prompt(story: str, key: str, model_id: str) -> str:
    return LABEL_INSTR.format(label=GRAMMAR_TO_LABEL[key], story=story)


# F0 collapses the bare label prompt onto its stage-1 validate task and just
# answers Yes/No. These two variants break that reflex without revealing the
# grammar's rules. PRODUCE says "produce, don't judge"; the prefill (below)
# seeds the reply with the grammar's own first line label so a Yes/No answer
# is impossible and the model must complete a statement.
PRODUCE_INSTR = ("Translate the following into {label}. Write the {label} "
                 "statement itself; do not answer yes or no.\n\n{story}")


def build_produce_prompt(story: str, key: str, model_id: str) -> str:
    return PRODUCE_INSTR.format(label=GRAMMAR_TO_LABEL[key], story=story)


def prefill_for(key: str) -> str:
    """The grammar's first line label as an assistant-turn seed, e.g.
    'ASSUME: ' for RG-1. Reveals only the label token (which F0 saw
    thousands of times in stage 1), not the op(a, b) notation."""
    return f"{GRAMMARS[key].labels[0]}: "


def grade(response: str, key: str, canonical_e: str, canonical_f: str) -> dict:
    return grammars.grade_b(response, key, canonical_e, canonical_f)
