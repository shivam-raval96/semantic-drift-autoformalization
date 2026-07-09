"""Confirmatory template bank v2 — implements handcheck-2026-07-05.md sign-offs.

v1 strings (laws.NL_TEMPLATES) remain frozen and untouched: exploratory record.
This bank is the CONFIRMATORY instrument. Every (law, register) carries a status:
  ok                   — v1 string approved as-is
  rewritten-v2         — approved rewrite (rabsorb/medial paraphrases)
  obscure-name-stratum — kept but NEVER pooled into faithful-rate denominators
  dropped              — excluded from confirmatory data entirely
The new distractor-adjacent register is WELL-POSED by design rule: the surface
correctly describes the intended law while borrowing the neighbor's vocabulary
(gold = intended). It measures careful reading vs lexeme pattern-matching.

Hash of this bank is pinned by test_prompt_freeze_v2 (conscious-change protocol).
"""
from laws import NL_TEMPLATES as V1

REGISTER_STATUS = {  # (lid, register) -> status; default 'ok'
    ('unipot', 'canonical'): 'obscure-name-stratum',
    ('labsorb', 'canonical'): 'dropped',
    ('rabsorb', 'canonical'): 'dropped',
    ('rabsorb', 'paraphrase'): 'rewritten-v2',
    ('medial', 'paraphrase'): 'rewritten-v2',
}

_REWRITES = {
    ('rabsorb', 'paraphrase'):
        "take a thing and combine it with a second thing; then combine that result "
        "with the same second thing again: you get the original thing back",
    ('medial', 'paraphrase'):
        "combine one pair, separately combine a second pair, then combine the two "
        "results; swapping the second element of the first pair with the first "
        "element of the second pair does not change the outcome",
}

# New distractor-adjacent register (approved, all 10): correctly describes the
# intended law in the neighbor's vocabulary.
DISTRACTOR_V2 = {
    'comm':      "the law where the two elements being combined can trade places — "
                 "whatever grouping surrounds them — without changing the result",
    'assoc':     "the law where the order in which you perform the combinations does "
                 "not matter, so long as the left-to-right order of the elements stays fixed",
    'idem':      "the law where an element combined with itself is absorbed back into "
                 "that same element",
    'unipot':    "the law where every element, combined with itself, collapses to one "
                 "single value — not necessarily the element you started with",
    'lproj':     "the law where, of the two elements combined, it is always the earlier "
                 "one — never the later — that comes back",
    'rproj':     "the law where, of the two elements combined, it is always the later "
                 "one — never the earlier — that comes back",
    'labsorb':   "the law where a thing, combined with the combination of that same "
                 "thing with another, cancels itself out and returns the other",
    'rabsorb':   "the law where combining with the same second element twice in a row "
                 "undoes itself, returning the first element",
    'lselfdist': "the law where a combination with a combined pair can be regrouped — "
                 "provided the outer element is copied into both parts",
    'medial':    "the law where combining two combinations distributes their four "
                 "elements into new pairs — firsts together, seconds together — "
                 "without changing the result",
}

def bank():
    """{lid: {register: (surface, status)}} — confirmatory bank, dropped excluded."""
    out = {}
    for lid, regs in V1.items():
        out[lid] = {}
        for reg, v1_surface in regs.items():
            status = REGISTER_STATUS.get((lid, reg), 'ok')
            if reg == 'distractor':
                out[lid][reg] = (DISTRACTOR_V2[lid], 'rewritten-v2')
                continue
            if status == 'dropped':
                continue
            surface = _REWRITES.get((lid, reg), v1_surface)
            out[lid][reg] = (surface, status)
    return out
