#!/usr/bin/env python3
"""Answer-extraction contract for the verification gate."""

import modal_verify
from modal_verify import extract

# The module is imported inside the container too: config loading must stay
# lazy (a module-level pxc broke container startup once; keep it that way).
assert not hasattr(modal_verify, "pxc"), "config must not load at module level"

CASES = [
    ("Yes", "yes"),
    (" yes.", "yes"),
    ("NO\n", "no"),
    ("Yes, the formalization is correct.", "yes"),
    ("The answer is no.", "no"),
    ("no, the order of inputs is swapped", "no"),
    ("I cannot tell.", None),
    ("", None),
    ("yesterday", None),          # must not match inside a word
    ("Notably wrong", None),
]

for text, want in CASES:
    got = extract(text)
    assert got == want, f"extract({text!r}) = {got!r}, want {want!r}"
print(f"extract: {len(CASES)} cases pass")
