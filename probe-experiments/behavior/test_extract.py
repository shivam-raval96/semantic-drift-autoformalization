#!/usr/bin/env python3
"""Answer-extraction contract for the verification gate."""

from modal_verify import extract

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
