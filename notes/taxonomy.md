# Failure taxonomy (draft v0) — semantic drift in formal translation

Scope: NL intent → equational law over one binary operation. "Valid" is staged
(v0 parseability, v1 Lean elaboration — see ROADMAP). Faithfulness is decided by the
certificate-level semantic diff in `pipeline/magma.py`, never by string match alone.

## Axis 1 — the 2×2 outcome (from the project page)
FV (faithful·valid) · **VBU (unfaithful·valid — target)** · FI (faithful·invalid) ·
UI (unfaithful·invalid). Standard pipelines silently pass VBU; verification catches
the Invalid column.

## Axis 2 — drift type within VBU (generator = drift move in `pipeline/laws.py`)
| Type | Definition | Semantic-diff signature | Example |
|---|---|---|---|
| **Weakening** | Constraint dropped; output implied by intent | intended → output, not conversely ("weaker, under-constrained") | comm ⇒ `x = x` |
| **Strengthening** | Over-constrained; output implies intent | output → intended, not conversely ("stronger, over-constrained") | comm ⇐ `x = y` |
| **Neighbor confusion** | A confusable nearby law substituted | typically incomparable | assoc ↔ left-permutation |
| **Variable role swap** | Same shape, variable roles permuted on one side | typically incomparable | assoc ↔ `x*(y*z)=(x*z)*y` |
| (open) **Operator drift** | Wrong operation/order (the webpage's worked example: `a·b = b·a` vs `b·a = e`) | needs multi-op signature — post-pilot | left-inverse ↔ commutes-with-some |

Direction matters: weakening vs strengthening is recoverable from the certificate
(which side has the countermodel), so the taxonomy is machine-labelable — no rater
needed for types 1-4. Hand-checking (benchmark v0) validates the NL side: does the
surface really pin down the intended law?

## Axis 3 — L-code (invalid column)
Unparsable output, wrong arity/signature, prose contamination. Tracked by Family D
controls (syntax floor with CI in the report); subtract narratively, never silently.

## Known limitations
- Symmetric laws (comm) have no distinct variable_role_swap target — the swap
  degenerates; deduped into neighbor confusion (documented in `pipeline/laws.py`).
- Faithful·Invalid is empty until validity v1 (Lean elaboration) lands: with
  parseability as validity, a parse failure can't be certified faithful.
- Single-operation signature: operator drift (the richest real-world VBU class)
  needs the signature extension; keep it in the taxonomy as the growth direction.
