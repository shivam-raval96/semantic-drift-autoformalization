# VERIFICATION.md — review guide for the 2026-07-04 work batch

Tasks 1-5 of the CLAUDE.md queue are done (17/17 tests pass, all offline). This file
orders what a human should check by (risk x leverage) so review time goes where
machines can't. Everything else is pinned by tests — re-running `python test_pilot.py`
IS the re-check.

## 1. Judgment calls that need your sign-off (~10 min total)

### a. New certified route: substitution instance (the one real convention change)
CLAUDE.md's honest-labels convention said implications are construction-known or the
pair is EXCLUDED. I added a second certified route: if `conclusion = sigma(premise)`
for a substitution sigma of terms for variables, then premise entails conclusion
(soundness: universally quantified equations entail their instances — one line of
equational logic). sigma is stored inline as the certificate and re-verified by
`test_dataset_gold_is_certified` (apply sigma, compare terms). It certifies 21
implications that were previously excluded (e.g. assoc -> flex with z:=x).
**If you reject this**, delete the `substitution_instance` branch in
`magma._known_or_instance` — everything else stands; Family I just shrinks.
Check: read `substitution_instance` + `_subst_match` in laws.py (~30 lines).

### b. The 5 new NL template sets (the actual experimental stimuli)
Machines cannot verify that English means what the law says. Read `NL_TEMPLATES` in
laws.py for **rproj, unipot, rabsorb, lselfdist, medial** (4 registers each).
Particular care: does the medial paraphrase ("exchanging the two middle elements")
really pin down (x*y)*(z*w) = (x*z)*(y*w)? Is the rabsorb paraphrase unambiguous
about associativity of reading ("(a combined with b) combined with b")?

### c. Smaller calls, listed for transparency
- `tclass` theory classes (8) are **reporting strata only** — no gold flows through them.
- Family D got whitespace-mangled "dense" variants for CORE laws to hold the ~1:4
  control ratio (34 controls / 139 experimental items). Confirm against spec §D.
- comm and the projections still dedupe variable_role_swap into neighbor_confusion —
  structural (comm is symmetric under the swap), documented in laws.py. assoc,
  labsorb, medial now have genuinely distinct varswap targets (tested).
- ApiBackend sends temperature 0 only to models that accept it; newer Claude models
  (opus-4-7/4-8, sonnet-5, fable) reject sampling params, so those default to 3 seeds
  (`--seeds` overrides). Judgment call: refusals return '' -> scored syntax_failure.
- `extract_equation` strips ONLY markdown fences/backticks — anything more would mask
  the L-code failures we're measuring.

## 2. Machine-verified — do NOT spend eyes on these
- **ETP ids** (all 20 in-range laws): matched mechanically by `verify_etp.py` against
  the ETP repo's equation files (fetched 2026-07-04), unique hit each; laws.py is
  pinned to the recorded ETP strings by an offline test. The 4 laws with n_ops > 4
  are outside ETP's enumeration (it stops at 4 op applications) -> etp=None, tested.
  Only residual trust point: that ETP_STRINGS in verify_etp.py quotes the ETP list
  faithfully — spot-check 2 entries in the equation explorer if you like.
- **Certificates**: every countermodel in the dataset re-verified by satisfies();
  every substitution certificate re-applied and compared; Z3 proposals re-checked by
  our own engine before acceptance.
- **Family A gold integrity**: every (intended, drift-target) pair is separated by a
  countermodel in >=1 direction (test) — no drift item can score 'faithful'.
- **No leakage**: prompts built only from surface/premise/conclusion; tested against
  drift laws, move names, certificates; both implication options appear symmetrically.
- **Sizes**: A=99 (spec 50-100), B=40, D=34 (~1:4), I=352 (88 implies / 264 non, 1:3).
- **Perfect-model invariant** still holds over the expanded dataset (zero drift in =>
  zero drift verdicts out).

## 3. Not verified — known gaps
- **Live API path never exercised** (no key in env). The full code path minus the HTTP
  call is covered by the fake-transport test. First real run: task 7 in CLAUDE.md.
- **Spec files unavailable**: `12_DATASET_SPEC.md` / `13_PREREGISTRATION_AND_WEEK_PLAN.md`
  are not in this tree; work followed CLAUDE.md's summary. Cross-check §1c decisions
  against the real spec when you have it. Spec wins; code updates, not spec.
- Mock rates remain fictions. report.md/comparison.md from the demo are pipeline
  validation only.

## 4. Z3 sweep findings (informational, not in the dataset)
`python z3_check.py 6` on the 53 pairs excluded at <=4:
- **5 new size-5 countermodels**: labsorb_sw -/-> {labsorb, rabsorb, comm, medial,
  medial_sw}. Available if the spec ever raises the bound; NOT added to Family I
  (spec's <=4 bound kept).
- 48 pairs remain unresolved at <=6 — many are true implications correctly refusing
  to certify (lproj -> assoc: left projection IS associative; central -> idem: central
  groupoids ARE idempotent), i.e. checkable-not-decidable behaving as intended.

## 5. Where things are
| File | What changed |
|---|---|
| pilot.py | ApiBackend (real), PROMPTS, extract_equation, seed loop, score memo, boot CIs, report.csv, `compare` cmd |
| laws.py | 24 laws w/ verified ETP ids + tclass, substitution_instance route, NEIGHBOR/VARSWAP tables, 5 new template sets |
| magma.py | substitution route in implication_status, certify_all (batched search) |
| dataset.py | CORE=10, certified_pairs cache, Family D variants, tclass fields |
| verify_etp.py | NEW — mechanical ETP id verification (network) + ETP_STRINGS record |
| z3_check.py | NEW — optional Z3 backend, mirrors find_countermodel, self-re-verifying |
| test_pilot.py | 7 -> 17 tests |

Review with `git diff` (nothing committed; tree left dirty deliberately for your review).
