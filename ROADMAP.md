# ROADMAP — from the project page's plan to running code

The project page (index.html) commits to: a benchmark of translation attempts across
{faithful, not-faithful} × {valid, not-valid} focused on VBU; lightweight detectors;
mechanistic analyses on matched pairs; MVP by end of the in-person week (July),
publication target October.

## Status (updated 2026-07-07)
Mock-era artifacts removed. Current state lives in pipeline/ (confirmatory tier:
code, specs, cards) and the lab tree's ledger/findings. Highlights: certified
failure-migration with CIs; drift monitor gauntlet-hardened; steering causal w/
random control; judge-vs-certificate decomposition; Gao BoN reproduction; MAD v0
w/ frozen spec; CSE-validated ambiguity ladder; three-way verdict rescore in
progress. See pipeline/CLAUDE.md-equivalents: mad-spec.md, cards/, DEFENSE plan.

<details><summary>Original 2026-07-04 status (historical)</summary>

## Status (2026-07-04)

### Prework deliverables — DONE
- **Minimal pipeline / toy end-to-end example** → `pipeline/` (imported from the
  semdiff_pilot working tree, tasks 1-5 of its queue complete; 17 tests, offline).
  NL surface → model translation → parse → certificate-level semantic diff
  (countermodel or substitution-instance certificates, direction of drift) →
  stratified report with bootstrap CIs.
- **Failure taxonomy (draft)** → `notes/taxonomy.md`. Drift moves double as
  taxonomy classes and as benchmark generators.
- **Ground truth** → 24 laws, ETP node ids mechanically verified
  (`pipeline/verify_etp.py`); gold only from certified routes.

### In-person week targets — scaffolded, ready to fill with real-model data
- **Benchmark v0** (~100s hand-checked / ~1000s generated) → `benchmark/make_benchmark.py`
  emits quadrant-labeled items from any pipeline results file. Mock data flows today;
  swap in API results when keys are available. Hand-check queue = the `A-*` items.
- **First detector baselines** → `detectors/baselines.py`: self-consistency across
  seeds + output-space features, AUROC vs certificate gold. Runs offline on mock
  multi-seed data now; identical interface for real runs.
- **Error analysis** → report.md drift-direction table + per-move/per-class CIs.

</details>

## Mapping the paper's 2×2 onto pipeline verdicts
| Quadrant | Pipeline signature |
|---|---|
| Faithful · Valid (correct) | parses AND verdict `faithful` |
| Unfaithful · Valid (**VBU**, target) | parses AND verdict `drift: *` |
| · · Invalid (either row) | `syntax_failure (L)` — L-code; "faithful·invalid" needs a stronger validity notion than parseability (see below) |

**Validity axis, staged:** v0 = parses as a single equation over `*` (what the
pipeline checks). v1 = Lean elaboration against the ETP formalization (each library
law carries its verified ETP node id, so wiring to Lean/`equational_theories` is a
lookup, not a research task). Until v1, invalid cells are under-split — documented,
not hidden.

## Next (remote phase, Aug–Sep)
1. Real-model benchmark runs (>=2 models × >=3 seeds where sampling is on);
   `pipeline/pilot.py compare`.
2. Detector families: add confidence/logprob signals (OpenAI path exposes logprobs;
   Anthropic path: self-consistency + round-trip similarity), back-translation
   semantic similarity, then hidden-state probes on an open-weight model
   (pipeline task 6 — AFTER template split is wired).
3. Matched-pair mechanistic study on faithful/drifted pairs (Family A is built as
   matched pairs for exactly this).
4. Cost–performance analysis: detector AUROC vs cost of full verification.
5. Scale the library toward ~1000s of generated items: more laws (ETP has 4694
   equations; the matcher in `verify_etp.py` verifies new ids mechanically),
   template banks per register, harder implication items (Z3 extends certificate
   reach past size 4 — 5 new size-5 certificates already found).

## Housekeeping
- index.html stays at repo root (GitHub Pages serves it as-is).
- `pipeline/` provenance: copied 2026-07-04 from the local `semdiff_pilot` working
  tree (its own git history lives there). Treat this copy as canonical going forward
  or re-sync deliberately — don't edit both.
- Nothing here has been pushed; review locally first.
