# Harsh — The SQL register: executable semantics for magma laws

## Question

Can we add a register whose faithfulness is checked by *execution* rather
than parsing — and does model translation quality transfer to a register
with genuinely different semantics (relational queries over finite tables)?

## The idea (Luiza's proposal, yours to own)

A magma law over a finite magma is a statement about its operation table.
Render the table as a SQL relation `op(a, b, result)` and a law as a query
that returns the set of counterexample rows:

```sql
-- x * (y * y) = x   over table op(a,b,r)
SELECT t1.a AS x, t2.a AS y
FROM op t2 JOIN op t1 ON t1.b = t2.r
WHERE t2.a = t2.b AND t1.r <> t1.a;
```

A law holds on the magma iff the query returns the empty set. Two
translations are equivalent iff their result sets are equal on every test
table — **set-equivalence grading by execution**, no parser leniency
questions at all (experiment 12 quantified those; here they vanish).

The certified countermodels from the certificate pipeline are exactly the
test tables: for a certified-False pair, the countermodel table must make
the premise query empty and the conclusion query non-empty. That is a
mechanical end-to-end check of a model's SQL translation with decidable
ground truth.

## What exists to build on

- Certified countermodels (≤4 elements) for every False pair —
  certificate-pipeline branch, `pipeline/laws.py` + `pipeline/magma.py`.
- The register machinery: `causalab-integration/tasks/etp_implication/`
  treats register as the template variable; a `sql` register slots in as
  one more surface form per law.
- Grading harness patterns from `informalizing-etp/checkform.py`
  (extraction, run-dir conventions) — but your grader executes instead
  of parsing (sqlite3 in stdlib is enough).

## Design

1. **Renderer** (deterministic, no LLM): law AST → canonical SQL query.
   Property test: on every certified countermodel, the canonical query's
   emptiness must match the certificate. This is your ground truth
   generator and it must pass 100% before any model is called.
2. **Translation eval**: model gets the NL register (instance / story),
   must produce SQL. Grade by execution on a battery: the certified
   countermodel + K random tables (sizes 2–4). Statuses: equivalent /
   inequivalent-but-runs (silent!) / SQL error (loud) — the same
   silent/loud axis as the main pipeline, now with executable ground truth.
3. **Cross-register comparison**: same pairs, formal→SQL vs story→SQL vs
   instance→SQL. Does the story register cost more in an executable
   target than in the prefix-grammar target?

**Controls**: table battery must distinguish laws (two different laws
should disagree on some table in the battery — verify offline, grow K
until they do); leave-law-out for anything fitted; report per ops level.

## Kill baselines

- Copy-the-example baseline: a template-matching dummy that emits the
  worked example's query shape with symbols substituted. If it scores
  near the model, the task measures format imitation, not translation.
- Premise-strength shortcut does not apply here (no truth judgment), but
  per-level BoW floors for "did it just memorize law→query pairs" do:
  hold out laws, not just pairs.

## Deliverable

The renderer + its property test (this is publishable infrastructure by
itself), one eval run over 3 models × 2 registers, and the three-way
status breakdown per complexity level. Grammar-transfer tie-in: this is
your "new grammar" (experiment 10) with real-world syntax — compare
fine-tuned vs prompted acquisition later.
