# Template hand-check — 2026-07-05 (walkthrough with Luiza, live sign-offs)

Scope: all 10 CORE NL template sets (40 surfaces), the distractor-register gold
audit, and the stimulus decisions feeding prereg v1. This file is the source of
truth for the template bank to be implemented in the MARS tree (two-tier
decision: confirmatory changes land there; this lab tree keeps the exploratory
strings frozen for reproducibility of past runs).

## Finding: distractor register was unwinnable by construction
Every `distractor` surface described the NEIGHBOR law while Family B assigned
gold = the original law (dataset.py family_B includes all registers). Verified
against results: 0/10 faithful (haiku, llama, 4o-mini), 1/10 (sonnet); e.g.
`B-assoc-distractor` surface = "the law where the order of the two operands
does not matter" (= comm), sonnet output `x * y = y * x` — a CORRECT reading —
scored "drift: incomparable".

Corrected exploratory numbers (distractor excluded), for the FINDINGS appendix:
| model | B faithful (was) | B faithful (corrected) | VBU share of failures (was) | (corrected) |
|---|---|---|---|---|
| claude-3-haiku | 35% | 47% | 27% | 31% |
| llama-3.3-70b | 50% | 67% | 95% | 90% |
| gpt-4o-mini | 55% | 73% | 78% | 50% |
| claude-sonnet-4.6 | 62% | 80% | 100% | 100% (6/6) |
Failure-migration pattern SURVIVES; magnitudes change. Library-trained probe
runs (0.804 / 0.84 monitors) had mislabeled distractor items in TRAIN — noted
for the monitor grumeter card. Big-E monitor (0.96) and H1 (fresh ETP): unaffected.

## Per-set verdicts (all signed off by Luiza in-session)

| law | canonical | paraphrase | instance | notes |
|---|---|---|---|---|
| comm | keep | keep | keep | approved as-is |
| assoc | keep | keep | keep | approved as-is |
| idem | keep | keep | keep | approved as-is |
| lproj | keep | keep | keep | approved as-is |
| rproj | keep | keep | keep | approved as-is; 'projection' names judged compositionally derivable |
| lselfdist | keep | keep | keep | approved as-is; real name (racks/quandles) |
| medial | keep | **REWRITE** | keep | real name kept; paraphrase was the documented right-nesting ambiguity |
| unipot | **OBSCURE-NAME STRATUM** | keep | keep | canonical items stay but are preregistered as a separate 'obscure-name behavior' stratum, never pooled into faithful rates |
| labsorb | **DROP** | keep | keep | name collides with lattice absorption (different law); symmetric with rabsorb |
| rabsorb | **DROP** | **REWRITE** | keep | name collision + paraphrase tightened to force left-nesting; instance register is the sonnet re-nesting anomaly item — kept unchanged |

## Approved rewrites

rabsorb paraphrase (was: "combining something with a second thing and then with
that same second thing again recovers the original"):
> "take a thing and combine it with a second thing; then combine that result
> with the same second thing again: you get the original thing back"

medial paraphrase (was: "when combining two combinations, exchanging the two
middle elements does not change the result"):
> "combine one pair, separately combine a second pair, then combine the two
> results; swapping the second element of the first pair with the first element
> of the second pair does not change the outcome"

## New distractor-adjacent register (approved, all 10)

Design rule: the surface CORRECTLY describes the intended law (gold = intended,
well-posed) while borrowing the neighbor's vocabulary — the register measures
careful reading vs lexeme pattern-matching. Each checked against its equation.

| law | evokes | surface |
|---|---|---|
| comm | assoc | "the law where the two elements being combined can trade places — whatever grouping surrounds them — without changing the result" |
| assoc | comm | "the law where the order in which you perform the combinations does not matter, so long as the left-to-right order of the elements stays fixed" |
| idem | absorption | "the law where an element combined with itself is absorbed back into that same element" |
| unipot | idem | "the law where every element, combined with itself, collapses to one single value — not necessarily the element you started with" |
| lproj | rproj | "the law where, of the two elements combined, it is always the earlier one — never the later — that comes back" |
| rproj | lproj | "the law where, of the two elements combined, it is always the later one — never the earlier — that comes back" |
| labsorb | idem | "the law where a thing, combined with the combination of that same thing with another, cancels itself out and returns the other" |
| rabsorb | labsorb | "the law where combining with the same second element twice in a row undoes itself, returning the first element" |
| lselfdist | assoc | "the law where a combination with a combined pair can be regrouped — provided the outer element is copied into both parts" |
| medial | lselfdist | "the law where combining two combinations distributes their four elements into new pairs — firsts together, seconds together — without changing the result" |

## Implementation notes (Phase 1, MARS tree)
- Template bank carries a REGISTER_STATUS field per (law, register):
  ok | obscure-name-stratum | dropped | rewritten-v2. Old strings preserved
  under *_v1 keys for the exploratory record.
- Report generators: exclude `dropped`, stratify `obscure-name-stratum` out of
  faithful-rate denominators, and stratify substitution-certified implications
  (separate decision, same session).
- Prompt-freeze hash updates are a conscious act: new hashes recorded in
  frozen-config when the MARS bank is built.
