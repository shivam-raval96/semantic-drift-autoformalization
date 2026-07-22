# 10 — Invent-your-own-grammar (qualitative mini-experiment)

## Question

When a model is asked to *invent its own* rigid notation for a Storyform
story — instead of being handed `formalize_prompt.md`'s `op(...)` /
`ASSUME:` / `ASK:` grammar — what does it design, and does showing it more
example stories (K = 1, 2, 3) change the design?

This is qualitative by decision: nothing is graded, and the deliverable is
the collected grammars and encodings themselves. The existing grader cannot
parse self-invented notation; a round-trip decoding arm (a second model
translating the invented notation back into the standard grammar) was
deliberately left as a possible follow-up.

## Setup

- Models: `openai/gpt-5.5`, `google/gemini-2.5-flash`.
- Reasoning: native default (no `reasoning` payload, no regime wrapper).
  gpt-5.5 spent 516–1552 reasoning tokens per call; gemini spent 0.
- Temperature 0, `max_tokens` 16384, one user message, no system prompt.
- Materials (seed 0, drawn from the ETP list, vacuous laws excluded, all
  slots filled from one RNG stream by `run_experiment.py`):
  - Target low: `E17-E4` (`x = y ◇ (y ◇ y)` ⇒ `x = x ◇ y`), ops_total 3,
    depth 2, signal theme.
  - Target high: `E3156-E3446` (`x = (((y ◇ y) ◇ y) ◇ z) ◇ y` ⇒
    `x ◇ y = z ◇ (w ◇ (w ◇ y))`), ops_total 8, depth 4, signal theme.
  - Examples (context only, never encoded): `E10-E4044` (paint),
    `E13-E1576` (tea), `E28-E1750` (graft), all ops_total 6. Example sets
    are nested — K=2 shows K=1's story plus one more — and the targets'
    signal theme is never among the examples, so at every K the notation
    must carry to an unseen setting.
- 2 models × 2 targets × K ∈ {1,2,3} = 12 calls, $0.31 total.

## Prompts

Template `invent_prompt.md`
(sha256 `d4edcb57eafc93591a8239c59d3a039ee89456073117d998dfb56c729f89338b`),
with `{examples}` replaced by the K example stories under
`### Example story N` headings and `{story}` by the target story. It
describes the two-part story shape (exceptionless custom + closing
question), requires a fully specified, unambiguous, complete-but-minimal
notation that separates asserting from asking, and asks for exactly two
sections, `GRAMMAR:` and `ENCODING:`. It deliberately never shows any
notation — no `op(...)`, no `ASSUME`/`ASK`, no equations. The six exact
prompts sent are committed under `runs/run-mini/prompts/`.

## Reproduce

```sh
set -a; source ../.env; set +a   # OPENROUTER_API_KEY
python3 experiments/10-invent-grammar/run_experiment.py --dry-run  # prompts only
python3 experiments/10-invent-grammar/run_experiment.py
```

Report gallery (self-contained HTML, hand-rolled — charts.py has nothing to
chart here): `report/gallery.html`.

## Results

Artifacts: `runs/run-mini/` (`run_meta.json`, `samples.jsonl`,
`results.jsonl`, `prompts/`), gallery in `report/`. All 12 calls succeeded
and all 12 responses split cleanly into GRAMMAR/ENCODING.

**Every encoding is semantically faithful.** Checked by hand against the
ground-truth pairs (up to variable renaming and the usual side-swap/dual
symmetries): 12/12 encodings translate back to exactly the right two laws,
including the depth-4, 8-op target. Whatever notation each model invented,
it encoded the story correctly in it — fidelity was never the
differentiator here.

**Both models converge on the project's own design.** Every invented
grammar has the same skeleton the repo itself uses: declared
universally-quantified starting things, SSA-style named intermediate steps,
one final equality, and an assert-vs-ask marker — i.e. the literalform /
named-intermediaries shape. 11 of 12 encodings use flat named steps
(`r2 := C(hum, r1)`); only gemini (low target, K=2) inlined nested
expressions (`(hum -> (hum -> hum))`), the nested-prose alternative. At
K=3 both models independently reinvent prefix application — gpt-5.5's
`C(first, second)`, gemini's `op(first, second)` — which is byte-for-byte
the hidden benchmark notation with `op` renamed. Notably, `->`/`▷` arrows
or `@`/`*` infix appear at K=1–2 but prefix-with-keywords wins by K=3.

**K systematically changes gpt-5.5's design, monotonically toward
legibility.** K=1: cryptic sigils (`!{w,h | r1 := (h @ h); … | w = r2}`).
K=2: bracketed blocks with a formal BNF (`![w,h,c]{r1:=h▷h;…}=>w=r4`).
K=3: spelled-out keywords and the story's own names
(`ASSERT [whistle, hum, chirp] { relay1 := C(hum, hum); … } ALWAYS whistle
== relay4;`). Spec length grows monotonically with K on both targets
(1128→1486→1508 and 1097→1352→1524 chars), and every spec stays internally
consistent — each encoding conforms to its own grammar.

**Gemini shows no monotone K-trend; it redesigns from scratch each time.**
Six calls produced six structurally different notations (meta-tuples with
an embedded worked example; CUSTOM/QUERY equivalence-*sets*; positional
pair-tuples where assertion vs question is carried by tuple position; at
K=2 high, a tuple whose first element is the operator symbol itself).
Specs run 2–3× longer than gpt-5.5's and mix real syntax rules with prose
"conventions (for clarity, not part of formal syntax)". Two latent spec
defects: at K=1 (low) it glosses `X -> Y` with two *mutually dual*
theme-specific readings ("X is fed through Y" or "Y is poured into X"), an
inconsistency that would flip operand order across themes — the K=2 spec
quietly fixes the gloss — and at K=1 (high) the encoding drops the braces
its own BNF requires around the statement list.

## Conclusions

- Given freedom, both models design what the project already built:
  quantifier declaration + named intermediate steps + final equality +
  assert/ask marker. The repo's given grammar appears to be a natural
  attractor, not an arbitrary convention — and named intermediaries beat
  nested expressions 11:1 even with nobody prescribing either.
- More examples never degraded the invented grammar. For gpt-5.5, K acts
  as a standardization pressure: notation drifts monotonically from
  ad-hoc sigils toward verbose, keyword-labeled, human-readable syntax.
  For gemini, K mostly adds an opportunity to redesign; its instability
  across K (and its dual-reading gloss at K=1) is exactly the kind of
  spec looseness a round-trip decoder would be expected to punish.
- Encoding fidelity was perfect at both complexity extremes, so on this
  tiny sample the hard part of exp 01–09's task is not "express the story
  formally" but "express it in *someone else's* fixed notation" — worth a
  follow-up: a graded round-trip arm (second model decodes the invented
  grammar + encoding back to `op(...)`) at exp-08 scale, to test whether
  self-invented notation is *interpretable*, not just faithful.
