# 10 — Invent-your-own-grammar (qualitative mini-experiment)

## Question

When a model is asked to *invent its own* rigid notation for a Storyform
story — instead of being handed `formalize_prompt.md`'s `op(...)` /
`ASSUME:` / `ASK:` grammar — what does it design, and does showing it more
example stories (K = 1, 2, 3) change the design? A second arm asks the
same question with the grammar's *format* pinned down: the specification
must be given as BNF production rules.

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
- Two arms over identical materials (sampling never consults the
  template, so both runs cover the same targets and examples):
  - **Free-form** (`runs/run-mini`): notation left entirely open. 12
    calls, $0.31.
  - **BNF-constrained** (`runs/run-mini-bnf`): identical prompt except
    the GRAMMAR section must be BNF production rules (nonterminals in
    angle brackets, quoted terminals, `|` alternatives) followed by
    numbered meaning notes. 12 calls, $0.73 — gpt-5.5's reasoning spend
    rose 2–4× under the constraint (516–1552 → 2069–5178 tokens/call,
    $0.29 → $0.72 for its six calls; gemini still spent 0).

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

The BNF arm uses `invent_bnf_prompt.md`
(sha256 `3f673789da44ab5a95807a948ce14159fb15dd2fc4aa4c8847d8733ac7b8a641`),
byte-identical except for the GRAMMAR paragraph of the required-output
section, which prescribes BNF production rules plus meaning notes; its
prompts are under `runs/run-mini-bnf/prompts/`.

## Reproduce

```sh
set -a; source ../.env; set +a   # OPENROUTER_API_KEY
python3 experiments/10-invent-grammar/run_experiment.py --dry-run  # prompts only
python3 experiments/10-invent-grammar/run_experiment.py
python3 experiments/10-invent-grammar/run_experiment.py \
    --prompt-template invent_bnf_prompt.md \
    --out-dir experiments/10-invent-grammar/runs/run-mini-bnf
```

Report gallery (self-contained HTML, hand-rolled — charts.py has nothing to
chart here): `report/gallery.html`.

## Results

Artifacts: `runs/run-mini/` and `runs/run-mini-bnf/` (`run_meta.json`,
`samples.jsonl`, `results.jsonl`, `prompts/`), combined gallery in
`report/`. All 24 calls succeeded and all 24 responses split cleanly into
GRAMMAR/ENCODING.

### Free-form arm

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

### BNF-constrained arm

**The format constraint held everywhere.** All 12 responses lead with
production rules. gpt-5.5's grammars are complete formal artifacts —
lexical productions down to individual digits — and all six of its
encodings are single-line strings its own productions derive, taken so
literally that at K=1 (high) the keywords concatenate (`ASSERTFORALL`)
because its BNF defines no whitespace. Gemini's "BNF" is looser: at K=1
(low) it slips into EBNF `(...)*` repetition, and that same grammar
overfits the target theme (keywords `SIGNALS`, `THROUGH` — the notation
could not represent a paint or tea story without renaming).

**BNF pushes gpt-5.5 to full abstraction, at 2–4× the reasoning spend.**
The story's names vanish from every encoding in favor of canonical
`x1, x2 / r1, r2` (the free-form arm kept `whistle`/`relay1` at K=3), and
the K-drift toward keyword syntax now shows up as `A`/`Q` at K=1 becoming
`ASSERT`/`ASK` by K=2–3. At K=3 (high) its `<application>` production is
literally `"op" "(" <term> "," <term> ")"` — the hidden benchmark
notation again.

**BNF stabilizes gemini's design but degrades its execution.** Where the
free-form arm produced six unrelated designs, all six BNF calls share one
keyword skeleton (`CUSTOM`/`QUESTION`, declared inputs, semicolon steps,
`ASSERT`/`ASK`-`QUERY` comparisons). But the arm contains the
experiment's only unfaithful encoding — low target at K=3:
`hum=hum->hum; Relay1=hum; Relay2=hum->Relay1`, where the middle step is
not derivable from its own BNF and no consistent reading of the shadowed
`hum` yields the right law — plus a K=1 (high) encoding that writes every
step result-last (`hum->hum=Relay1`), reversing its own
`<Procedure> ::= <Identifier> "=" <Operation>` production, with `Relay1`
also outside its lowercase-only `<Identifier>` alphabet.

**Faithfulness: 11/12 vs the free-form arm's 12/12** — and notably the
failure is on the *low*-complexity target.

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
- Encoding fidelity was near-perfect at both complexity extremes (23/24),
  so on this tiny sample the hard part of exp 01–09's task is not
  "express the story formally" but "express it in *someone else's* fixed
  notation" — worth a follow-up: a graded round-trip arm (second model
  decodes the invented grammar + encoding back to `op(...)`) at exp-08
  scale, to test whether self-invented notation is *interpretable*, not
  just faithful.
- Forcing the spec into BNF is not free and not obviously good: it made
  gpt-5.5's grammars machine-grade (at 2–4× reasoning cost, with the
  story's vocabulary abstracted away) while for gemini it standardized
  the *shape* of the design yet produced the experiment's only broken
  encoding and its worst self-conformance violations — echoing exp 08's
  lesson that adding formal scaffolding to the prompt can subtract from
  execution.
