# Storyform — Deterministic Informalization of ETP Implications

Storyform turns implication statements from the [Equational Theories
Project](https://github.com/teorth/equational_theories) (ETP) into
natural-language stories. Each implication "law E implies law F" becomes a
short narrative: a world whose custom always upholds the first law, ending
with an open question asking whether a second regularity must also hold.
The magma operation becomes a concrete, order-sensitive action — pouring
paint, steeping tea, grafting plants, relaying signals.

This is the *reverse* of autoformalization: formal → informal, with
fidelity guaranteed by construction rather than checked after the fact.
The resulting (story, formal statement) pairs are a corpus for training
and stress-testing autoformalization systems on non-standard phrasings.

## Example

Input: `x ∘ y = (y ∘ y) ∘ x` implies `x ∘ y = y ∘ x` (ETP: E387 ⇒ E43),
rendered with the paint theme:

> In a certain paint workshop, the colorist follows one unbreakable habit.
> Take any two pigments at all — call the first crimson and the second
> ochre. She runs two procedures side by side.
>
> In the first, she pours crimson into ochre and sets the result aside as
> Batch 1.
>
> In the second, she pours ochre into ochre and calls the result Batch 2;
> then she pours Batch 2 into crimson and calls that Batch 3.
>
> However she chooses her two starting pigments, Batch 1 and Batch 3
> always come out the exact same color. That is simply how this workshop
> works, without exception.
>
> One morning her apprentice wonders about something. Take any two
> pigments — again call the first crimson and the second ochre. Pour
> crimson into ochre and set the result aside as Batch 1. Separately,
> pour ochre into crimson and call the result Batch 2.
>
> In this workshop, must Batch 1 and Batch 2 always come out the same
> color?

The story never answers the question, never mentions equations, variables,
or the ETP, and never asserts the converse. The formal side of the pair
lives only in external metadata.

## Usage

Requires Python 3; standard library only.

```sh
# Render a story (theme chosen by a deterministic hash of the pair)
python3 storyform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x"

# Force a theme
python3 storyform.py "x = x ◇ x" "x ◇ y = y ◇ x" --theme tea

# Print a JSON corpus record: story plus formal metadata
python3 storyform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x" \
    --e-label E387 --f-label E43 --json

# Write the JSON record to a file instead (prints the path)
python3 storyform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x" \
    --e-label E387 --f-label E43 --out-dir corpus
```

With `--out-dir`, the record is written to `DIR/<name>.json` with a
deterministic filename: `E387-E43-paint.json` when both ETP labels are
given, otherwise `pair-<hash>-<theme>.json` from a hash of the
canonicalized pair. Re-rendering the same pair overwrites the same file,
so a corpus directory never accumulates duplicates.

Equations follow the ETP convention: fully parenthesized nesting, with
`∘`, `◇`, or `*` as the operation symbol. Ambiguous chains like
`x ∘ y ∘ z` are rejected. ETP numbering (E387, E43, ...) is accepted only
as metadata labels — it never appears in story text.

## How it works

Pipeline: equation string → AST → SSA-style procedure → themed
question-story.

1. A recursive-descent parser turns each equation into a term tree.
2. Each side of a law is linearized post-order into numbered steps with
   named intermediates ("Batch 1", "Brew 2"); the counter is shared
   across both sides of a law so names never clash.
3. Variables map to the theme's fixed name palette by order of first
   appearance (LHS first, then RHS). The habit (E) and the question (F)
   are quantified independently, each with its own "take any ..." clause.
4. The habit is narrated as an exceptionless custom; the question is
   posed one-directionally and left unanswered.

The renderer is a pure function: the same (E, F) pair always produces the
byte-identical story. Theme selection hashes the *canonicalized* pair, so
whitespace and the choice of op symbol don't change the output.

### Themes

| Theme    | Action (`a ∘ b`)                  | Intermediate | Palette                     |
|----------|-----------------------------------|--------------|-----------------------------|
| `paint`  | pours *a* into *b*                | Batch        | crimson, ochre, teal, ...   |
| `tea`    | pours *a* over the leaves of *b*  | Brew         | jasmine, oolong, rooibos, ...|
| `graft`  | grafts *a* onto *b*               | Graft        | quince, medlar, damson, ... |
| `signal` | feeds *a* through *b*             | Relay        | whistle, hum, chirp, ...    |

Every action phrasing is order-sensitive by design — symmetric wordings
("mix a and b") are banned because they silently assert commutativity the
law may not have. Each palette has six names, enough for the maximum of
six distinct variables an ETP law can use.

Themes live as JSON files in [themes/](themes/), one file per theme,
loaded and validated at import time. To add a theme, drop a new
`themes/<key>.json` in place and run the test suite — every theme on
disk is automatically covered by the round-trip and no-leakage tests.
See [themes/README.md](themes/README.md) for the full schema, an
annotated example story, and the authoring rules.

### Invertibility

The rendering grammar is injective: the story text alone determines both
term trees, the palette-name/variable correspondence, and every argument
order. [backparse.py](backparse.py) implements the reverse direction —
story text → term trees — and the test suite round-trips every story
through it. This is what makes the metadata trustworthy without any
annotations in the text.

## Literal descriptions — the direct arm

[literalform.py](literalform.py) is the contrasting renderer: instead of a
fuzzy themed story, it renders the same implication as a **direct**
natural-language description — plain English that openly talks about an
operation, its two ordered inputs, and lettered variables (x, y, z, ...),
with no story dressing:

> Suppose the following always holds. For every choice of objects x and y,
> apply the operation to x as its first input and y as its second input,
> and call the result Value 1; then apply the operation to y as its first
> input and y as its second input, and call the result Value 2; then apply
> the operation to Value 2 as its first input and x as its second input,
> and call the result Value 3. Then Value 1 is always equal to Value 3.
>
> Now consider the following question. ... Does it follow that ...?

Terms render as definition steps — each application of the operation is
introduced on its own and its result named ("call the result Value 1"),
the literal counterpart of the story arm's named intermediates
("Batch 1") — so nesting never appears inline, is unambiguous without
parentheses, and the text stays injective (`backparse_literal` recovers
both term trees; the test suite round-trips every pair). Like storyform
it is a pure function of the (E, F) pair, and it emits the same record
schema, so `checkform.py` grades answers to either style unchanged.
(Through experiment 03 the literal arm instead rendered nesting inline in
a words-only prefix grammar; the definition-step grammar replaced it for
experiment 04.)

[literal_prompt.md](literal_prompt.md) is the companion system prompt:
same self-contained `op(first, second)` / `ASSUME:` / `ASK:` contract as
`formalize_prompt.md`, but teaching the mapping from the literal grammar
("first input" → first argument of `op`, Value names unfolded into nested
expressions) instead of from story conventions. Pass it to `build_prompt`
via `template_path`.

```sh
# Render a literal description
python3 literalform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x"

# Write a checkform-compatible record (filename tagged -literal)
python3 literalform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x" \
    --e-label E387 --f-label E43 --out-dir corpus
```

Comparing model accuracy on the two arms — literal description vs. themed
story of the identical implication — measures how much of the
formalization difficulty comes from the story indirection itself.
`benchmark.py --form literal` runs the eval on this arm end to end
(rendering with literalform and prompting with `literal_prompt.md`);
sampling ignores the form, so story and literal runs with the same seed
score the identical pair set and differ only in the rendering.

## Grading model formalizations

The corpus exists to test whether models can formalize the stories back.
[checkform.py](checkform.py) closes that loop without assuming the model
knows Lean or any ETP convention: [formalize_prompt.md](formalize_prompt.md)
teaches a tiny self-contained answer syntax inside the prompt itself, and
grading is pure syntactic comparison — no LLM judging.

The model is asked to end its response with two lines in a prefix notation
defined entirely in the prompt (an expression is an ingredient name or
`op(first, second)`):

```
ASSUME: op(x, y) = op(op(y, y), x)
ASK: op(x, y) = op(y, x)
```

The checker extracts the last `ASSUME:`/`ASK:` lines from the raw response,
parses them, canonicalizes, and compares against the record's
`canonical_e`/`canonical_f`. Three symmetries are accepted, because each is
a faithful reading of the story:

- **Variable renaming** — names are arbitrary; canonicalization absorbs them.
- **Side swap** (per equation) — "the two results always come out the same"
  is symmetric, so the order of an equation's sides is not recoverable intent.
- **Consistent dualization** — the story never says which participant of the
  action is `op`'s first argument; an answer using the opposite convention
  uniformly across *both* equations is the dual implication, semantically
  the same claim.

The direction of the implication (ASSUME vs ASK) is never lenient. The
verdict JSON reports `status` (`correct` / `wrong` / `unparseable`) and the
minimal accepted `transform` (`swap_e`, `swap_f`, `dual` flags), so eval
stats can distinguish exact-convention answers from dualized ones.

```sh
# Print the filled prompt for a corpus record
python3 checkform.py prompt corpus/E387-E43-paint.json

# Grade a raw model response; exit 0 correct, 1 wrong, 2 unparseable
python3 checkform.py grade corpus/E387-E43-paint.json response.txt
```

## Files

- [examples.md](examples.md) — one implication (E387 ⇒ E43) rendered
  under every theme, side by side.
- [examples-tea.md](examples-tea.md) — the complementary cut: one theme
  (tea) rendering several implications of different shapes.
- [storyform.py](storyform.py) — parser, theme loader, renderer, and CLI.
- [themes/](themes/) — one JSON file per theme; schema and authoring
  rules documented in [themes/README.md](themes/README.md).
- [backparse.py](backparse.py) — recovers both laws from story text alone.
- [literalform.py](literalform.py) — the direct arm: literal
  natural-language descriptions of the same implications, with its own
  back-parser.
- [formalize_prompt.md](formalize_prompt.md) — self-contained prompt
  teaching the answer syntax; `{story}` placeholder filled per record.
- [literal_prompt.md](literal_prompt.md) — companion prompt for literal
  descriptions; same answer syntax, graded by the same checker.
- [checkform.py](checkform.py) — answer extraction, prefix parser, and
  syntactic grader with CLI.
- [test_storyform.py](test_storyform.py) — renderer test suite.
- [test_literalform.py](test_literalform.py) — literal-renderer test suite.
- [test_checkform.py](test_checkform.py) — grader test suite.
- [experiments/](experiments/) — the committed lab notebook: each
  benchmark experiment with its write-up, run data, and chart report
  (conventions in [experiments/README.md](experiments/README.md)).
- [CLAUDE.md](CLAUDE.md) — full design document: invariants, conventions,
  pitfalls, and roadmap.

## Testing

```sh
python3 -m unittest test_storyform test_checkform
```

The renderer suite covers, in priority order:

1. **Round-trip** — the back-parser recovers both term trees and the
   palette order from story text, across all pairs and themes.
2. **No-leakage** — stories contain no operator symbols, no banned formal
   vocabulary ("equation", "law", "magma", ...), no standalone variable
   letters, and no letter-attached digits like "E387".
3. **Determinism** — two renders of the same pair are byte-identical, and
   theme selection is stable across input formatting.
4. **Coverage** — degenerate shapes render and round-trip: bare-variable
   sides (`x = ...`), repeated variables, maximal 4-operation nesting,
   and the six-variable maximum.

It also byte-compares the E387 ⇒ E43 habit against the worked example in
CLAUDE.md.

The grader suite covers answer extraction from messy responses, rejection
of malformed syntax, one test per verdict class — including the symmetries
that must be accepted (side swap, uniform dualization) and those that must
not (reversed implication, dual applied to only one equation) — plus
determinism and a round-trip where a perfect answer built from each corpus
record's own equations grades correct.

## Roadmap

- **ETP data ingestion** — load equations and implication pairs from the
  `teorth/equational_theories` data files instead of hard-coding examples.
- **Corpus export** — batch JSONL export pairing stories with formal
  metadata (ETP numbers, raw equations, Lean statements).
- **CI round-trip gate** — run the back-parser over every exported story.
- **Batch eval harness** — run a model over every corpus prompt and
  aggregate checkform verdicts (correct / dualized / swapped / wrong /
  unparseable rates).
- **Theme library growth** — more action domains, each vetted against the
  order-sensitivity, injectivity, and no-leakage invariants.
