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

## Files

- [examples.md](examples.md) — one implication (E387 ⇒ E43) rendered
  under every theme, side by side.
- [examples-tea.md](examples-tea.md) — the complementary cut: one theme
  (tea) rendering several implications of different shapes.
- [storyform.py](storyform.py) — parser, theme loader, renderer, and CLI.
- [themes/](themes/) — one JSON file per theme; schema and authoring
  rules documented in [themes/README.md](themes/README.md).
- [backparse.py](backparse.py) — recovers both laws from story text alone.
- [test_storyform.py](test_storyform.py) — test suite.
- [CLAUDE.md](CLAUDE.md) — full design document: invariants, conventions,
  pitfalls, and roadmap.

## Testing

```sh
python3 test_storyform.py
```

The suite covers, in priority order:

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

## Roadmap

- **ETP data ingestion** — load equations and implication pairs from the
  `teorth/equational_theories` data files instead of hard-coding examples.
- **Corpus export** — batch JSONL export pairing stories with formal
  metadata (ETP numbers, raw equations, Lean statements).
- **CI round-trip gate** — run the back-parser over every exported story.
- **Theme library growth** — more action domains, each vetted against the
  order-sensitivity, injectivity, and no-leakage invariants.
