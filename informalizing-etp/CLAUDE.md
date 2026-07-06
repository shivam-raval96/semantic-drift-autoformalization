# CLAUDE.md

## Project: Storyform — Deterministic Informalization of ETP Implications

This project converts implication statements from the Equational Theories
Project (ETP) into natural-language stories. The magma operation becomes a
concrete, order-sensitive action (pouring paint, steeping tea, grafting
plants), and each implication statement is rendered as a deterministic story
built around that action.

Scope and form:
- We informalize the *statement* "law E implies law F" — never its proof and
  never a counterexample. Truth is irrelevant; every (E, F) pair is treated
  identically.
- Every story reads as a **question**: it narrates a world whose custom
  always holds (law E) and ends by asking whether a second regularity
  (law F) must also hold.
- Stories are **pure narrative**. They contain no mathematical notation, no
  equation numbers, no variable letters, no bracketed formal statements —
  nothing that reveals the abstract source. The formal side of each pair
  lives only in external metadata, never in the story text.

This is the *reverse* of autoformalization: formal → informal, with fidelity
guaranteed by construction rather than checked after the fact. Output pairs
(story, formal statement) are intended as a corpus for training and
stress-testing autoformalization systems on non-standard phrasings.

## Background context

- The ETP (Tao et al., 2024–2025) resolved all 22,028,942 implications among
  the 4,694 simplest equational laws for magmas, fully verified in Lean 4.
  Repo: `github.com/teorth/equational_theories`. Report: arXiv:2512.07087.
- A magma is a set with one binary operation and no axioms. Laws are
  universally quantified identities like `x ∘ y = (y ∘ y) ∘ x`.
- "E implies F" means: every magma satisfying E also satisfies F.
- Background docs live in `/mnt/project/` (read-only).

## Intended architecture

Pipeline: equation string → AST → SSA-style procedure → themed question-story.

- **AST**: parse equations into a term tree. Use a recursive-descent parser
  expecting the ETP convention of fully parenthesized nesting. Op symbols
  to support: `∘`, `◇`, `*`.
- **Themes**: each theme defines a setting, agent, element noun, a fixed
  palette of element names, and a sentence template for the operation. Theme
  selection should be a deterministic function of the equation pair (e.g. a
  hash mod the number of themes).
- **Law renderer**: renders a single equational law as a narrated habit —
  "take any two pigments; follow these steps; the two results always come
  out the same color." Universal quantification is expressed as "take any
  ... at all", with concrete palette names introduced as stand-ins.
- **Question-story renderer**: the single top-level renderer for all (E, F)
  pairs. It narrates a world where the first habit always holds, then poses
  the second regularity as an open question ("must it also be true that
  ...?"). The story never answers the question and never labels either habit
  with a number, letter, or formula.

## Worked example — target output shape

Input: the implication `E387 ⇒ E43`, where E387 is `x ∘ y = (y ∘ y) ∘ x` and
E43 is `x ∘ y = y ∘ x`. Suppose the hash of this pair selects the paint
theme (`a ∘ b` = "pour a into b"). The rendered story:

```
In a certain paint workshop, the colorist follows one unbreakable
habit. Take any two pigments at all — call the first crimson and the
second ochre. She runs two procedures side by side.

In the first, she pours crimson into ochre and sets the result aside
as Batch 1.

In the second, she pours ochre into ochre and calls the result
Batch 2; then she pours Batch 2 into crimson and calls that Batch 3.

However she chooses her two starting pigments, Batch 1 and Batch 3
always come out the exact same color. That is simply how this
workshop works, without exception.

One morning her apprentice wonders about something. Take any two
pigments — again call them crimson and ochre. Pour crimson into ochre
and set the result aside as Batch 1. Separately, pour ochre into
crimson and call the result Batch 2.

In this workshop, must Batch 1 and Batch 2 always come out the same
color?
```

Things to notice, all of which follow from the invariants below:
- Nothing in the text mentions equations, rules by number, variable
  letters, or the ETP — it reads as a self-contained story ending in a
  question, and no formal statement is appended.
- The story asserts only that the first habit holds and *asks* about the
  second; it never answers, hints, or argues.
- Argument order is preserved everywhere ("pour A into B" for `A ∘ B`),
  so the question's two procedures are visibly different.
- Each nested subterm becomes a named batch; batch numbering is shared
  across the procedures of a single habit so names never clash.
- The habit and the question each get their own fresh "take any two
  pigments" clause — their choices are independent even though the same
  palette names reappear.

## Design invariants — do not break these

1. **Statements only, phrased as questions.** The system renders the
   implication claim as an unanswered question. No proofs, no
   counterexamples, no Cayley tables, no truth-checking, no answers or
   hints anywhere in the pipeline.
2. **No formal leakage.** Story text must contain no mathematical notation,
   equation numbers, variable letters, formulas, or references to laws,
   magmas, or the ETP. The (story, formal statement) pairing is recorded
   only in metadata alongside the story, never inside it.
3. **Pure functions only.** The same (E, F) pair must always produce the
   byte-identical story. No randomness, no clocks, no ambient state. Any
   "variety" must be a deterministic function of the input (hashing is fine).
4. **Order-sensitive action rendering.** `a ∘ b` renders with a fixed
   argument convention ("pour a into b"). Never use symmetric phrasings
   ("mix a and b") — they silently assert commutativity the law may not have.
5. **No ambiguous nesting.** Term trees are always rendered as sequences of
   steps with named intermediates ("Batch 1", "Brew 2"), never as nested
   prose. Intermediate counters should be shared across both procedures of
   a single law so names never clash.
6. **Invertibility from narrative alone.** The rendering grammar must stay
   injective: the story text by itself must determine both term trees, the
   correspondence between palette names and variables (order of first
   appearance), and every argument order. This is what makes the metadata
   trustworthy without annotations in the text.
7. **Faithfulness over fluency.** If a nicer-sounding phrasing loses
   information (quantifier scope, argument order, tree shape, direction of
   the question), reject it.

## Conventions

- Variables map to the theme's fixed name palette by order of first
  appearance (LHS first, then RHS), introduced narratively ("call the first
  crimson"). At least 6 palette names per theme — enough, since ETP laws
  use at most 4 operations.
- The habit (E) and the question (F) are quantified independently — reuse
  of palette names across them carries no meaning, and each gets its own
  "take any ..." clause.
- Universal quantification is always narrated explicitly ("take any two
  pigments at all", "however she chooses") — never left implicit.
- The question must be one-directional: it asks whether the second
  regularity follows, never whether the two habits are equivalent.
- Canonical equation statements come from the ETP repo's equation list, not
  reconstructed from memory; ETP numbering appears only in metadata.
- Python 3, standard library only unless a dependency clearly pays for
  itself.

## Testing

No tests yet. When adding them, priority order:
1. Round-trip test — a back-parser recovers both term trees from the story
   text alone, matching the original pair of ASTs.
2. No-leakage test — story text contains no digits attached to equations,
   no operator symbols, no single-letter variable mentions, and none of a
   banned-word list ("equation", "law", "magma", "implies", "theorem", ...).
3. Determinism test — two renders of the same pair are identical.
4. Coverage test — the renderer handles degenerate shapes (bare-variable
   sides, repeated variables, maximal 4-operation nesting) without error.

## Roadmap

- **ETP data ingestion**: load the equation list and implication pairs from
  the `teorth/equational_theories` repo data files rather than hard-coding
  examples.
- **Back-parser**: mechanical story → implication parser to enforce
  invariant 6 in CI.
- **Corpus export**: JSONL records pairing each story with its formal
  metadata (equation numbers, raw equations, Lean statement) for
  autoformalization robustness evaluation.
- **Theme library growth**: more action domains, each vetted against the
  order-sensitivity, injectivity, and no-leakage invariants.

## Pitfalls to remember

- ETP equations are only meaningful with explicit parenthesization —
  never assume associativity when parsing or rendering.
- The LHS of many ETP laws is a bare variable (`x = ...`); the law renderer
  must handle the degenerate side ("the result is always just crimson
  again").
- Named intermediates ("Batch 1") are narrative devices, not formal labels —
  keep them, but never let genuine formal identifiers (x, y, E387, ∘) slip
  into the text.
- Do not let story language accidentally assert the converse, an
  equivalence, or the answer to the question.
- Do not "improve" stories with LLM rewriting inside the pipeline — that
  reintroduces exactly the fidelity risk the deterministic design removes.
  LLM polish, if ever added, goes in a separate post-processing stage with
  a round-trip check gating its output.
