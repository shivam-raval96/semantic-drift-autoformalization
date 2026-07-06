# Theme files

Each `*.json` file in this directory defines one story theme. The
renderer loads every file here at import time (in sorted filename
order), validates it, and exposes it under its `key`. To add a theme,
drop in a new file and run the test suite — the round-trip, no-leakage,
and determinism tests automatically cover every theme on disk.

A theme decides the *setting* of a story; the renderer owns the story's
*skeleton*. Sentences like "Take any two pigments at all — call the
first crimson...", the "; then she ..." step chaining, the intermediate
naming phrases ("sets the result aside as" / "calls the result" /
"calls that"), "However she chooses her two starting...", and
"Separately, ..." are fixed scaffolding; a theme only fills in the
words marked below.

## Annotated example

The `paint` theme rendering the habit `x ∘ y = (y ∘ y) ∘ x` and the
question `x ∘ y = y ∘ x`, with every theme-supplied fragment marked:

    In a certain paint workshop, the colorist follows one unbreakable habit.   <- intro
    Take any two pigments at all — call the first crimson and the second
                 ^^^^^^^^ element_plural       ^^^^^^^ palette[0]
    ochre. She runs two procedures side by side.
    ^^^^^ palette[1]
           ^^^ subject (capitalized)

    In the first, she pours crimson into ochre and sets the result aside
                  ^^^ subject
                      ^^^^^^^^^^^^^^^^^^^^^^^^^ op_agent, {a}=crimson {b}=ochre
    as Batch 1.
       ^^^^^ result_noun

    In the second, she pours ochre into ochre and calls the result Batch 2;
    then she pours Batch 2 into crimson and calls that Batch 3.

    However she chooses her two starting pigments, Batch 1 and Batch 3
                        ^^^ possessive   ^^^^^^^^ element_plural
    always come out the exact same color. That is simply how this
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ same_habit
    workshop works, without exception.
    ^--- closing (whole second sentence)

    One morning her apprentice wonders about something. Take any two          <- question_intro
    pigments — again call the first crimson and the second ochre. Pour
    crimson into ochre and set the result aside as Batch 1. Separately,
    ^^^^^^^^^^^^^^^^^^ op_imperative
    pour ochre into crimson and call the result Batch 2.

    In this workshop, must Batch 1 and Batch 2 always come out the same color?
    ^^^^^^^^^^^^^^^^ question_lead              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ same_question

## Field reference

| Field | Example (`paint`) | Role in the story |
|---|---|---|
| `key` | `"paint"` | Unique id. Must equal the filename stem (`paint.json`). Used in CLI `--theme`, metadata, and corpus filenames. |
| `notes` | free text | Optional. Author documentation; ignored by the renderer. |
| `intro` | `"In a certain paint workshop, the colorist follows one unbreakable habit."` | First sentence of the story. Must name the setting and the agent, and end with a period. The back-parser identifies the theme by this exact sentence. |
| `subject` | `"she"` | The agent's subject pronoun in step sentences ("she pours ..."). Must be exactly `she` or `he` — the back-parser's step regex matches these two pronouns only. |
| `possessive` | `"her"` | Matching possessive, used in "However she chooses **her** two starting pigments" (and typically inside `question_intro`). |
| `element_singular` | `"pigment"` | What one element is called, in "Take any **pigment** at all" and "starting **pigment**". |
| `element_plural` | `"pigments"` | Plural form, in "Take any two **pigments** at all" and "starting **pigments**". |
| `palette` | `["crimson", "ochre", ...]` | Concrete stand-ins for variables, assigned by order of first appearance (LHS first, then RHS). At least 6 names — an ETP law can use up to 6 distinct variables. Each name must be a single lowercase word, all distinct. |
| `op_agent` | `"pours {a} into {b}"` | The operation as a third-person clause. `{a}` is the left operand and `{b}` the right operand of `a ∘ b`; each placeholder must appear exactly once. |
| `op_imperative` | `"pour {a} into {b}"` | The same clause in the imperative, used in the question section ("Pour crimson into ochre ..."). Must keep the identical operand roles. |
| `result_noun` | `"Batch"` | Label for named intermediates: "Batch 1", "Batch 2", ... A single capitalized word, distinct from every palette name. |
| `same_habit` | `"come out the exact same color"` | Completes the habit's comparison: "X and Y always **...**." Must read correctly after a plural subject and carry no directionality (see rules below). |
| `same_question` | `"come out the same color"` | Completes the closing question: "must X and Y always **...**?" May equal `same_habit`. |
| `closing` | `"That is simply how this workshop works, without exception."` | Final sentence of the habit section. Should restate that the custom holds without exception. |
| `question_intro` | `"One morning her apprentice wonders about something."` | Introduces the wonderer and separates the habit from the question. The back-parser splits the story on this exact sentence, so it must differ from everything else in the theme. |
| `question_lead` | `"In this workshop"` | Opens the final question: "**In this workshop**, must ...?" No trailing comma — the renderer adds it. |

## Authoring rules

These follow from the design invariants in [CLAUDE.md](../CLAUDE.md);
the loader (`storyform.theme_from_dict`) enforces the mechanical ones
and the test suite catches the rest.

1. **The action must be order-sensitive** (invariant 4). `{a}` and
   `{b}` must play visibly different roles — "pours {a} into {b}",
   "grafts {a} onto {b}". Never a symmetric phrasing like "mixes {a}
   with {b}" or "combines {a} and {b}": those silently assert the
   commutativity the law may not have, and the story would no longer
   determine the argument order (invariant 6).
2. **No formal vocabulary anywhere** (invariant 2). No field may
   contain words like "equation", "law", "rule number", "magma",
   "implies", operator symbols, single-letter variable names, or
   digits attached to letters. The no-leakage tests scan rendered
   stories for all of this.
3. **Vocabulary must not collide.** Palette names, the `result_noun`,
   and the words of the op clauses are what the back-parser uses to
   re-read the story. Palette names must be single lowercase words,
   all distinct, and none may equal the `result_noun`; avoid palette
   words that also appear inside the op clause (e.g. a palette color
   named "leaves" would collide with the tea theme's op phrasing).
4. **Sameness phrases must be symmetric and non-committal.** They
   complete "X and Y always ..." — use a plural verb ("come out ...",
   "taste ..."), do not favor one side ("becomes", "turns into ...
   the other"), and never hint at an answer (invariant 1).
5. **Pronouns are a contract.** `subject` must be `she` or `he`, and
   `possessive` must match; the back-parser and the fixed scaffolding
   conjugate for singular third person.
6. **Everything must stay deterministic** (invariant 3). A theme is
   plain data — no templating logic, no alternatives to pick from.
   One theme, one voice.

Adding or removing a theme file changes how the hash-based automatic
theme selection distributes pairs across themes (it hashes into the
sorted list of theme keys). Individual stories are still fully
deterministic; pass `--theme <key>` to pin a specific theme when
regenerating an existing corpus.

## Checking a new theme

```sh
python3 -c "import storyform"          # loader validation errors surface here
python3 test_storyform.py              # full suite runs over every theme on disk
python3 storyform.py "x ∘ y = (y ∘ y) ∘ x" "x ∘ y = y ∘ x" --theme <key>
```
