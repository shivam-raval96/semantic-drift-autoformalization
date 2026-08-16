# Deeper problems are harder, and it is not because they are longer

*Qwen3-4B, 200 problems at each of eight settings, one fixed reasoning budget of
2048 tokens, seed 0. Produced by `depth-at-fixed-length.py` in this folder; the
run directory holds every graded answer.*

## The headline

The model was given problems that are word-for-word the same length, take the
same number of steps to write out, and use the same number of variables, and
that differ only in how much of the work has to be done in sequence. Accuracy
fell from **91.5% to 70.0%** in one problem set and from **87.5% to 54.5%** in
the other, 200 problems at each point. The prompt is 1467 tokens long at every
point in the first set and 1525 at every point in the second — a single value,
not an average.

That last sentence is the whole reason for the experiment. Every earlier claim
in this project about problem complexity was a claim about text length wearing a
different name.

## What the model was asked to do

Each problem gives two algebraic laws written out in plain English — a
"description" of the form *take a value, combine it with another, call the
result Value 1; combine Value 1 with ..., call that Value 2; ...* — and asks the
model to write those same two laws back in the project's terse formal notation.
It is a translation task, not a proof: the model never has to decide whether one
law implies the other, only to reconstruct the two nested expressions the
description spells out step by step.

An answer counts as correct only if a syntactic checker can parse it and it
matches the intended pair of laws, allowing for the two laws being named in
either order.

## The confound this was built to remove

The way these problems are generated, a law with more operations in it renders
into proportionally more text: each operation becomes one fixed-length sentence.
Measured directly, the number of words in a problem and the number of operations
in its law correlate at 0.93 for the story rendering and 1.00 for the formal
notation. So "the model's internal representations separate simple from complex
problems", which is the finding Shivam singled out as the interesting one, and
"the model's internal representations separate short from long inputs" were, up
to that point, the same statement.

Neither obvious repair works. Subtracting length statistically removes the
complexity signal with it when the correlation is 1.00. Picking length-matched
problems out of a bigger pool fails because there is no such thing as a long
two-operation problem to pick — a two-operation law renders as two sentences and
that is the entire text.

**Depth** is a way out. It is how many operations sit nested inside one another,
which is the length of the chain that has to be worked through one link at a
time. At a fixed number of operations, depth is free: six operations can be
arranged as three independent pairs, or as a single chain six deep. Both render
into six sentences and identical text.

```
x = (x ◇ (y ◇ x)) ◇ (z ◇ ((z ◇ x) ◇ (y ◇ w)))      depth 4
x = (y ◇ (y ◇ (z ◇ (z ◇ ((w ◇ z) ◇ w))))) ◇ z      depth 7
```

Both are seven operations, four variables, and both render into exactly 399
words of plain-English description. The first breaks into branches that can be
worked out in any order; the second is one chain where every step needs the
previous step's answer.

Holding text length exactly fixed took pinning two more things besides the
operation count — which side of the equation the operations sit on, and how many
distinct variables appear — because both change how long the rendering is and
both drift with depth if left alone. With all three pinned, word count has a
standard deviation of 0.0 across every depth, in all six ways of writing the
problem. The reasoning is written up in `depth-at-fixed-length.md`.

A **problem set** below is a group that agrees on all three pinned quantities
and differs only in depth. There are two of them, so every result gets a
built-in replication: an effect that appears in one and not the other belongs to
that set's operation count, not to depth.

## Results

Six operations per equation, three variables, prompt 1467 tokens at every depth:

| depth | accuracy | 95% confidence interval | reasoning tokens used | hit the 2048-token limit |
|---|---|---|---|---|
| 3 | 91.5% (183/200) | 87%–95% | 972 | 0% |
| 4 | 91.0% (182/200) | 86%–94% | 1053 | 0% |
| 5 | 80.0% (160/200) | 74%–85% | 1188 | 6% |
| 6 | 70.0% (140/200) | 63%–76% | 1343 | 6% |

Seven operations per equation, four variables, prompt 1525 tokens at every depth:

| depth | accuracy | 95% confidence interval | reasoning tokens used | hit the 2048-token limit |
|---|---|---|---|---|
| 4 | 87.5% (175/200) | 82%–91% | 1078 | 3% |
| 5 | 80.5% (161/200) | 74%–85% | 1259 | 11% |
| 6 | 60.5% (121/200) | 54%–67% | 1399 | 22% |
| 7 | 54.5% (109/200) | 48%–61% | 1411 | 18% |

Deepest minus shallowest is **−21.5 points** (95% interval −28.9 to −13.9) in
the first set and **−33.0 points** (−40.9 to −24.4) in the second. Both
intervals exclude zero. The comparison across depths is unpaired and cannot be
otherwise — one law cannot be both depth 4 and depth 7 — so these are Newcombe
intervals for the difference between two independent proportions, which are
wider than a paired test would give. That is why each point uses 200 problems
rather than the roughly 50 used elsewhere in this project.

The model also visibly works harder as depth rises, spending 972 rising to 1343
reasoning tokens in the first set and 1078 rising to 1411 in the second, on
inputs of identical length.

## The one objection, and what happened to it

The share of problems that ran out of reasoning room rises with depth, from 0%
to 6% in the first set and 3% to 22% in the second. So some of the fall could be
"ran out of room to think" rather than "harder problem".

Restricting to only the problems where the model finished reasoning on its own,
with no truncation, the fall survives almost intact:

| problem set | depth | all problems | finished reasoning on its own |
|---|---|---|---|
| six operations | 3 | 91.5% (183/200) | 91.5% (183/200) |
| six operations | 4 | 91.0% (182/200) | 91.0% (181/199) |
| six operations | 5 | 80.0% (160/200) | 83.5% (157/188) |
| six operations | 6 | 70.0% (140/200) | 72.3% (136/188) |
| seven operations | 4 | 87.5% (175/200) | 88.7% (172/194) |
| seven operations | 5 | 80.5% (161/200) | 86.5% (154/178) |
| seven operations | 6 | 60.5% (121/200) | 66.7% (104/156) |
| seven operations | 7 | 54.5% (109/200) | 57.0% (94/165) |

The drop goes from 21.5 to 19.2 points in the first set and from 33.0 to 31.7 in
the second. Truncation accounts for a couple of points at most. This is a
selected subset rather than a clean experiment — the problems that finish early
are the easier ones — so it bounds the objection rather than eliminating it. The
definitive version is a rerun at 4096 tokens where the limit almost never binds;
it is eight new settings and about two hours of GPU time, and it reuses the
existing conditions rather than redoing them.

## What this does and does not establish

It establishes that this model has a difficulty axis that text length cannot
explain, and gives an ordered handle on it: four levels, replicated in two
independent problem sets, with the input length constant to the token.

It does not establish that depth is "complexity" in general. Depth here is one
specific demand — the length of the chain of nested references that has to be
unfolded to write the answer — and on a translation task rather than a reasoning
task. That is narrower than the word "complexity" suggests, and worth stating
plainly rather than letting it slide.

It also says nothing yet about what is happening inside the model. Accuracy
falling is a behavioural fact; whether the model's internal state carries the
depth of the problem it is reading is the next measurement.

## Next

`depth-in-activations.py`, in this folder, reads the model's internal
activations on these same problems — the same seed draws the same ones, so every
activation can be matched to whether that problem was answered correctly — and
asks three things at each layer: whether the depths line up in order along a
single direction, whether depth can be read off one problem's activations better
than it can be read off plain word counts of the same text, and whether the
activations predict which individual problems the model will get wrong. It needs
no generation, so it costs minutes rather than hours.

The word-count comparison is the one that matters. There is a structural reason
to expect it to land near the 25% chance rate: in a description with *k* steps,
exactly *k* − 1 intermediate results get consumed by a later step no matter how
the chain is arranged, so every problem in a set mentions the same words the
same number of times, and only *which* step refers to *which* changes. If that
comparison comes out well above chance anyway, then something in the surface
does track depth after all, and only the margin above it is evidence about the
model.
