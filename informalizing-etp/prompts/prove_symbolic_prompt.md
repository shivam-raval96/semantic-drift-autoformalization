You will read a two-line problem in a small formal notation and decide its
answer. The notation is defined completely on this page — you need no
outside knowledge, and the two lines are all that is given.

## The notation

The problem concerns a collection of objects and an operation that
combines two objects into one. The operation takes a first input and a
second input, and the order of the inputs matters.

- An **expression** is either a letter (x, y, z, w, u, v), standing for an
  object, or `op(first, second)`, where `first` and `second` are
  themselves expressions. `op(...)` stands for one application of the
  operation to its two inputs, in order.
- An **equation** is `expression = expression`. It asserts that the two
  sides are equal for *every* choice of objects for its letters — each
  equation quantifies over its letters afresh.

The problem is exactly two lines:

```
ASSUME: <an equation that is assumed to always hold>
ASK: <an equation whose necessity is in question>
```

## Your task

The ASSUME equation is everything you may assume. Beyond it, nothing is
known: the collection may be of any size — finite or infinite — and the
operation may act in any way at all, so long as the ASSUME equation holds
for every choice of objects. Your task is to decide whether the ASK
equation follows from the ASSUME equation alone.

- Answer **True** if the ASK equation must hold in *every*
  collection-and-operation satisfying the ASSUME equation — that is, the
  assumption forces it, with no exceptions possible.
- Answer **False** if some collection and operation satisfy the ASSUME
  equation completely and still violate the ASK equation for some choice
  of objects.

Exactly one of the two is the case; work it out rather than guessing. To
establish True, derive the ASK equation from the ASSUME equation. To
establish False, describe a concrete counterexample — a collection and an
operation table where ASSUME holds but ASK fails; a small finite
collection often suffices, though some failures need larger or infinite
ones.

## How to answer

Reason as much as you need, then end your response with exactly one line:

```
ANSWER: True
```

or

```
ANSWER: False
```

Nothing may follow that line.

## Worked example — answer True (a different problem)

```
ASSUME: op(x, y) = y
ASK: op(op(x, y), z) = z
```

The assumption says every application returns its second input. The ASK
side `op(op(x, y), z)` is one application whose second input is z — the
assumption applies with first input `op(x, y)` and second input z — so it
equals z, whatever x and y are. The equation is forced:

ANSWER: True

## Worked example — answer False (a different problem)

```
ASSUME: op(x, x) = x
ASK: op(x, y) = op(y, x)
```

Consider a two-object collection {a, b} with the rule "the result is
always the first input". The assumption holds there: `op(x, x)` is x for
either object. But `op(a, b)` is a while `op(b, a)` is b, and a and b are
different objects, so the ASK equation fails. A structure can satisfy the
assumption and violate the equation:

ANSWER: False

## The problem

{story}

Now decide this problem. End your response with the single line
`ANSWER: True` or `ANSWER: False`.
