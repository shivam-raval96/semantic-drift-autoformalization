You will read a short description and translate what it says into a small
formal notation. The notation is defined completely on this page — you need
no outside knowledge, and you should not use any other notation you know.

## What the description contains

The description concerns a collection of objects and an operation that
combines two objects into one. The operation takes a **first input** and a
**second input**, and the order of the inputs matters: applying it to "this
and that" is not assumed to give the same result as applying it to "that
and this".

The description has two parts:

1. A regularity that is assumed to always hold, introduced by "Suppose the
   following always holds": for every choice of the named objects, two
   described results are always equal.
2. A closing question, introduced by "Now consider the following question",
   asking whether a second regularity must also always hold.

Each application of the operation is introduced as its own step, and its
result is given a name: "apply the operation to x as its first input and y
as its second input, and call the result Value 1". A later step may use a
named result as an input, so a name like Value 2 can stand for a compound
expression built out of earlier steps. Each Value name stands for exactly
the expression its defining step describes — no more, no less.

## Your notation

- An **expression** is either an object's letter name, or
  `op(first, second)`, where `first` and `second` are themselves
  expressions. `op(...)` stands for one application of the described
  operation to its two inputs, in order.
- An **equation** is `expression = expression`, meaning the two sides are
  always equal however the objects are chosen.

Your answer is exactly two lines:

```
ASSUME: <equation for the regularity assumed to always hold>
ASK: <equation for the regularity the closing question asks about>
```

## Rules

1. Reuse the description's own letter names for the objects.
2. What the description calls the operation's **first input** is the first
   argument of `op`, and the **second input** is the second argument —
   `op(first input, second input)`, everywhere in both lines.
3. The named results (Value 1, Value 2, ...) are shorthand, not objects:
   replace each one by the expression its defining step describes, nesting
   `op(...)` as deep as needed. No Value name may appear in your two lines.
4. If one side of an equality is just an object's name, that side of the
   equation is just that letter.
5. The assumption and the question each quantify over their objects afresh
   and each restarts its Value numbering; translate each on its own terms
   even though the same letters and Value names reappear.
6. End your response with exactly the two lines, nothing after them.

## Worked example (a different description)

> Consider a collection of objects together with an operation that combines
> two objects into one. The operation takes a first input and a second
> input, and the order of the inputs matters.
>
> Suppose the following always holds. For every choice of objects x and y,
> apply the operation to x as its first input and y as its second input,
> and call the result Value 1; then apply the operation to x as its first
> input and Value 1 as its second input, and call the result Value 2. Then
> Value 2 is always equal to y.
>
> Now consider the following question. For every choice of objects x and y,
> apply the operation to x as its first input and y as its second input,
> and call the result Value 1; then apply the operation to y as its first
> input and x as its second input, and call the result Value 2. Does it
> follow that Value 1 is always equal to Value 2?

In the assumption, Value 1 is `op(x, y)`; Value 2 applies the operation to
x as its first input and Value 1 as its second input, so unfolding Value 1
it is `op(x, op(x, y))`; and the assumption says Value 2 always equals y
alone. In the question, Value 1 is `op(x, y)` and Value 2 is `op(y, x)` —
one application in each input order:

```
ASSUME: op(x, op(x, y)) = y
ASK: op(x, y) = op(y, x)
```

## The description

{story}

Now translate this description. End your response with the two lines,
`ASSUME:` and `ASK:`, in the notation defined above.
