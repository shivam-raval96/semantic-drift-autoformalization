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
2. A closing question, introduced by "Does it follow that", asking whether
   a second regularity must also always hold.

Described results may be nested: an input of the operation can itself be
"the result of applying the operation to ...". Read these inside out — each
"the result of applying the operation to A as its first input and B as its
second input" is one application of the operation, whose inputs A and B may
themselves be such results.

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
3. Translate each nested "the result of applying the operation to ..."
   phrase into a nested `op(...)` expression, nesting as deep as needed.
4. If one side of an equality is just an object's name, that side of the
   equation is just that letter.
5. The assumption and the question each quantify over their objects afresh;
   translate each on its own terms even though the same letters reappear.
6. End your response with exactly the two lines, nothing after them.

## Worked example (a different description)

> Consider a collection of objects together with an operation that combines
> two objects into one. The operation takes a first input and a second
> input, and the order of the inputs matters.
>
> Suppose the following always holds: for every choice of objects x and y,
> the result of applying the operation to x as its first input and the
> result of applying the operation to x as its first input and y as its
> second input as its second input is always equal to y.
>
> Does it follow that for every choice of objects x and y, the result of
> applying the operation to x as its first input and y as its second input
> is always equal to the result of applying the operation to y as its first
> input and x as its second input?

In the assumption, the outer application has `x` as its first input; its
second input is itself "the result of applying the operation to x as its
first input and y as its second input", i.e. `op(x, y)`. So the left side
is `op(x, op(x, y))`, and it is always equal to `y` alone. The question
compares one application in each input order:

```
ASSUME: op(x, op(x, y)) = y
ASK: op(x, y) = op(y, x)
```

## The description

{story}

Now translate this description. End your response with the two lines,
`ASSUME:` and `ASK:`, in the notation defined above.
