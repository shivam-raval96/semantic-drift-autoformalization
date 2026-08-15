# Depth as a difficulty variable, with text length held fixed

*Status: dataset design built and verified; the experiment is written but has
not been run.*

## The problem this solves

Every claim this project has made about problem complexity is confounded with
how much text the problem takes to write down. The dataset-geometry experiment
measured it: the number of words in a problem correlates with the number of
operations in its underlying law at 0.928 for the story rendering and 1.000 for
Rigid Grammar, the terse formal rendering. So "the model's activations separate
simple from complex problems" and "the model's activations separate short from
long inputs" have been the same statement, and every later complexity result
inherits that. Shivam picked out the complexity separation as the interesting
finding, which makes closing this the gate on the whole line of work.

The question is whether we can get an ordered difficulty variable that text
length cannot explain.

## Why the two obvious fixes fail

**Statistically removing length** — regressing word count out of the
activations and analysing what is left — is close to useless here. Within Rigid
Grammar the correlation is 1.000: length and operation count are one variable
with two names, so removing length removes the complexity signal along with it.
The result would be a null whether or not the model represents complexity, which
means it would tell us nothing.

**Selecting length-matched problems** fails too, and not for want of a larger
pool. Every renderer emits exactly one fixed-length step per operation — the
literal renderer's step is the same roughly 20-word sentence every time, the
story renderer's a shorter fixed one — so a law's word count is essentially a
straight-line function of its operation count. There is no such thing as a long
two-operation problem to select as a counterweight, because a two-operation law
renders as two steps and that is the entire text. The correlation is arithmetic,
not an artifact of sampling.

## What works: vary depth, pin everything else

Depth is the height of the term tree — how many steps have to be evaluated one
after another before the last one can be computed. It is free at a fixed
operation count. Two laws from the pool, both seven operations, four variables,
with a bare left side:

```
x = (x ◇ (y ◇ x)) ◇ (z ◇ ((z ◇ x) ◇ (y ◇ w)))      depth 4
x = (y ◇ (y ◇ (z ◇ (z ◇ ((w ◇ z) ◇ w))))) ◇ z      depth 7
```

Both render to seven definition steps and to exactly 399 words of literal
description. The first splits into independent branches that can be worked on in
any order; the second is a single chain where every step consumes the previous
result.

Making this hold exactly took two corrections beyond "hold the operation count
fixed":

1. **The side split has to be pinned.** Depth is a property of one *side* of the
   equation, so a seven-operation equation only reaches depth 7 by putting all
   seven operations on one side and leaving the other a bare variable. Sampling
   on operation count and depth alone therefore drifts toward bare-sided laws as
   depth rises, and a bare side renders shorter. Measured on the story
   rendering, word count correlated with depth at −0.645.
2. **The variable count has to be pinned.** It sets the length of every
   quantifier clause, and the pool's mix of variable counts shifts slightly with
   depth. After fixing the split, word count correlated with variable count at
   +0.846, leaving a residual correlation with depth of +0.28.

So a **cell** is (minor side operations, major side operations, variables,
depth), and a **family** is a set of cells differing only in depth.

## The result

Two families, four depths each, 40 pairs per cell. Word counts per surface form:

| family | depths | operations per equation | literal | graft | paint | signal | tea | Rigid Grammar |
|---|---|---|---|---|---|---|---|---|
| `0:6:3` | 3–6 | 6 | 351 | 246 | 246 | 242 | 280 | 20 |
| `0:7:4` | 4–7 | 7 | 399 | 281 | 281 | 277 | 321 | 22 |

Those counts are exact and identical at every depth within a family — standard
deviation 0.0 across all 320 pairs in all six surface forms. Not a reduced
correlation; no variance at all. Across families length still tracks operation
count at +1.000, which is the original confound behaving exactly as first
measured, so all depth analysis has to stay inside a family.

Two families means every result carries a built-in replication: an effect that
shows up in one and not the other is tied to that family's operation or variable
count, not to depth.

## Using it

```sh
cd mech-interp-experiments
python3 -m shared.dataset --list-cells        # what the law pool can fill
python3 tests/check_length_balance.py         # exits non-zero if length drifts
python3 -m unittest discover -s tests -t .
```

```python
from shared.dataset import build_pool, parse_cells, sample_depth_balanced

samples = sample_depth_balanced(
    build_pool(), 40, seed=0,
    cells=parse_cells("0:6:3:3-6,0:7:4:4-7"), form="literal",
)
# each sample carries the usual complexity tags plus shape = [minor, major, vars, depth]
```

`build_pool()` synthesizes 8000 laws with the vendored generator rather than
reading a file, so nothing needs committing or uploading. The published ETP law
list stops at four operations per equation, well below what these families need.

Nothing under `informalizing-etp/` is touched; it is imported read-only.

## The experiment

`depth-at-fixed-length.py` in this folder. At a fixed thinking-token budget,
does accuracy fall as depth rises? Depth is serial chain length, and the earlier
budget sweep found accuracy is a smooth function of how many reasoning tokens
the model is allowed, which says the bottleneck is serial computation — so this
is a real prediction rather than a fishing expedition. The full statement of
what each possible outcome would mean is in the file's header comment, written
before the run.

Two details worth knowing before reading its output:

- **The depth comparison is unpaired and cannot be otherwise.** A law cannot be
  both depth 4 and depth 7, so no problem is shared between depth levels and
  McNemar's test does not apply. Differences across depth get Newcombe
  intervals, which are wide; that is why the default is 200 pairs per cell
  rather than the ~50 used where pairing was available. Comparisons across
  *budget* at a single depth do share their problems and are reported paired.
- **The run checks its own premise.** Word count is constant by construction,
  but the model reads tokens, not words. Every prompt's tokenized length is
  recorded and the analysis reports loudly if it varies within a family.

The geometric follow-up — do activations order by depth, and at which layers —
comes second, and only makes the same claim if the behavioural arm shows depth
costs the model something. If accuracy is flat across depth, a depth axis found
in the activations would need a reading other than "the model represents
difficulty".

## Known limits

- Four depth levels is a short ladder. Wider families exist further up the
  inventory (a ten-operation major side reaches depth 10) at the cost of much
  smaller pools.
- Both families have a bare minor side, i.e. laws of the form `x = <term>`. That
  restriction is what buys the widest depth range; a balanced-split family would
  test whether any finding survives outside it.
- Depth and "difficulty" are not the same thing by definition. Depth is one
  concrete way a problem can demand serial work, and it is the one that can be
  varied here at constant length.
