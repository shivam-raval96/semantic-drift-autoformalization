# Experiment 1 — Contrastive steering, story → literal-NL: findings

Notebook: [`01-contrastive-steering-story-to-literal.ipynb`](01-contrastive-steering-story-to-literal.ipynb) ·
Artifacts: `exp01-outputs/` (`summary.json`, `budget-summary.json`, per-condition `.jsonl`,
`exp01-results.png`, `exp01-budget-results.png`)

**Status:** both parts run to completion. Model `Qwen/Qwen3-4B`, seed 0, 53 fit / 52 eval pairs.

**Question (Part 1).** Can a single injected direction close the no-think accuracy gap on
story → Rigid Grammar formalization?
**Question (Part 2, added after Part 1's null).** If the bottleneck is serial compute instead, is
accuracy a function of the thinking-token budget?

## Headline

Part 1 is a **clean, well-controlled null**: the story→literal direction is real and coherent, and
moving along it changes task accuracy by exactly nothing at any dose the model survives. Part 2 is
the positive result that replaces it: **accuracy is a smooth function of thinking-token budget,
and the story framing costs roughly a 2× budget multiplier over its literal-NL twin.** The
bottleneck is serial compute, not representation style.

## Part 1 — Steering (null)

### Baselines (52 held-out pairs)

| condition | correct | note |
|---|---|---|
| story, no-think | 9.6% | floor |
| literal-NL, no-think | 19.2% | the realistic target for a form intervention |
| story, thinking | 90.4% | ceiling |

The 9.6% → 90.4% gap is large, so there was plenty to close. But literal-NL at 19.2% reframes what
"closing it" could ever have meant: perfectly converting the story representation into the literal
one buys ~10 points, not ~80.

### The vector is real

Mean-difference vector (last-token, literal minus story) per layer:

| layer | ‖v‖ | median ‖h‖ | relative | mean pairwise cos |
|---|---|---|---|---|
| 6 | 7.3 | 22.3 | 0.33 | 0.912 |
| 12 | 18.4 | 34.9 | 0.53 | 0.806 |
| 18 | 17.4 | 47.7 | 0.37 | 0.628 |
| 24 | 30.7 | 92.6 | 0.33 | 0.686 |
| 30 | 69.1 | 234.2 | 0.30 | 0.763 |

Per-pair differences cohere at cos 0.63–0.91, so this is a consistent direction, not an average of
noise.

### Steering does nothing survivable

Coarse sweep, correct rate on 20 eval pairs (unsteered floor on that subset: 20%):

| layer | α=1 | α=2 | α=4 | α=8 | α=16 |
|---|---|---|---|---|---|
| 6 | 10% | 0% | 0% | 0% | 0% |
| 12 | 0% | 0% | 0% | 0% | 0% |
| 18 | 5% | 0% | 0% | 0% | 0% |
| 24 | 20% | 0% | 0% | 0% | 0% |
| 30 | 15% | 15% | 0% | 0% | 0% |

At α ≥ 2 generation degenerates into repetition loops — 100% `unparseable` at essentially every
layer. On the full eval pile the best cell (layer 24, α = 1) scored **9.6%, exactly the floor**,
and the norm-matched random control also scored 9.6%. **Gap closed: 0.0%.**

So: α = 1 is indistinguishable from doing nothing, α ≥ 2 breaks the model, and there is no window
in between. The `unparseable` canary behaved exactly as designed — nothing was converted from
`wrong` to `correct`, only from `wrong` to broken.

### α = 1 is not "too weak", it is ineffective

Diffing the cached generations against the unsteered baseline (same seed and batching, so
identical logits would give identical text):

| condition | text differs from baseline | verdict changed |
|---|---|---|
| steer L24 α=1 | 52/52 | 5/52 |
| random control L24 α=1 | 45/52 | 8/52 |
| steer L30 α=1 | 33/52 | 3/52 |
| steer L30 α=2 | 43/52 | 14/52 |

The logits move on every example, so the smallest dose does register — it just moves nothing
directional, and changes *fewer* verdicts than a matched-norm random direction does. That rules
out a sweet spot below α = 1, and means the interesting unexplored variable is the injection site
rather than the dose.

### Magnitudes, for reference

α multiplies the **raw, un-normalized** mean-difference vector, so the smallest dose tested is
already 30–53% of the median residual-stream norm (α=1), α=2 is comparable to the residual itself,
and α≥4 is 1.2–8× larger than the thing it is added to. The hook also adds `α · v` at *every* token
position, during prefill as well as decode, so the whole context is shifted — a much heavier
intervention than the steering-vector literature usually applies, and the likely reason coherence
dies at only ~0.66× the residual norm.

## Part 2 — Thinking-budget dose–response (positive)

Budget forcing: at most B thinking tokens, `</think>` spliced in if the budget ran out, then the
answer is generated from whatever reasoning fit.

| B | story | literal-NL |
|---|---|---|
| 0 (baseline) | 9.6% | 19.2% |
| 32 | 1.9% | 17.3% |
| 64 | 1.9% | 19.2% |
| 128 | 1.9% | 17.3% |
| 256 | 11.5% | 36.5% |
| 512 | 38.5% | 96.2% |
| 1024 | 78.8% | 100% |
| ∞ | 90.4% | — |

Three things worth remembering:

1. **Small budgets are actively harmful.** Story accuracy at B = 32–128 (1.9%) is *below* the
   B = 0 floor (9.6%). A cut-off trace leaves a half-finished draft, and `checkform` grades the
   last `ASSUME:`/`ASK:` lines it finds — so the draft gets graded instead of an answer.
2. **The story penalty is a budget multiplier of about 2×.** Literal-NL reaches 96% at B = 512
   where story needs B = 1024 to reach 79%. That offset is a fairly direct measurement of the
   token cost of de-narrativizing.
3. **The story arm is not saturated at B = 1024.** Only 4/52 traces closed naturally and mean
   usage was 1015/1024 tokens, so 79% is still budget-limited and has not met the 90.4% unlimited
   ceiling. The literal arm did saturate (47/52 closing naturally, 736 tokens mean).

Complexity split on the story arm: simple problems (ops ≤ 5) reach 95% at B = 1024 while complex
ones (ops > 5) reach 67%, and the two curves separate from B = 256 onward.

## What this establishes

- The story→literal direction is decodable and causally inert for this task at survivable doses.
  A pure "represent this abstractly" intervention cannot be the mechanism.
- What separates 10% from 90% is room to write intermediates out, and the story framing raises the
  price of that in tokens rather than making the problem unsolvable.
- Cross-check: experiment 3's independent budget sweep on prompted stories gives 15.4% at B = 0,
  25% at 256, 44.2% at 512 — consistent with the story column here.

## Parts 3 and 4 — written, not yet run

Two follow-ups are now in the notebook and awaiting a Colab run.

**Part 3 — injection site instead of dose.** Sweeps three local patch sites against
`RESTRICTED_ALPHAS` up to 32: `story_end` (the single token the vector was actually measured at),
`story_span` (the story tokens only, leaving the shared instruction boilerplate untouched), and
`decode_only` (generated tokens only). For the prompt-side sites the patch is off during decoding,
so the vector reaches generation only through the KV cache of the story positions — the sharp
version of "does the story's representation control what gets formalized?". Adds a **negated
vector** control alongside the random one: a symmetric response to +v and −v would mean we are
measuring perturbation magnitude, not direction. The best cell is picked as the largest dose that
stays coherent, not the highest raw accuracy.

**Part 4 — steering × budget.** Runs the steered and random-control curves at B = 128/256/512/1024
against Part 2's cached unsteered rows, with the patch live in both phases of budget forcing. B = 0
comes free from Part 3's full-eval run of the same cell, so the curve is anchored at both ends.
Three accounts get separated by the *shape* of the steered-minus-unsteered delta across B:

- **inert** — flat at zero everywhere, and Part 1's null simply extends;
- **a substitute for reasoning tokens** — a mid-range hump, i.e. the dose–response curve shifted
  **left**, reaching the same accuracy at smaller B. Part 2 sized what that is worth: literal-NL
  needs about half the budget of the story arm;
- **a premise the model has to use** — the delta *grows* with B, because an injected abstraction is
  worthless without a trace long enough to exploit it. Opposite ordering from the previous case.

Two things make the comparison readable that the earlier draft lacked. Arms are graded on the same
`pair_id`s, so differences are read paired (McNemar) rather than as two independent rates — on 52
examples that is the difference between resolving a 10-point shift and not. And the shape is read
off the *share of available headroom* captured, `(steered − unsteered) / (1 − unsteered)`, because
the raw delta is not comparable across budgets: the unsteered story arm is at 79% by B = 1024, so
the largest gap arithmetically available there is +21 points against +88 at B = 256 — a metric that
collapses precisely where the third account predicts an effect.

Working against that third account: a long trace also lets the model attend back to the unpatched
story text and talk itself out of the injection, which would suppress the same signal.

## Open items / next steps

- Extend the story sweep past B = 1024 to find where it saturates, since 79% is still climbing.
  Part 4's top budget inherits this: if the steered benefit is still growing at B = 1024, the
  unsteered arm not having saturated is a live alternative explanation.
- Re-check whether the sub-floor dip at B = 32–128 is a grading artifact by scoring only text
  after the spliced `</think>`.
- If Parts 3 and 4 are also null, the B = 0 arm needs more eval pairs before another attempt: the
  maximum effect it could show is ~10 points against a ±5-point standard error on 52 examples, so
  it cannot cleanly resolve even its own best case.
- Same sweep on a larger model, and the SAE version of the contrast (experiment 5).
