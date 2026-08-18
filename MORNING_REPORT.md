# Morning report — night of 2026-08-18/19

Read this first. `DASHBOARD.md` = live status, `RESEARCH_LOG.md` = full
reasoning and every failure. All numbers trace to committed JSON under
`probe-experiments/runs/`.

---

## 1. Executive summary

We went in with a null: correctness is linearly readable in Qwen3-32B
(law-disjoint 0.599 vs 0.503 floor) but steering that direction did nothing.
A null like that is uninterpretable on its own — it could mean the direction is
inert, or that our steering code simply doesn't work.

**Tonight that null became a mechanism, with the controls to prove it.**

The correctness direction is *orthogonal to the model's verbalization
pathway* and *verbally silent*: by the model's own causal readout, pushing
along it makes the model disposed to say nothing in particular — no more than
a random direction does. Meanwhile the same steering harness, given the
J-lens yes/no direction, swings the margin by 24 logits and flips the model's
answer from 6% yes to 100% yes.

**The one-sentence version:** *Qwen3-32B knows whether a formalization is
correct, and encodes that knowledge in a direction it has no way to say.*

We also **withdrew two of our own claims** under adversarial review and
**caught a real bug** in the method with a validation gate built specifically
to try to break it.

---

## 2. Most important findings

### 2.1 The correctness direction is verbally silent (HIGH confidence)

`runs/lens-v1/direction_vocabulary.json`, figure `steering_mechanism.png`

The J-lens readout of a direction gives, per vocabulary token, how much that
direction raises the model's disposition to eventually emit that token.

| direction | max readout | top tokens |
|---|---|---|
| J-lens yes/no (**positive control**) | **0.9503** | `yes`, ` yes`, `_YES`, ` Yes` … |
| raw unembedding yes/no | 1.0397 | `yes`, ` Yes`, ` oui`, ` yeah` … |
| **our correctness direction** | **0.1143** | unrelated vocabulary |
| random direction (**null**) | 0.1121 | unrelated vocabulary |

The method demonstrably works (positive control reads its own tokens at 8× the
noise floor). Our direction reads out at exactly the random-direction level.

### 2.2 Steering: the harness works; our direction is inert (HIGH confidence)

`runs/steer-v1/qwen3-32b-jlens.json` vs `qwen3-32b.json`

| direction, α=+0.5 | mean margin | Δ | yes-rate |
|---|---|---|---|
| baseline | −2.618 | — | 6% |
| **J-lens (positive control)** | **+21.540** | **+24.16** | **100%** |
| our correctness direction | −2.685 | −0.07 | 6.7% |
| random, norm-matched | −3.399 | −0.78 | 3.7% |

Our direction moves the model *less than random noise*. Dose–response for the
J-lens direction is monotone across α ∈ [−1, +1]: −45.4 → +22.4.

Nuance: J-lens steering shifts the model's **bias** (says yes to everything)
while barely denting **discrimination** (AUROC 0.669 → 0.647). The verbal axis
controls what it says, not what it knows.

### 2.3 Geometry: orthogonal to the verbalization axis (HIGH confidence)

`runs/lens-v1/jspace_alignment.json` — cosine(probe direction, J-lens yes/no
direction) = **0.011**, versus a random-direction |cos| p95 of **0.027** in the
same 5120-dim space. **This holds at every layer, not just the one we probed:
across all 63 lens blocks the mean |cosine| is 0.0065 and the maximum is
0.0189 — zero of 63 exceed the random-direction p95.** Independently, the J-lens direction used *as* a
correctness classifier reaches only 0.556–0.562: these are two different
directions carrying different information.

### 2.4 Reader vs asked: the yes/no channel is silent during passive reading
(HIGH confidence for the reader-mode half)

`runs/lens-v1/margin_lens.json`, `margin_lens_asked.json`

Reading story+RG as bare text, the model's own yes/no readout is flat at every
one of 65 layers (max 0.545) while a supervised probe on the *same activations
at the same position* reads 0.62–0.64. Under the verification prompt the same
readout climbs to 0.686 and ends at 0.6701 — which matches the independently
measured behavioral gate (0.6694) to 0.0007.

**The placebo control, and then a correction to my own correction.** A placebo
question ("Is the story written in English?") reaches **0.6366** at the final
layer versus **0.6701** for the real question — which initially looked like the
whole effect was format, not content. But the layer-resolved comparison shows
that was an artifact of looking at one layer:

Both conditions at **n=2000** (matched):

| layer | real question | placebo | difference |
|---|---|---|---|
| 45 | 0.5037 | 0.5028 | +0.001 |
| 48 | 0.5709 | 0.5120 | +0.059 |
| 50 | 0.6685 | 0.5287 | +0.140 |
| **53** | **0.6889** | **0.4831** | **+0.206** |
| 54 | — | — | **+0.2115 (max gap)** |
| 56 | 0.6777 | 0.4929 | +0.185 |
| 64 (final) | 0.6677 | 0.6366 | +0.031 |

**Two mechanisms, separable by depth.** (1) Question-specific routing is real
at blocks ~48-57, where the real question yields 0.67-0.69 and the placebo sits
at chance — a +0.21 gap, roughly **11 sigma** at matched n=2000. (2) A weaker
format/position effect appears only in the final blocks, where the placebo
catches up (0.6366) and the remaining gap (+0.031) is ~1.7 sigma. The honest
version of 2.4:

> Reading bare text mid-document, the yes/no axis carries no correctness
> information at any depth (max 0.545). Asking about correctness routes it onto
> that axis from about two-thirds depth (blocks ~48-57), where an unrelated
> yes/no question leaves the axis at chance. A weaker, question-independent
> effect appears only in the final blocks, where being in an answer position
> surfaces correctness whatever was asked.

Both curves are now n=2000; the earlier n=300 values were within 0.004 of
these, so nothing hinged on sample size.
`runs/lens-v1/{asked_last_full,margin_lens_placebo}.json`.

### 2.4b REPLICATION on a second model, two independent lenses (HIGH confidence)

`runs/lens-v1/replicate_alignment_qwen3.6-27b.json`

Qwen3.6-27B has a far STRONGER correctness representation (law-disjoint
**0.821**) than Qwen3-32B (0.599) - so any "the direction is just weak/noisy"
objection is off the table. Yet:

| lens | cosine at matched layer | all-block mean \|cos\| | blocks exceeding random p95 |
|---|---|---|---|
| J-lens | +0.0127 | 0.0083 | **0 / 63** |
| R-lens (independent method) | +0.0151 | 0.0085 | **0 / 63** |

Same result on a different model, with lenses fitted by a different group
using a different recipe (n=25 pile prompts vs n=1000 wikitext), and with two
methodologically distinct lenses. The orthogonality is not an artifact of one
model or one lens.

### 2.5 The scaling is SPECIFIC to correctness, not generic (HIGH confidence)

`runs/analysis-v1/control_probe.json`

The obvious alternative to co-emergence is "bigger/newer models represent
everything better." Tested directly with a task-irrelevant label (the story's
theme) decoded from the same activations, same splits, depth-matched layers:

| Model | correctness (law-disjoint) | theme accuracy (chance 0.25) |
|---|---|---|
| Llama-3.1-8B | 0.533 | **1.000** |
| Qwen3.5-4B | 0.533 | **1.000** |
| Qwen3-32B | 0.599 | **1.000** |
| Qwen3.6-27B | **0.811** | **1.000** |

Every model, including the weakest, decodes the irrelevant property
perfectly. Only correctness scales with capability. The generic-quality
explanation is dead.

(These numbers use fixed fractional depths rather than best-of-all-layers, so
they also neutralize the "deeper models get more chances at a maximum"
concern.)

### 2.6 Capability and representation co-emerge (MODERATE-HIGH confidence)

`runs/analysis-v1/coemergence.json`

Authoritative numbers, pre-registered `mean` site, paired clustered bootstrap
over the 1000 problems, permutation null under the SAME law-disjoint grouping
(`runs/analysis-v1/probe_uncertainty.json`):

| Model | capability | representation | 95% CI | sigma above null |
|---|---|---|---|---|
| Llama-3.1-8B | 0.522 | 0.5204 | [0.511, 0.530] | 1.91 (NOT clearly above chance) |
| Qwen3.5-4B | 0.626 | 0.5490 | [0.537, 0.562] | 3.89 |
| Qwen3-32B | 0.669 | 0.5985 | [0.586, 0.611] | 8.50 |
| **Qwen3.6-27B** | **0.826** | **0.8146** | [0.801, 0.829] | **18.68** |

Paired adjacent differences, same problems for both models, all **P(>0) = 1.0**:
+0.0286 [0.016, 0.040], +0.0495 [0.035, 0.065], +0.2161 [0.199, 0.233].

Note the significance ladder is itself monotone (1.9 -> 3.9 -> 8.5 -> 18.7
sigma). The 8B is the honest boundary case: at the pre-registered site it is
1.91 sigma above its null, i.e. **at or barely above chance** - so "weakest",
never "absent".

Pearson r = 0.948, Spearman = 1.0 (n=4). Shuffled-label controls clean at every
point (0.48-0.52). Qwen3.6-27B is a striking data point: it reads 0.815 on
completely held-out law families, far above anything else we have measured, and
close to its own behavioral capability.

**A second skeptic agent audited this claim in depth, reproduced all four
numbers bit-exactly from the activations, and found real problems. Its
corrections are adopted:**

- **Report it as ORDINAL, not linear.** Spearman 1.0 at n=4 has an exact
  permutation P of 1/24 = 0.042 (best achievable at this n); Pearson r=0.948
  gives p=0.052. Worse, the relation is not one line: the slope through the
  three low-capability models under-predicts Qwen3.6-27B by 0.16 AUROC.
  "Pearson r = 0.95" is withdrawn; "ranks agree" is what the data supports.
- **The fold SDs we reported are NOT valid uncertainty.** The law-disjoint
  GroupKFold is pathological: fold 0 holds 1112 texts (all easy + some
  medium), fold 1 holds 666 (all hard/genform), folds 2-4 hold 74 each. It is
  effectively a TIER split, and its SD overstates estimator noise ~4x versus a
  permutation null. Replaced by a paired clustered bootstrap over the 1000
  problems: 0.527 [0.511,0.542], 0.549 [0.537,0.561], 0.599 [0.586,0.612],
  0.815 [0.800,0.829]. All adjacent pairs separate, but the Llama->Qwen3.5-4B
  gap (+0.022) collapses to ~0.004 under an equally defensible layer/site rule.
- **"Absent in the incapable 8B" is too strong.** Llama-3.1-8B's 0.527 is 2.8
  sigma above a within-pair label-flip null - small but real. Corrected
  wording: weakest, not absent.
- **The specificity control is STRONGER than we reported.** At full dimension
  theme decoding saturates at 1.000 for every model (a ceiling). At matched
  capacity (k principal components) Spearman(capability, theme) = **-0.4** -
  the most capable model has the WORST task-irrelevant decodability. Generic
  representational quality is not just uncontrolled-for, it runs the opposite
  way.
- **Survives:** capacity matching (random projection to 1024 dims, PCA-64
  preserve the ordering), parameter count (a 4B beats an 8B; a 27B beats a 32B
  at identical probe geometry), identical folds across models (verified
  byte-identical ids/labels/groups), and layer selection (reported values sit
  BELOW each model's layer-wise maximum, so no winner's curse).
- **Unresolved on the x-axis too:** Qwen3-32B - Qwen3.5-4B = +0.044 with CI
  [-0.004, +0.090]. One of the three adjacent ranks is a near coin flip.
- **Near-circularity to state explicitly:** x is the model's own yes/no
  readout and y is a probe of the same construct. What makes it non-trivial is
  that y is measured in READER mode (no question, no chat template) while x is
  asked mode.
- **"Co-emerges" does not mean "supports."** Our own steering result shows the
  direction is causally inert; no phrasing may imply it underlies the behavior.

**THE DECISIVE TEST RAN, AND IT REVISED THIS FINDING.** Llama-3.3-70B has
capability 0.629 - statistically identical to Qwen3.5-4B's 0.626 - at 17x the
size. Co-emergence predicted ~0.55; scale predicted 0.60+. **It came back
0.6337** - and was then independently replicated at 0.6337 by a separate
computation path (remote-on-volume vs local-on-download; see log entry 23).

| comparison | capability | representation |
|---|---|---|
| Qwen3.5-4B | 0.626 | 0.549 |
| **Llama-3.3-70B** | **0.629 (+0.003)** | **0.634 (+0.085)** |

The ordering also inverts once: the 70B (capability 0.629) out-reads Qwen3-32B
(capability 0.669), 0.634 vs 0.599. Spearman falls 1.0 (n=4) -> **0.90 (n=5)**.

**Revised claim:** representation strength is strongly associated with
capability but **not determined by it** - at matched capability, 17x the
parameters buys +0.085, and a less capable model can out-read a more capable
one. Both capability and scale contribute. The earlier "co-emerge" framing
overstated it.

This does not touch findings 2.1-2.4b (geometry, verbal silence, steering,
routing controls), which concern one model's direction and stand unchanged.
(Gemma-3-27B was excluded for float16 overflow - see 7a.)

---

## 3. What we withdrew (honesty ledger)

1. **"The model knows much more than it says" — WITHDRAWN at the answer
   position.** Our probe used problem-grouped CV (leaky: the same law appears
   on both sides) while the lens is fit-free. Under law-disjoint splits the
   gap collapses: probe 0.6708 vs model's own readout 0.6701. The honest —
   and more interesting — statement is the reverse: *when asked, the model's
   verbal readout captures essentially everything a linear probe can extract.*
2. **"Peaks at layer 53 then declines to the output" — DEAD.** The endpoint
   was a bug artifact. Corrected Δ = 0.0156, bootstrap 95% CI [−0.004, +0.036].
3. **"Task framing performs late-depth routing" — CONFOUNDED**, controls in
   flight (§5).

## 4. Failures and what caused them

- **Double-normalization bug (the important one).** HuggingFace overwrites the
  last hidden state with the *post-final-norm* value; our lens normalized it
  again. Caught by a validation gate that requires the lens to reproduce the
  model's true logits (it failed at 28.2 logits of error). Diagnosed with a
  discriminating test (no-norm error: 0.06), fixed, everything recomputed.
  After the fix, two independent measurement paths agree to 0.0007.
- Numpy cannot read bf16 → read via torch. `debian_slim` lacks git → apt git.
  Qwen3.6-27B nests weights under `model.language_model.*` → key autodetection.
  The Neuronpedia lens is a raw `fit()` checkpoint, not a saved lens →
  reconstructed the mean exactly as `jacobian_sum / n_done`.
- Modal volume read-after-write races corrupt large downloads → wrote a
  settle-verify-retry helper; all activation files are now zip-verified before
  any analysis touches them.

## 5. Controls still running (results will be appended)

- **Position control (2×2) — DONE, confound REFUTED.** Holding position fixed
  at the end of the RG, adding the question moves the channel from chance
  (0.499) to **0.6294**. Position adds a further increment (→0.689 at the
  generation position) but explains none of the base effect.
  `runs/lens-v1/asked_ansend.json`
- **Placebo question — DONE, and it falsified the routing claim** (see 2.4).
  0.6366 with an unrelated question vs 0.6701 with the real one.
- **Specificity control.** Does a probe for a task-irrelevant label (story
  theme) also scale with capability? If yes, co-emergence is generic.

## 6. What this means for the paper

The project now has three linked results, each mechanically grounded:

1. **Grammar-only fine-tuning is a behavioral override** — it perfects syntax,
   adds no semantics, and destroys existing translation ability in a model that
   had it (34% → 3–4%).
2. **Correctness is linearly represented, and representation tracks
   capability** — absent in models that cannot do the task, present in those
   that can.
3. **That representation is verbally silent and causally inert** — with a
   positive control proving the measurement apparatus works.

Together they say something sharper than any one alone: *what a model knows,
what it can say, and what training can install are three different things, and
we can now measure the gaps between them mechanically.*

## 7. Highest-value next experiments

1. Replicate §2.1–2.3 on Qwen3.6-27B (free J+R lenses, best verifier at 0.826).
   Note its lenses use n=25 prompts vs Neuronpedia's n=1000 — weaker instrument.
2. If the position/placebo controls survive: characterize the recruitment
   operation (which heads/layers move the information into the channel).
3. FT v2 with the task in-distribution (story→RG pairs), measured on all three
   axes: behavior, representation, and verbal accessibility.

## 7a. One model excluded, with cause

**Gemma-3-27B is excluded from the co-emergence analysis.** Its activations
contain 21,075-25,285 non-finite values caused by float16 overflow at capture
(Gemma-3 carries unusually large activation magnitudes; float16 caps at
65504). This is a measurement failure on our side, not a result about Gemma.
A hard non-finite check now runs at capture time, and including Gemma later
just needs a float32 recapture. All other models were verified clean.

## 7b. Self-audit of this report

Every headline number in this document was re-read programmatically from its
committed JSON artifact and checked (`21/21 verified`). Checks covered: the
vocabulary readout values and that the positive control's top token is
literally "yes"; all steering margins and yes-rates for both directions plus
the random control; the alignment cosine and its random baseline; the
Qwen3.6-27B replication (0/63 for both lenses) and its probe AUROC; the
bootstrap/permutation statistics for every model; the asked-vs-placebo curves
at the quoted layers and the maximum gap; and both validation-gate numbers.
Three initial "mismatches" were my audit demanding exact equality against
correctly-rounded values (21.5402 vs 21.540, -2.6846 vs -2.685, -2.6179 vs
-2.618) - resolved, not errors.

No number in this report was typed from memory.

## 8. Where everything is

| What | Where |
|---|---|
| Live status | `DASHBOARD.md` |
| Full reasoning, every failure | `RESEARCH_LOG.md` |
| Headline figure (mechanism) | `probe-experiments/steering_mechanism.png` |
| Headline figure (routing by depth) | `probe-experiments/routing_by_depth.png` |
| Lens/steering/alignment results | `probe-experiments/runs/{lens-v1,steer-v1}/` |
| Probes per model | `probe-experiments/runs/probe-*/` |
| Behavioral gates | `probe-experiments/runs/verify-v1/` |
| Co-emergence | `probe-experiments/runs/analysis-v1/` |
| FT experiment (complete) | `ft-experiments/RESULTS.md` |
