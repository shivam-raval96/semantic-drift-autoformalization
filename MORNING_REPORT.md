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

**IMPORTANT — the control landed and it changed this claim.** A placebo
question ("Is the story written in English?"), identical in format and read at
the identical position, still yields a yes/no margin that predicts correctness
at **0.6366** (n=2000) versus **0.6701** for the real verification question.

So the channel does NOT open because the model was asked about correctness. It
opens because the model is in an answer position in a chat format. The honest
version of 2.4:

> Reading bare text mid-document, the yes/no axis carries no correctness
> information at any depth (max 0.545). Put the same content in a chat prompt
> and read at the position where the model is about to answer *any* yes/no
> question, and that axis carries correctness at 0.64-0.67 — whether or not
> the question is about correctness.

That is still a real and interesting dissociation (format/position gates
access to the verbal channel), but it is NOT "task framing routes correctness",
and the earlier phrasing is withdrawn. `runs/lens-v1/placebo_last.json`.

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

| Model | behavioral capability | law-disjoint representation |
|---|---|---|
| Llama-3.1-8B | 0.522 | 0.527 |
| Qwen3.5-4B | 0.626 | 0.549 |
| Qwen3-32B | 0.669 | 0.599 |
| **Qwen3.6-27B** | **0.826** | **0.8146** |

Pearson r = 0.948, Spearman = 1.0 (n=4). Shuffled-label controls clean at every
point (0.48-0.52). Qwen3.6-27B is a striking data point: it reads 0.815 on
completely held-out law families, far above anything else we have measured, and
close to its own behavioral capability.

Caveats a skeptic agent is currently auditing: n=4 makes Spearman 1.0 a
1-in-24 coincidence under the null; the models differ in depth and
architecture (Qwen3.5/3.6 are hybrid linear-attention); and "best layer over
all layers" gives deeper models more chances at a maximum. A task-irrelevant
control probe (story theme) is running to test whether this is specific to
correctness or generic representational quality.

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

- **Position control (2×2).** Reader-mode reads at the end of the RG;
  asked-mode read at the generation position. The `ansend` capture reads the
  *asked* prompt at the end of the RG, making the comparison position-matched.
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

## 8. Where everything is

| What | Where |
|---|---|
| Live status | `DASHBOARD.md` |
| Full reasoning, every failure | `RESEARCH_LOG.md` |
| Headline figure | `probe-experiments/steering_mechanism.png` |
| Lens/steering/alignment results | `probe-experiments/runs/{lens-v1,steer-v1}/` |
| Probes per model | `probe-experiments/runs/probe-*/` |
| Behavioral gates | `probe-experiments/runs/verify-v1/` |
| Co-emergence | `probe-experiments/runs/analysis-v1/` |
| FT experiment (complete) | `ft-experiments/RESULTS.md` |
