# Research log

Chronological, hypothesis-first. Each entry: question, hypothesis, what was
done and why, result, whether the hypothesis survived, what it changes.
Dashboard (live status) is `DASHBOARD.md`. Numbers trace to committed run
records under `probe-experiments/runs/` and `ft-experiments/runs/`.

---

## Where we stood entering tonight

Established over the previous rounds (all committed, fact-checked):

- **Grammar-only fine-tuning is a behavioral override, not a skill injection.**
  Syntax perfected (unparseable 45% -> 0% on literal/two-stage), zero
  correctness gain on Llama-3.1-8B, and on Qwen3-32B it *destroyed* existing
  translation ability (34% -> 3-4% pooled). `ft-experiments/RESULTS.md`
- **Correctness is linearly represented in a capable model but not in an
  incapable one.** Probe law-disjoint AUROC: Qwen3-32B 0.599 vs lexical floor
  0.503; Llama-3.1-8B 0.520 (= floor). `probe-experiments/RESULTS.md`
- **The direction is causally inert.** Steering at +-1.0x residual norm, with
  negated and norm-matched random controls, moved margin AUROC by <0.006.
- **The model's own readout only weakly tracks what the probe reads**
  (Spearman 0.30 between probe score and yes/no margin on the same items).

The unresolved question that motivated tonight: **why is correctness readable
but unused?** Two competing explanations were on the table:

- **E1 (routing/verbalization).** The information exists in the residual
  stream but sits outside the pathway that produces output tokens; it is only
  recruited into that pathway under specific conditions (e.g. being asked).
- **E2 (epiphenomenal).** The probe reads a byproduct that the model never
  uses for anything - a correlate of correctness, not a functional signal.

Both predict the steering null. They differ in what should happen when the
model is *asked* the verification question: E1 predicts the signal enters the
output channel at some depth; E2 predicts it never does, and the behavioral
0.669 comes from a separate computation.

---

## 1. Literature reconnaissance (4 parallel agents)

**Why.** Before spending GPU time, find out whether the tools to distinguish
E1/E2 already exist. Specifically: methods that read *what the model is
disposed to say* at intermediate depths.

**What was found (all verified, links in agent reports):**

- **Logit lens** (nostalgebraist 2020): apply the model's own final norm +
  unembedding to intermediate residual states. Reads the output channel at any
  depth. Known caveat: assumes intermediate states share the final layer's
  basis, which fails on some model families ("representation drift", Belrose
  et al. 2023, arXiv 2303.08112). Recommended mitigation: pair with supervised
  probes - which we already have.
- **Tuned lens**: fixes drift with learned per-layer affine maps. **No
  pretrained tuned lens exists for any Qwen model** (verified by enumerating
  the AlignmentResearch registry). Training one would be ~1.7B params of
  translators. Deprioritized.
- **J-lens / J-space** (Anthropic, "Verbalizable Representations Form a Global
  Workspace in Language Models", transformer-circuits.pub/2026/workspace,
  July 2026): per-layer Jacobian of final-layer residual w.r.t. layer-l
  residual, averaged over prompts. Rows of `W_U J_l` give, per vocabulary
  token, the direction that most raises the disposition to eventually emit it.
  J-space = sparse cone spanned by such directions = the model's verbalizable
  workspace. **Critical for us:** the paper reports that when a supervised
  probe direction is decomposed, its J-space component (~10-15% of variance)
  carries most of the causal power to redirect answers. That is a direct,
  falsifiable candidate mechanism for our steering null.
- **R-lens** (Blank, Bhatia, Nanda; LessWrong Aug 5 2026): drop-in J-lens
  replacement using layerwise-relevance-propagation rules in the backward
  pass; fixes early-layer noise that Anthropic's own paper acknowledges.
  Pre-fitted J-lens AND R-lens weights are free on HF (`camilablank/
  workspace-lenses`, `neuronpedia/jacobian-lens`).
- **Our own repo**: Denis designed a J-space experiment (his exp 10) but
  never ran it, on Qwen3-4B, asking a different question (story-abstraction vs
  grammar-emission dissociation). No lens method has ever been run in this lab.
  Shivam's standing guardrail: lens methods only when attached to a concrete
  question. Ours is attached.

**Decision.** Run the cheapest discriminating experiment first (logit-lens
style readout on activations we already have), then the J-space decomposition
on a model with a free pre-fitted lens.

---

## 2. Margin lens, reader mode (Qwen3-32B)

**Question.** During passive reading, does correctness information enter the
model's output channel at *any* depth?

**Hypothesis (E1).** It enters somewhere - probably late, since the probe
signal itself only becomes strong past ~two-thirds depth.

**Method.** Extracted just two tensors from the checkpoint: the final RMSNorm
weight and the 8 `lm_head` rows for yes/no token variants (~100 KB, CPU-only
Modal job). Applied `RMSNorm(h_l) @ (w_yes - w_no)` at every one of the 65
captured layers on the existing 2,000 reader-mode activations, scored AUROC
against the mechanical labels. Cost: pennies, no GPU.
Code: `probe-experiments/analysis/{extract_head_rows,margin_lens}.py`.

**Result.** Flat. last site: 0.503 (L0), 0.520 (L48), 0.518 (L61), 0.509
(L64), max 0.545 @L59. mean site never exceeds 0.53. On the *same*
activations a supervised probe reaches 0.705.

**Verdict.** Hypothesis not supported in reader mode. Correctness is present
(probe 0.70) but never enters the verbal channel at any depth during passive
reading. This is a stronger statement than "the signal is lost late" - it is
never routed at all.

**Caveats recorded.** Raw logit lens can under-read on Qwen-class models due
to basis drift; a null here is weaker evidence than a positive. This is
exactly why the asked-mode contrast below matters: same method, same model,
same texts - if the method were simply blind, asked mode would also be flat.

---

## 3. Margin lens, asked mode (Qwen3-32B) - the recruitment experiment

**Question.** When the verification question is in the prompt (where the model
behaviorally scores 0.669), where along depth does the signal enter the
channel?

**Hypothesis.** If E1 is right, the curve should rise at some depth. The
*shape* is informative: early rise = the question changes processing from the
start; late rise = a routing operation near the output.

**Method.** Added an asked-mode capture path (verification prompt through the
chat template, thinking disabled) and captured all 65 layers. Ran the
identical margin lens.

**Result.** Rises from ~0.50 through the first two-thirds, then climbs:
0.574 (L48), **peak 0.686 (L53)**, 0.666 (L61), **0.659 at the final layer**.

**Verdict.** E1 supported, and it also rules out "the method is blind to
Qwen-class geometry": the same instrument that read flat 0.50 in reader mode
reads 0.686 in asked mode. The question performs a **late-depth routing
operation** - correctness information is moved into the output pathway only in
the last third of the network, only when asked.

**Unexpected observation (the interesting one).** The curve PEAKS at L53 and
then DECLINES by ~0.027 to the output. The model's own readout is better 11
layers before the end than at the end.

---

## 4. Knows-vs-says (same asked-mode activations)

**Question.** Is there more correctness information in the stream than the
model expresses, and where is the gap largest?

**Method.** Per layer, on identical activations: supervised probe (grouped CV
by problem) vs the model's own margin readout. `analysis/knows_vs_says.py`.

**Result.** last site: probe peaks 0.725, model's margin peaks 0.686 and ends
at 0.659; max gap 0.128 at L46. mean site: probe 0.737, margin ends 0.508,
gap up to 0.206.

**Interpretation (moderate confidence).** The model carries substantially more
correctness information than its verbal channel expresses, at every depth,
even when directly asked. Combined with entry #3, the picture is: information
present throughout -> partially routed to the channel when asked -> and even
the routed portion degrades slightly before the output.

**Data-quality flaw found and fixed.** This ran on 300 texts (150 problems)
because asked-mode inherited the behavioral gate's subsample. Too small for
law-disjoint splits and for trusting per-layer differences of ~0.03. Capture
now uses the full 2,000 texts by default (`--gate-sample` reproduces the old
subset). Full-data reruns launched for Qwen3-32B and Qwen3.6-27B; the numbers
above should be treated as provisional until those land.

---

## 5. Roster expansion and the flagship model choice

**Question.** Which model should carry the workspace (J-space) experiment?

**Constraint discovered in the literature pass:** fitting a J-lens for
Qwen3-32B is expensive (63 source layers x 5120^2, ~40 backward passes per
prompt, ~1000 prompts). Free pre-fitted J-lens AND R-lens exist for
Qwen3.6-27B - which has **identical geometry to our Qwen3-32B (64 layers,
hidden 5120)**, so our entire pipeline ports without shape changes.

**Behavioral gates (margin AUROC, 300 balanced texts, threshold-free):**

| Model | Margin AUROC |
|---|---|
| Llama-3.1-8B | 0.522 |
| Qwen2.5-7B | 0.564 |
| Qwen3.5-4B | 0.626 |
| Llama-3.3-70B | 0.629 |
| Qwen3-32B | 0.669 |
| **Qwen3.6-27B** | **0.826** |
| gemma-3-27b | (running) |

**Decision.** Qwen3.6-27B becomes the flagship: it is the most capable
verifier we have found, it has free lenses, and its geometry matches our
existing captures. Fitting our own 32B lens is deferred - it would cost real
GPU hours to answer the same question on a *weaker* model. This is a
resource-efficiency call, not a scientific compromise: if the flagship result
is strong, confirming it on Qwen3-32B afterwards is a well-motivated
follow-up rather than a prerequisite.

**Secondary value:** the gate ladder above is itself a result - capability at
this task varies hugely across families and generations (a 4B model from a
newer generation beats a 70B from an older one), which is the x-axis for the
capability-vs-representation co-emergence analysis.

---

---

## 6. A real bug, caught by validation (IMPORTANT)

**Why this entry exists.** Before building on the margin-lens results I ran the
obvious correctness check a reviewer would demand: applied to the FINAL layer
state, the lens must reproduce the model's true logits.
`analysis/validate_lens.py`.

**Result: FAIL.** Max absolute logit difference 28.2, argmax mismatch.

**Diagnosis (tested, not assumed).** I hypothesized that HuggingFace appends
the last hidden state AFTER applying the final norm, meaning
`hidden_states[-1]` is already normalized and our lens normalized it twice. I
tested this by computing logits three ways on the same states and comparing to
the model's true logits:

| computation | max abs diff vs true logits |
|---|---|
| our lens (norm applied) | 28.215 |
| no norm applied | **0.0622** (= bf16 rounding) |

Diagnosis confirmed. Rows 0..n-1 of our captures are raw residuals (single
norm correct); the LAST row is pre-normed (must be used as-is).

**Impact and correction.** Only the final capture row was affected. Fixed in
`margin_lens.py` and `knows_vs_says.py`; all analyses recomputed. Corrected
numbers (Qwen3-32B):

- reader mode: still flat - max 0.545 @L59, final 0.499. Finding unchanged.
- asked mode: peak 0.6855 @L53, **final layer 0.6701** (was mis-computed as
  0.6588).
- knows-vs-says: probe 0.7247 vs model's expressed 0.6701 -> gap 0.055 at the
  output, max gap 0.128 @L46.

**A validation that came free with the fix.** The corrected final-layer lens
value is **0.6701**, and the independently measured behavioral margin gate on
the same 300 texts is **0.6694**. Two completely separate measurement paths
(saved activations + reconstructed readout vs live generation logits) agree to
0.0007. This is strong evidence the method is now correct.

**Claim that got weaker and must be reported honestly.** I previously wrote
that the model's readout "peaks at L53 then DECLINES by 0.027 to the output."
With the bug fixed the decline is 0.686 -> 0.670 = **0.015**, on 300 texts.
That is within plausible noise for this sample size and is NOT currently a
defensible claim. It is demoted to an observation pending the full 2,000-text
rerun and a significance test.

**Lesson recorded.** The reader-mode null was never at risk (all its layers
are raw residuals), but had I skipped this check, a headline sentence in the
write-up would have been wrong. Every lens-style readout in this project now
carries a reproduce-the-true-logits gate.

---

## 7. Model-roster reconnaissance (agent) and a plan change

Verified inventory of free pre-fitted lenses (agent enumerated the HF APIs):

- **Qwen3-32B - our own primary model - HAS a free J-lens** at
  `neuronpedia/jacobian-lens` (n=1000, wikitext, Anthropic recipe). This was
  the decisive discovery: the flagship workspace experiment can run on the
  exact model where every probe and steering number already lives, with no
  cross-model caveat and no expensive lens fitting.
- `camilablank/workspace-lenses` has matched J+R pairs for qwen3.5-4b/9b/27b,
  qwen3.6-27b, gemma-3-27b-it, plus MoE models.
- **Architecture warning that changes model choice:** Qwen3.6-27B and
  Qwen3.5-4B are NOT plain dense transformers - 3 of every 4 blocks are gated
  linear attention (DeltaNet/SSM family) carrying recurrent state across
  positions. Residual capture and probing remain valid, but **steering is not
  position-local** in these models: an intervention at token t propagates
  through recurrent state. For a causal steering claim this is a serious
  confound.

**Decision (plan change).** Qwen3-32B - dense, standard residual path, free
lens, and the model all our results are on - becomes the flagship for the
J-space experiment. Qwen3.6-27B remains valuable as the strongest verifier
(margin 0.826) and for probe/co-emergence work, but any steering result there
would need the recurrence caveat stated. Fitting our own 32B lens (previously
planned, tens of GPU-hours) is now unnecessary.

## Running hypotheses entering the next block

- **H-route (favored).** Correctness lives outside the verbalization pathway
  and is recruited into it late, only under task framing. Predicts: on
  Qwen3.6-27B (a better verifier) the asked-mode rise should be earlier and/or
  larger; and our probe direction should have a small J-space component.
- **H-workspace-steer.** Steering failed because we pushed a direction that is
  mostly workspace-orthogonal. Predicts: the J-lens yes/no direction
  (`J_l^T (w_yes - w_no)`) should steer where our probe direction did not, and
  the two directions should have low cosine similarity.
- **H-capability.** The steering null is partly a capability artifact (the 32B
  is a weak verifier at 0.669). Predicts: steering on Qwen3.6-27B (0.826)
  produces a measurable effect where the 32B did not. This is a confound we
  can now test directly, and it must be tested before attributing the null to
  workspace geometry.

These three make different predictions and can be separated by the
experiments queued next.

---

## 8. FLAGSHIP: is our correctness direction inside the verbalization pathway?

**Question.** Our probe reads correctness (law-disjoint 0.599) but steering
along that direction moved nothing. The workspace paper offers a candidate
mechanism: a probe direction can be almost entirely outside the subspace that
actually drives what the model says. Is ours?

**Hypothesis (H-workspace).** cosine(probe direction, J-lens yes/no direction)
is near zero. If so, steering along the probe direction cannot move the yes/no
readout, and the "causally inert" result is explained by geometry rather than
by the information being epiphenomenal.

**Method.** Qwen3-32B has a free pre-fitted J-lens (Neuronpedia, n=1000).
Note: the published file is a raw `fit()` checkpoint, not a saved lens, so the
lens was reconstructed exactly as the running mean `jacobian_sum / n_done`.
The J-lens yes/no direction at block L is
`v_L = (mean W_U[yes_ids] - mean W_U[no_ids]) @ J_L`. Index mapping made
explicit: capture row R = output of decoder block R-1, so our layer-61 probe
direction corresponds to J-lens block 60.
Code: `steering/extract_lens_directions.py`, `analysis/jspace_alignment.py`.

**Result.**

| measurement | value |
|---|---|
| cosine(probe dir, J-lens yes/no dir) at the matched layer | **0.0110** |
| random unit-direction \|cosine\| baseline, p95 (5120 dims, n=2000) | 0.0270 |
| J-lens direction used AS a classifier, reader mode (best layer) | 0.5566 |
| J-lens direction used AS a classifier, asked mode (best layer) | 0.5615 |

**Verdict: H-workspace supported.** Our correctness direction is
statistically indistinguishable from orthogonal to the model's yes/no
verbalization direction - the observed cosine is *below* the 95th percentile
of random directions in the same space. Independently, the verbalization
direction itself is nearly uninformative about correctness (0.556-0.562),
i.e. these are two genuinely different directions carrying different
information.

**Why this matters.** It converts our steering null from "we pushed and
nothing happened" into a mechanical statement: we pushed along a direction
with essentially no component on the axis that controls the model's answer, so
no effect was possible. The information is real, readable, and sits outside
the verbalization pathway.

**The control this demands (running).** If the geometry explanation is right,
steering along the J-lens direction *should* move the margin. That is also the
positive control our original steering experiment lacked: without it, a null
cannot distinguish "this direction is inert" from "our steering harness does
nothing". Launched: `modal_steer.py --direction-src jlens`.

---

## 9. Adversarial review (independent agent) - what survived

I commissioned an agent to falsify the recruitment claim. It reproduced every
number from the artifacts and found real problems. Recording its verdict in
full because several of my claims changed.

**SAFE (survives all objections):**
> The model's own yes/no readout is uninformative about correctness at every
> depth when reading story+RG as bare text (0.499-0.545, n=2000), while a
> supervised probe on the *same activations at the same position* reaches
> 0.62-0.64. The information is in the residual stream and is not on the
> yes/no unembedding axis.

**CONFOUNDED (withdrawn pending controls):** the "task framing performs
late-depth routing" interpretation, for two reasons I accept:

- **C1 position.** Reader-mode `last` = final token of the RG mid-document;
  asked-mode `last` = the generation position inside the chat template. These
  are different sites, so part of the "late rise" is just attention hauling
  content to a fresh position. Control: capture the asked-mode prompt at the
  *end of the RG* (mid-prompt) - the `ansend` site. **Now running.**
- **C2 licensing.** In reader mode "yes"/"no" are not licensed continuations
  at all, so AUROC = 0.5 there is close to guaranteed a priori rather than
  measured. Control: an identical chat prompt at the identical position asking
  a question unrelated to correctness ("Is the story written in English?"). If
  the margin still tracks correctness, the effect is format/position, not
  verification framing. **Now running** (`behavior/placebo_prompt.md`).

**DEAD:** "peaks at L53 then declines to the output." The endpoint was the
double-norm artifact; corrected Delta = 0.0156 with a bootstrap 95% CI of
[-0.004, +0.036] (agent's computation, 400 resamples over problems). The
corrected curve is a step at ~block 49 followed by a flat plateau, and L53 is
a post-hoc max over 15 near-identical values. Clause deleted.

**C5, and it changed a headline.** `knows_vs_says` compared a probe under
problem-grouped CV (which leaks: the same law can appear on both sides) to a
fit-free lens. Re-run with law-disjoint grouping:

| site | probe (problem CV) | probe (law-disjoint) | model readout | honest gap |
|---|---|---|---|---|
| last | 0.7247 | **0.6708** | 0.6701 | **0.034 max, ~0.001 at output** |
| mean | 0.7369 | 0.6866 | 0.5480 | 0.153 (but `mean` site is confounded, C4) |

**So the "model knows much more than it says" claim is withdrawn at the
answer position.** Under a proper split the probe reads 0.6708 and the model
expresses 0.6701 - essentially identical. The honest and more interesting
statement is the reverse: *when asked, Qwen3-32B's verbal readout captures
essentially everything a linear probe can extract at that position.* Hidden
knowledge appears only in reader mode, where the channel is not licensed.

**Also flagged and accepted:** C4 (the asked-mode `mean` site uses a character
midpoint heuristic, averaging ~211 tokens vs 34 in reader mode - cross-mode
mean comparisons are invalid and are dropped), C6 (index 64 is post-final-norm,
not a residual state; block 63's raw residual is never exposed by HF - so
curves should be described in block terms and index 64 plotted separately),
C7 (asked capture was n=300 while reader was n=2000; full 2000 runs launched),
C8 (asked-mode `meta.json` inherited the reader-mode `text_template` string -
fixed; transformers version differed 5.14.1 vs 5.15.0 between captures -
recorded, not yet controlled).

**Ruled out by the agent, so we stop worrying about them:** label/surface
leakage (RG-statistics-only classifier 0.5111), yes/no token id correctness
(margin sign agrees with the model's actually generated word on 297/300),
bf16->fp16 overflow (0 NaN/Inf), BOS handling, right-padding index.

---

## 10. What does the correctness direction SAY? (vocabulary readout)

**Question.** Orthogonal to the yes/no axis is not the same as outside the
verbalizable workspace. Is our direction verbalizable at all?

**Method.** The J-lens readout of a residual direction v at block L is
`(W_U J_L) v`: one score per vocabulary token, how much v raises the model's
disposition to eventually emit that token. Applied to our probe direction,
its negation, a random direction, and - critically - two POSITIVE CONTROLS
(the J-lens yes/no direction and the raw unembedding yes/no contrast), which
are verbalizable by construction and must read out as yes-tokens if the code
is correct. `analysis/direction_vocabulary.py`.

**Result.**

| direction | max score | sd | top tokens |
|---|---|---|---|
| J-lens yes/no (POSITIVE CONTROL) | **0.9503** | 0.0889 | `yes`, ` yes`, `_yes`, `_YES`, ` Yes`, `Yes`, ... |
| raw unembedding yes/no | 1.0397 | 0.0230 | `yes`, ` Yes`, ` YES`, ` oui`, ` yeah` ... |
| **our probe direction** | **0.1143** | 0.0239 | `水稻`, `$LANG`, `不来`, `.initializeApp` ... |
| probe direction, negated | 0.1087 | 0.0239 | unrelated vocabulary |
| random direction (NULL) | 0.1121 | 0.0234 | unrelated vocabulary |

**Verdict.** The readout method demonstrably works: a verbalizable direction
reads out as its own tokens at 8x the noise floor. Our correctness direction
reads out **at the random-direction level** (0.1143 vs 0.1121). It is not
merely off the yes/no axis - it has no verbal expression at all.

**Statement of the finding.** Qwen3-32B linearly encodes whether a candidate
formalization is correct in a direction that, by the model's own causal
readout, corresponds to saying *nothing in particular*. The knowledge is
real and decodable but verbally silent.

---

## 11. STEERING POSITIVE CONTROL - the missing piece of the original null

**Question.** Our steering experiment moved nothing. Two explanations: (a) the
probe direction is causally inert, (b) our steering harness does not work. The
original experiment could not distinguish these because it had no positive
control - no direction that SHOULD move the readout.

**Hypothesis.** The J-lens yes/no direction is a causal direction for saying
yes by construction. If the harness works, injecting it must move the margin;
if the geometry explanation is right, our probe direction still will not.

**Method.** Identical harness, identical 300 texts, identical alphas and
controls; only the injected direction changes.
`steering/modal_steer.py --direction-src jlens`.

**Result.**

| condition (alpha=+0.5) | mean margin | delta vs baseline | yes-rate | AUROC |
|---|---|---|---|---|
| baseline (alpha 0) | -2.618 | - | 0.06 | 0.6694 |
| **J-lens direction** | **+21.540** | **+24.16** | **1.00** | 0.6468 |
| our probe direction | -2.685 | -0.07 | 0.067 | 0.6723 |
| random, norm-matched | -3.399 | -0.78 | 0.037 | 0.6706 |

Dose-response for the J-lens direction is monotone and enormous:
alpha = -1.0 -> -45.4, -0.5 -> -33.7, 0 -> -2.6, +0.25 -> -0.2, +0.5 -> +21.5,
+1.0 -> +22.4; yes-rate goes 0.00, 0.00, 0.06, 0.35, 1.00, 1.00.

**Verdict.** The harness works. A 24-logit swing and a 6%->100% flip in the
model's answer are available from this injection site with this code. Our
correctness direction produces a change (-0.07) **smaller than a random
direction of the same norm** (-0.78). Explanation (b) is eliminated; the
original null is real and is explained by geometry.

**Nuance worth reporting honestly.** Steering the J direction shifts the
model's *bias*, not its *evidence*: at alpha=+0.5 it answers yes to
everything, yet AUROC only falls 0.669 -> 0.647, i.e. it still ranks correct
above wrong. So the verbalization axis controls what the model *says*, not
what it *knows*. That is a nice independent illustration of the same
knows/says separation.

---

## 12. Are there TWO correctness representations?

**Question.** If the reader-mode direction is verbally silent, how does the
model answer at 0.67 when asked? Either it recruits that same direction, or it
uses a different one.

**Method.** Fit a direction on each activation set at its own best layer,
compare. `analysis/two_directions.py`.

**Result.** cosine(reader direction, asked direction) = **0.0533** (random
|cos| p95 = 0.0265). Cross-application: the asked-mode direction reads
correctness in reader-mode activations at **0.6004**, while the reader-mode
direction reads asked-mode activations at only **0.5367**.

**Interpretation (moderate confidence).** The two are largely different
directions - roughly twice the random baseline, so not literally orthogonal,
but sharing very little. The asymmetry is suggestive: what the model uses to
answer is partially present during passive reading, but what passive reading
encodes is largely NOT what answering uses.

**Caveat.** The asked set here is n=300 vs reader n=2000, and both directions
are fit in-sample (the 1.0 self-AUROCs are that, not a result). Re-run when
the full 2000-text asked capture lands before treating the asymmetry as solid.

---

## 13. Co-emergence across models, and the control that makes it meaningful

**Question.** Does a law-general correctness representation appear only in
models that can actually do the task, or is it there regardless?

**Method.** Same frozen dataset, same law-disjoint grouping, reader-mode
activations, four models spanning the capability range measured by the
threshold-free behavioral margin gate.

**Result.**

| Model | capability (margin AUROC) | representation (law-disjoint probe) |
|---|---|---|
| Llama-3.1-8B | 0.522 | 0.527 |
| Qwen3.5-4B | 0.626 | 0.549 |
| Qwen3-32B | 0.669 | 0.599 |
| Qwen3.6-27B | **0.826** | **0.8146** |

Pearson 0.948, Spearman 1.0 (n=4). Shuffled-label controls clean everywhere
(0.48-0.52).

**The obvious objection, and the control that kills it.** Maybe bigger/newer
models just represent everything better, and any probe would scale. Tested by
decoding a task-IRRELEVANT property - the story's theme - from the same
activations, same splits, at fixed fractional depths (which also removes the
"deeper models get more chances at a maximum" concern):

| Model | correctness (law-disjoint) | theme accuracy (chance 0.25) |
|---|---|---|
| Llama-3.1-8B | 0.533 | 1.000 |
| Qwen3.5-4B | 0.533 | 1.000 |
| Qwen3-32B | 0.599 | 1.000 |
| Qwen3.6-27B | 0.811 | 1.000 |

Every model decodes the irrelevant property perfectly, including the weakest.
Only correctness scales. **The generic-representational-quality explanation is
eliminated**, and the scaling is specific to the property we care about.

**Remaining caveat.** n=4 means Spearman 1.0 is a 1-in-24 coincidence under
the null, and the models differ in architecture (Qwen3.5/3.6 are hybrid
linear-attention). The correlation is descriptive; the specificity control is
the load-bearing part.

---

## 14. Replication of the orthogonality result (second model, two lenses)

**Why.** A geometric null on one model with one lens is thin evidence. The
strongest available replication: a different model, lenses fitted by a
different group with a different recipe, and two methodologically distinct
lens types (J-lens and its LRP-based successor R-lens).

**Result** (`runs/lens-v1/replicate_alignment_qwen3.6-27b.json`). Qwen3.6-27B,
whose correctness representation is far STRONGER (law-disjoint 0.821 vs the
32B's 0.599), so "the direction is weak or noisy" cannot explain a null:

| lens | cosine at matched layer | all-block mean abs cosine | blocks exceeding random p95 |
|---|---|---|---|
| J-lens (n=25 pile prompts) | +0.0127 | 0.0083 | **0 / 63** |
| R-lens (independent method) | +0.0151 | 0.0085 | **0 / 63** |

For comparison, Qwen3-32B with the Neuronpedia n=1000 wikitext lens: mean
0.0065, max 0.0189, 0/63.

**Verdict.** The orthogonality of the correctness direction to the
verbalization axis holds across models, across lens-fitting recipes, across
lens methods, and across every layer. It is a property of these
representations, not of one measurement.

---

## 15. THE PLACEBO CONTROL FALSIFIED OUR OWN INTERPRETATION

**Question.** Does the yes/no channel carry correctness because the model was
asked about correctness, or merely because it is in an answer position in a
chat format?

**Hypothesis under test (ours).** Task framing recruits correctness into the
verbal channel. Prediction: a yes/no question about something ELSE should
leave the channel uninformative (~0.50).

**Method.** Identical prompt structure, identical chat template, identical
capture position; only the final question changed to "Is the story written in
English?" - a property that is (a) unrelated to correctness and (b) identical
for both texts of a problem, since they share a story. n=2000.
`behavior/placebo_prompt.md`, `analysis/site_margin.py`.

**Result.**

| condition, generation position | lens AUROC | n |
|---|---|---|
| real verification question | 0.6701 | 300 |
| **placebo question (unrelated)** | **0.6366** | 2000 |
| bare text, mid-document (reader) | 0.4990 | 2000 |

**Verdict: our interpretation was WRONG.** An unrelated yes/no question
recovers most of the effect. The channel does not open because the model was
asked about correctness; it opens because the model is in an answer position
in a chat format. The claim "task framing performs a late-depth routing of
correctness" is **withdrawn**.

**What survives, stated precisely:**

> Reading story+RG as bare text, the yes/no axis carries no correctness
> information at any of 65 layers (max 0.545, final 0.499). Place the same
> content in a chat prompt and read where the model is about to answer *any*
> yes/no question, and that same axis carries correctness at 0.64-0.67 -
> whether or not the question concerns correctness.

That is still a genuine dissociation - format and position gate access to the
verbal channel - but it is a claim about *answer-position states*, not about
verification framing.

**Why this matters more than the loss.** It also strengthens the central
result. The reader-mode correctness direction is orthogonal to the
verbalization axis and verbally silent; the answer-position signal lives ON
that axis. Combined with the near-orthogonality of the two fitted directions
(cosine 0.053, entry 12), the coherent picture is:

> The model maintains a silent correctness representation while reading, and
> separately constructs a verbally-accessible one when placed in an answering
> context. The two are nearly orthogonal. Neither is the other.

**Process note.** This is the second time tonight an adversarial control
overturned one of my own interpretations (the first was the leaky-CV "knows
more than it says" claim). Both were predicted in advance by the skeptic
agent. The lesson I am recording: for any claim of the form "X causes the
model to do Y", run the version of the experiment where X is replaced by a
placebo BEFORE writing the interpretation down.

---

## 16. I OVER-RETRACTED. The layer-resolved placebo comparison restores a
sharper version of the claim.

**What happened.** In entry 15 I compared the placebo and real questions at the
FINAL layer only (0.6366 vs 0.6701), concluded the effect was format/position
rather than question content, and withdrew the routing claim. Then the full
placebo curve finished and I compared them layer by layer. The single-number
comparison was hiding the structure.

| layer | real question (n=300) | placebo (n=2000) | difference |
|---|---|---|---|
| 40 | 0.5080 | 0.5042 | +0.004 |
| 45 | 0.5112 | 0.5028 | +0.008 |
| 48 | 0.5744 | 0.5120 | **+0.062** |
| 50 | 0.6705 | 0.5287 | **+0.142** |
| **53** | **0.6855** | **0.4831** | **+0.202** |
| 56 | 0.6768 | 0.4929 | **+0.184** |
| 58 | 0.6680 | 0.6121 | +0.056 |
| 61 | 0.6658 | 0.5850 | +0.081 |
| 64 (final) | 0.6701 | 0.6366 | +0.034 |

First layer above 0.55: **real L48, placebo L58** - a ten-layer gap.
Peak: real 0.6855 @L53; placebo 0.6366 @L64 (its final layer).

**Two separable mechanisms, distinguished by depth.**

1. **Question-specific routing is REAL, and it happens at mid-late layers.**
   Between blocks ~47 and ~57 the real verification question puts correctness
   on the yes/no axis at 0.67-0.69 while the placebo sits at chance
   (0.48-0.53). The gap of +0.20 at L53 is roughly 5.6 sigma given the pooled
   standard error (SE_real ~0.033 at n=300, SE_placebo ~0.013 at n=2000).
2. **A format/position effect exists but only at the very end.** By the final
   layer the placebo also reaches 0.6366, and the remaining gap (+0.034) is
   under one sigma. Being in an answer position surfaces *some* correctness
   signal regardless of what was asked - but only in the last few blocks.

**Revised claim (replaces both the original and the retraction):**

> Asking the model about correctness routes that information onto the yes/no
> axis from about two-thirds depth onward (blocks ~48-57), where an unrelated
> yes/no question leaves the axis at chance. A weaker, question-independent
> effect appears only in the final blocks, where merely being in an answer
> position surfaces correctness whatever was asked.

**Process note, and the reason this entry exists.** I retracted on one number
and un-retracted on the curve. The lesson is not "trust the first
interpretation" - it is that a single-layer comparison was the wrong
instrument for a claim about depth, and I should have run the layer-resolved
comparison before either writing the claim or withdrawing it.

**Caveat, stated plainly.** The real-question curve here is n=300 and the
placebo is n=2000. The full 2000-text real-question capture exists on the
volume and is being pulled; the mid-late gap is far too large to be an n
artifact (5+ sigma), but the exact values will be refreshed when it lands.

---

## 17. Matched-n confirmation of the depth-separated placebo result

The n=300 caveat from entry 16 is resolved. Re-ran the real-question curve on
the full 2000-text capture; both conditions now n=2000, identical items.

| layer | real (n=2000) | placebo (n=2000) | difference |
|---|---|---|---|
| 45 | 0.5037 | 0.5028 | +0.001 |
| 48 | 0.5709 | 0.5120 | +0.059 |
| 50 | 0.6685 | 0.5287 | +0.140 |
| 53 | 0.6889 | 0.4831 | +0.206 |
| 56 | 0.6777 | 0.4929 | +0.185 |
| 58 | 0.6643 | 0.6121 | +0.052 |
| 64 | 0.6677 | 0.6366 | +0.031 |

Peak real 0.6889 @L53; peak placebo 0.6366 @L64 (its final layer). Maximum gap
**+0.2115 at block 54**. With SE ~0.013 per curve at n=2000, that is roughly
11 sigma - and the n=300 estimates were within 0.004 of these, so sample size
was never load-bearing.

**Final form of this result.** Two mechanisms, cleanly separated by depth:

1. **Question-specific routing**, blocks ~48-57: asking about correctness puts
   it on the yes/no axis at 0.67-0.69 while an unrelated yes/no question leaves
   the axis at chance.
2. **Question-independent answer-position effect**, final blocks only: the
   placebo reaches 0.6366 by the last layer; the remaining gap (+0.031) is
   ~1.7 sigma.

Both are real; neither alone is the story. This is the version to put in the
paper, and it exists only because the placebo control was run AND the
comparison was made layer-resolved rather than at a single layer.

---

## 18. POSITION CONTROL (the 2x2) - the confound is refuted

**Question.** The reader-vs-asked contrast differed in two ways at once:
whether a question was present, and WHERE the activation was read (reader at
the end of the RG mid-document; asked at the chat generation position). Which
one does the work?

**Method.** Capture the asked prompt at the *answer-end* position - the last
token of the RG inside the chat-formatted prompt - so the comparison against
reader mode is position-matched. n=2000. `analysis/site_margin.py` on the
`ansend` site.

**Result.**

| | answer-end position | generation position |
|---|---|---|
| **no question** (bare text) | 0.499 final / 0.545 max | n/a |
| **question present** | **0.6294** (best @L62) | 0.6889 (best @L53) |

**Verdict: the positional confound is refuted.** Holding position fixed at the
end of the RG, adding the verification question to the context moves the yes/no
channel from chance (0.499-0.545) to **0.629**. Position contributes an
additional increment (0.629 -> 0.689 when read at the generation position), but
it explains none of the base effect.

**Combining all three controls, the final picture:**

1. No question, any position -> the yes/no axis carries nothing (0.50).
2. Question in context, read mid-prompt -> 0.629. **The question does the
   work, not the position.**
3. Question in context, read at the answer position -> 0.689. Position adds
   ~0.06.
4. UNRELATED question at the answer position -> 0.637 overall, but **at chance
   through blocks 48-57** where the real question reads 0.67-0.69. So the
   question's *content* determines the mid-late-depth signal, while a generic
   answer-position effect appears only in the final blocks.

**What we can now say, fully controlled:**

> Correctness information sits outside Qwen3-32B's verbal channel while it
> reads. Posing a verification question routes it onto the yes/no axis from
> about two-thirds depth - an effect that is not explained by readout position
> (position-matched control: chance -> 0.629) and not explained by generic
> answer-formatting (placebo control: at chance through the same blocks).

That is the claim that survives every control we ran, and it is stronger than
the version I first wrote AND stronger than the version I retracted.
