# Experiments

## **1 Contrastive steering vector: story → abstracted NL**

**What it is.** Take matched pairs of the same underlying problem — the story version and the abstracted literal-NL version (which we generate deterministically, so pairs are exact). Run both through an open-weights model and collect residual-stream activations at one or more layers. The mean difference between the two sets of activations gives a steering vector that points from "free-form story representation" toward "structured mathematical-problem representation."

**The experiment.** Add this vector to the model's activations during no-think inference on the story → rigid-grammar task and measure whether task accuracy (scored by our existing checkform syntactic checker) rises toward the thinking-mode ceiling.

**Why it matters.** The two-stage pipeline result suggests the model can translate well once abstraction is done; if injecting this direction alone boosts accuracy, we have *causal* evidence that we can control a computation correlated with faithful formalization, not just observe it. Shivam called demonstrating this degree of control a good "midpoint result." A later second half could be fine-tuning with these features activated to push baseline accuracy to ceiling permanently.

**Question.** Can we causally close the no-think accuracy gap by injecting a single direction in activation space, i.e. is "represent this as a math problem, not a story" a controllable computation inside the model?

## **2 Probing and steering with contrastive datasets (the general recipe)**

**What it is.** The generalization of 1: any concept we want to monitor or control is defined by a contrastive set of exemplars. From that contrast you can train a linear probe (monitoring) or derive a steering direction (control). The story-vs-abstracted contrast is just the first instance; other contrasts (e.g. faithful vs unfaithful translations, thinking vs non-thinking traces) follow the same recipe.

**Key caveat from Shivam.** The choice of contrast *is* the experiment: it determines exactly which feature or concept you end up manipulating, so be deliberate about what the two sides of the pair differ in and nothing else. Exemplar generation can be sped up with an LLM, but the examples should be curated carefully (and, per our ground-truth discipline, anything used as labelled data must be re-labelled mechanically, not judged by the generating model).

**His process advice.** Don't write a one-pager first — probing, steering, and visualization can be run largely autonomously, so spend 30 minutes specifying the features to differentiate and the required data, then run and look at results. He offered to review a short doc of proposed contrasts if we want convergence.

**Question.** For any concept we can define by contrast (faithfulness, ambiguity, abstraction), can we read it off the model's internals (probe) and move the model along it (steer)?

## **3 PCA / activation-structure visualization**

**What it is.** Collect activations (token-level, sentence-level, or full-paragraph representations) across a well-partitioned dataset, project to 2D/3D with PCA, and color points by known tags: representation type (story / literal NL / rigid grammar), complexity level (number of operations, nesting depth), or underlying law identity. Then look for structure — clusters, orderings, trends.

**Why it matters.** If activations of many differently-themed stories cluster by the implication statement they encode, that's early evidence for a representation-invariant "semantic core," which feeds directly into our manifold-of-semantic-content direction. This also subsumes Luiza's clustering idea: rather than a clustering algorithm, first just look at whether the structure is visible.

**Caveats.** PCA is very cheap, so definitely run it — but it's exploratory and a null result (an undifferentiated point cloud) is fine and not a big deal. Crucially, the input data must contain genuine order or structure (his example: days of the week lie on a circle because there is an order to find); a poorly curated dataset will show nothing regardless of what the model represents.

**Question.** Does the model's activation space contain visible structure that tracks what we know about the data — does the same law cluster together across different surface forms, and do complexity or representation type show up as ordered geometry?

## **4 Attention-pattern analysis: story vs abstracted version**

**What it is.** For matched story/abstracted pairs, visualize attention patterns across all layers (a simple tool can render this) and compare how entities are linked.

**Hypothesis.** In the abstracted literal-NL version, attention should show cleaner, more direct connections between the entities that need to be linked — variables and operations — whereas in the story version those links are diluted or misdirected by narrative content. If confirmed, this gives a concrete mechanistic explanation for why the two-stage pipeline outperforms direct formalization: the abstracted input activates "mathematical problem" pathways and lets the model wire up the right dependencies.

**Role.** This is the exploratory/explanatory complement to the steering experiment (1): steering shows we can *control* the relevant computation; attention analysis helps *explain* why the gap exists in the first place.

**Question.** *Why* does the two-stage pipeline help: does the abstracted version let the model form cleaner attention links between variables and operations that the story's narrative content disrupts?

## **5 SAE feature identification and boosting**

**What it is.** Instead of a raw mean-difference vector, use sparse autoencoders to identify individual interpretable features that fire in formalization- or mathematical-structure contexts, then boost those features during no-think inference and measure the effect on task accuracy.

**Why it matters.** Same causal-control goal as 1, but with more interpretable handles: if a nameable feature (rather than an opaque direction) raises accuracy when boosted, that's a stronger and more communicable insight into *why* non-thinking models underperform. Practically this requires an open model with available pretrained SAEs, which constrains model choice.

**Question.** Same question as 1 but finer-grained: is there a nameable, interpretable feature (rather than an opaque direction) whose activation is what separates good formalizers from bad ones?

## **6 Quick ambiguity-grading pipeline (his self-described "dumb experiment")**

**What it is.** A fast probe of the ambiguity axis. Write a script that gives a frontier model a few examples and asks it to rewrite each one at a specified ambiguity level from 0 to 10, ideally few-shot.

**Two steps.** Step one is purely behavioral: check whether there is any perceivable, consistent difference between what the model produces at low vs high ambiguity — i.e., can the model even conceptualize graded ambiguity? Only if that works, step two is to collect activations for the graded versions and run PCA (per 3) to see whether ambiguity level shows up as structure in representation space.

**Caveats.** It's "dumb" by design: it assumes the model shares our notion of an ambiguity scale, so treat it as a cheap feasibility test, not a labelled dataset. Any rungs we later want to use as data must be mechanically re-labelled, and this connects to our open task of pinning down definitions of ambiguity vs noise vs uncertainty.

**Question.** Does the model even have a workable internal notion of graded ambiguity — can it produce consistently different outputs at "ambiguity level 2" vs "level 8", and if so, does that gradient show up as structure in its activations?

## **Shared guardrails he attached**

Lens-style methods (J-lens / logit lens / NLAs) were explicitly *not* on this list unless attached to a concrete question — exploratory lens output tends to yield plausible-looking text that's hard to convert into insight. Everything above should be timeboxed to a couple of hours of experiments plus brief interpretation; if it takes meaningfully longer, park it and circle back after the red-teaming and Saturday presentation. The adversarial/sandbagging direction (passphrase-triggered underperformance detected via mech interp) was acknowledged as an interesting safety framing but deferred, since it needs a multi-step fine-tuning setup first.

## **7 Steering the Formality Ladder: A Manifold Account of Pipeline Decomposition**

**One-page proposal — MARS V (Scalable Oversight via Formal Verification & Semantic Faithfulness)**

## **Motivation**

Our pipeline-decomposition experiment (Oren) asks whether the explicit two-hop route Story → Literal NL → Rigid Grammar (RG) is more faithful than the direct Story → RG route. If it is, *why*? We propose a mechanistic explanation, adapted from Goodfire's manifold-steering methodology (Wurgaft et al., 2025): the formality rungs of a single law occupy an ordered, low-dimensional path in activation space — an *abstraction manifold* — and the benefit of the explicit intermediate hop is that generating literal-NL tokens forces the model's representation to visit the middle of this path. If that account is right, we should be able to reproduce part of the two-hop gain *without generating intermediate tokens*, by steering the story's representation along the manifold toward the literal-NL region. This would (a) explain the pipeline result mechanistically and (b) constitute a faithfulness-increasing control intervention — directly serving the project's monitor-and-control goal.

Unlike Goodfire's weekday circle, our structure is not imprinted by pretraining co-occurrence; whether it exists at all is part of the question. Unlike our dialect axis, however, the formality ladder is *naturally ordered*, making it the correct object to fit a path to.

## **Hypotheses**

**H1 (geometry).** For a fixed ETP law, representations of its formality rungs (story → fuzzified variants → literal NL → semi-marked NL → RG) lie along a low-dimensional path; across laws these paths are approximately parallel, i.e. activation space factors into a law-identity component (invariant along the ladder) and a shared formality component.

**H2 (causality).** Steering a story's representation along the fitted manifold toward the literal-NL region improves Story → RG faithfulness; manifold steering outperforms linear steering (mean difference vector), and linear steering is more prone to breaking validity (off-manifold outputs).

## **Methodology**



### **Step 0 — Dataset construction (descent pipeline)**

Sample N ≈ 40–80 ETP laws, stratified by complexity (operations, nesting depth). For each law ℓ, generate a formality ladder of R ≈ 5–8 rungs, indexed r  0 (most narrative) to r  R−1 (RG):

- **r  R−1, RG:** produced deterministically from the law's term (`ASSUME: op(x,y) = op(op(y,y),x) / ASK: ...`). No generation step.  
- **r  R−2, semi-marked NL:** templated rendering with explicit quantifiers ("For every choice of objects x and y, …"). Deterministic template.  
- **r  mid, Literal NL:** templated abstract restatement (variables, no narrative).  
- **r  1 … low, fuzzified stories:** LLM-generated narrative wrappings at graded fuzziness (adverb count, concreteness of objects, distractor detail), several themes per rung. **Every LLM-generated rung is re-labelled mechanically after generation** (parse the rung back with our deterministic tooling / verify it was generated by filling a validated template slot-wise); any rung whose recovered law ≠ ℓ is discarded. No LLM judge in the labelling path.

Each ladder item is a *problem instance*: an implication question (ASSUME law ℓ, ASK law ℓ′, truth known from the ETP graph) or a pure translation instance, always tagged with task type. For this experiment we use the **translation** task as primary.

### **Step 1 — Behavioral tasks (what we prompt the model on)**

Model: open-weights, no-think (e.g. Llama-3.1-8B-Instruct class), greedy or low-temperature decoding, fixed few-shot examples held constant across conditions.

- **Condition A, one-hop:** prompt  few-shot examples of (story → RG)  target story; instruction: "Translate the problem into Rigid Grammar. Output only lines of the form `ASSUME: …` and `ASK: …`."  
- **Condition B, explicit two-hop:** call 1: few-shot (story → literal NL)  target story → model emits literal NL. Call 2 (fresh context): few-shot (literal NL → RG)  the emitted literal NL → model emits RG. Only the final RG is scored; the intermediate hop is also logged and mechanically re-labelled so we can score per-hop, not endpoint-only.  
- **Condition C, latent two-hop (Phase 2):** identical prompt to Condition A, plus the activation intervention of Step 4 No intermediate tokens.

**Scoring (all conditions):** deterministic RG parser → **valid** (parses, no free-variable leaks, operator/variable counts sane — the cheap exact sieve runs first); then normalize the emitted term and compare against the intended law via the ETP machinery → **faithful**. Report the full valid × faithful table; headline metric is faithfulness *conditional on* validity.

### **Step 2 — Activation extraction (what we read inside the model)**

During every Condition-A/B forward pass, hook the **residual stream** and cache activations at:

- **Layers:** a sweep of 6 layers spanning early/mid/late (e.g. layers 8, 12, 16, 20, 24, 28 of 32); prior work locates concept geometry mid-stack.  
- **Token positions:** (i) the final token of the problem statement, pre-generation (the "summary" position); (ii) mean-pool over the problem-statement tokens as a robustness alternative. Both are stored; analyses report which was used.

This yields, per (law ℓ, rung r, sample s, layer L), a vector hL(ℓ, r, s) ∈ ℝ^d. Storage  layers × hidden-dim × positions × examples (with 2 positions instead of full sequences, this stays in the tens of GB).

### **Step 3 — Geometry analysis (testing H1)**

1. **Per-law trajectory:** for each law, compute rung centroids ĥ(ℓ, r)  mean over samples; measure the intrinsic dimensionality of {ĥ(ℓ, 0), …, ĥ(ℓ, R−1)} (PCA variance explained by top 1–2 components). H1 predicts these points order along a low-dim curve *in rung order*.
2. **Cross-law parallelism:** compute per-law formality difference vectors Δ(ℓ)  ĥ(ℓ, literal-NL) − ĥ(ℓ, story); H1 predicts high pairwise cosine similarity across laws. More finely: fit a shared curve (spline through global rung centroids after subtracting each law's mean) and measure per-law residuals.
3. **Factorization probes:** (a) train a linear law-identity probe on one rung, test on all other rungs — transfer above chance  representation-invariant semantic core; (b) train a rung (formality) probe on a subset of laws, test on held-out laws — transfer  transportable formality direction; (c) **guardrail:** verify the law-identity subspace (top probe directions) cannot predict rung and vice versa, so we are not steering meaning when we intend to steer register.



### **Step 4 — Intervention (what we do inside the model, Phase 2**

Fit the manifold on a **train split of laws**; all steering evaluation is on **held-out laws** (transportability is the point). Let m(t), t ∈ 0,1, be the shared formality curve in the law-mean-subtracted space, with t=0 at the story centroid and t=t at the literal-NL centroid.

- **Manifold steering:** during a Condition-A forward pass at layer L, decompose h  (law component)  (formality component) via the Step-3 subspaces; replace the formality component's position on the curve: h ← h  m(t) − m(0), applied at the summary token (and, in a variant, at all subsequent generated-token positions). Sweep t from 0 → t and beyond (dose–response); sweep the intervention layer.  
- **Linear baseline:** h ← h  α·Δ̄, where Δ̄ is the mean story→literal-NL difference vector, α swept to match intervention norm with the manifold condition (norm-matched comparison, per Goodfire).  
- **Controls:** (i) random direction of matched norm; (ii) Δ from a *shuffled* law pairing; (iii) read the law-identity probe pre/post intervention — the law readout must be unchanged (drift  we are causing VBU mechanistically; logged and reported, not silently discarded).

**Readout:** generate RG under each intervention, score valid × faithful as in Step 1 H2 predicts: faithfulness-given-validity ordering C(manifold)  C(linear) ≥ A, with C(manifold) recovering a nontrivial fraction of the A→B gap, and validity degrading faster under linear steering at matched norms (the off-manifold analog of Goodfire's non-weekday outputs).

### **Step 5 — Bottleneck localization (Ke's question, free byproduct)**

For every Condition-A failure, record the story representation's distance to the literal-NL region and its position along the curve: failures far from the region  abstraction bottleneck; failures inside the region that still emit the wrong law  formalization bottleneck. Compare the distribution across models and complexity strata.

## **Interfaces with team questions**

Ke: failed cases localized geometrically (story representation far from literal-NL region  abstraction bottleneck; reaches region but emits wrong law  formalization bottleneck). Denis: do thinking tokens traverse the same manifold that steering shortcuts? Harsh: does swapping grammars move only the RG-end of the path, leaving the law-identity component fixed?

## **Kill criteria**

1. Two-hop does not beat one-hop in the no-think regime → no gap to explain; stop.
2. Rungs show no ordered low-dim structure, or paths are not shared across laws → nothing to steer; downgrade to descriptive finding.
3. Manifold steering ≤ linear steering ≤ no steering on faithfulness-given-validity → geometry is epiphenomenal; report negative result.



## **Cost estimate**

Phase 1: calls  examples × models × seeds × prompt-variants ≈ (60 laws × 6 rungs × 3 samples) × 2 models × 2 seeds ≈ 4–5k forward passes; activation cache ≈ tens of GB with layer/token restriction. Phase 2 adds intervention passes only on the story subset. Approximately 2–3 weeks for Phase 1 including rung generation and mechanical re-labelling.