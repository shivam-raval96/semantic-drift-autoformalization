# Law-Forced Manifolds: Does an ETP Law Leave a Controllable Geometry?

**One-page proposal — MARS V / mech-interp**  
Adapted from Goodfire manifold steering (Wurgaft et al., arXiv:2605.05115). For senior review.

## Motivation

Goodfire shows that when a model reasons over a concept domain with known structure (weekdays on a circle, letters on a line), its activations and next-token distributions lie on approximately isometric manifolds $M_h$ and $M_y$, and steering along $M_h$ produces natural behavioral trajectories that linear (diff-in-means) steering does not.

Our project asks a related but harder question: does an **ETP equational law** leave such a geometry, and does narrative surface form distort it? Prior work here is suggestive but incomplete:

- Mid-stack activations of Qwen3-4B encode **law identity** across themed stories (86% cross-theme 1-NN at L18–24; chance 1.9%; length-only floor 24.5%).
- Linear story→literal steering closes **0%** of the huge thinking gap (9.6% → 90.4% correct).
- Representation and behavior are dissociated: the law is readable; formalization still fails.

A direct port of Goodfire onto Story→RG fails a methodological prerequisite: their behavior manifold is the next-token distribution over a small concept set $Z$ on a task the model is competent at. Full `ASSUME:`/`ASK:` strings are multi-token and at floor accuracy, so there is no $M_y$ to fit. We need a task where an ETP law *forces* a small, ordered answer domain.

## Core idea

Construct a finite magma whose operation the storyform grammar can already narrate, such that a chosen ETP law forces a cyclic (or sequential) orbit on a small palette. Prompts ask for the result of an $n$-step chain; the answer is a single palette token. Surface form (symbolic / literal-NL / themed story) is the experimental variable.

**Toy instance.** Palette $\{c_0,\ldots,c_{k-1}\}$; habit “pour $a$ into $b$ yields $\mathrm{succ}(b)$.” That habit is an equational law; the law forces a $k$-cycle. Question: “start at $c_i$, pour $n$ times — which color?” Answer $\in Z$, $|Z|=k$, ground-truth metric = cyclic distance. Vary starting element (not chain length) so prompt length is constant across concept values — our data already show story length correlates $r=0.93$ with operation count, so complexity-indexed axes are confounded.

This is closest to Goodfire’s in-context graph setting: geometry is induced by structure in the prompt, not by pretraining co-occurrence of weekdays. Whether the model builds the cycle under narrative wrapping is part of the question.

## Hypotheses

**H1 (competence gate).** On the symbolic/literal form of the orbit task, the model is above-chance accurate and errors concentrate on metric neighbors (neighbor spill). If this fails, stop — there is no $M_y$.

**H2 (geometry).** For a law-forced cycle, both $M_h$ (activation centroids) and $M_y$ (Hellinger-embedded answer distributions) recover cyclic order; geodesic distances on $M_h$ and $M_y$ correlate more strongly than Euclidean distances in activation space (Goodfire’s isometry test).

**H3 (surface-form deformation).** Fitting $M_h,M_y$ separately per form: narrative themes either (a) preserve the cycle (shared geometry, dialect is a nuisance factor), (b) flatten it (higher intrinsic dim / weaker isometry), or (c) warp neighbor structure. (a) vs (b)/(c) is the project-relevant claim about semantic faithfulness under informalization.

**H4 (causality).** Manifold steering along $M_h$ yields ordered transitions through adjacent palette tokens and lower cumulative distance to $M_y$ than norm-matched linear steering (Goodfire’s energy comparison). Optional Phase 2: pullback from $M_y$ recovers $M_h$.

## Methodology (sketch)

1. **Dataset.** Deterministic finite magmas + orbit prompts; $k\in\{5,7\}$ cycles, plus an open-chain control law. Render each instance in RG, literal-NL, and 4 story themes via existing `storyform`/`literalform`. Mechanical invertibility already enforced by the renderer.
2. **Model / site.** Qwen3-4B (consistent with prior exps); residual stream at last prompt token; layer sweep around L18–30 (prior law-retrieval peak).
3. **Fit (Goodfire protocol).** Per concept value: activation centroids → 64-dim PCA → periodic cubic spline ($M_h$); answer distributions → $\sqrt{p}$ Hellinger → spline on the sphere ($M_y$). Isometry = Pearson $r$ of pairwise geodesic distances; MDS for structure plots.
4. **Steer.** Replace top-64 PCA components along intrinsic geodesic vs straight line in ambient space; report ordered mass transitions and cumulative Bhattacharyya energy to $M_y$. Controls: random path, shuffled concept pairing, length-matched prompts.
5. **Pilot first.** Before full manifold fit: accuracy + neighbor-error rate by form. Kill if symbolic competence fails.

## Kill criteria

1. Model not competent on symbolic/literal orbit task, or errors are not neighbor-structured → no behavior manifold; stop.
2. $M_h$ or $M_y$ fails to recover cyclic order under symbolic form → law does not induce the intended geometry; redesign magma or abandon.
3. Isometry and manifold-vs-linear advantage hold only for symbolic form and collapse under all narrative themes → report as a surface-form boundary on Goodfire’s account (still a result).
4. Manifold steering ≤ linear on energy / ordering under forms where geometry exists → geometry is epiphenomenal for control; report negative.

## Why this is worth doing even if H4 fails

A clean negative on the isometry under narrative form, with a positive under symbolic form, would be a **documented boundary condition** on “representation geometry causally shapes behavior” in a domain the project already cares about (informalized equational content). That is more informative than another null linear-steering result on Story→RG. A positive result gives a controllable semantic handle tied to a law, not a surface register.

## Cost / feasibility

Pilot: a few hundred forward passes, single-token readout — cheap on Colab. Full Phase 1 (centroids, isometry, steer): thousands of forward passes, activations in the low-GB range with layer/token restriction. Reuses existing HF hook infrastructure; main new pieces are PCA-subspace *replacement* (not additive) interventions and Hellinger spline fitting. Pullback optimization is optional Phase 2.

## Open design choices for review

- Magma choice: which concrete law forces the cleanest cycle with our 6-name palette and no accidental symmetries?
- Should the primary claim be “laws induce manifolds” (vary the law / topology) or “narrative deforms a fixed law’s manifold” (vary form)? Recommended: fix one cycle law, vary form first; topology sweep second.
- Relation to formality-ladder proposal (MARS §7): that proposal has no $M_y$ (formality is not a next-token concept set). Treat as a cheap activation-only add-on, not a substitute for this design.
