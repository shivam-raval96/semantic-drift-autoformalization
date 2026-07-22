# Background Literature Review

**Topic:** deterministically rendering formal mathematical statements as
natural-language stories, used as (i) a fidelity-guaranteed corpus-generation
method and (ii) a formalization task for evaluating models on non-standard
phrasings.

This review situates the storyform idea against five bodies of prior work:
autoformalization and LLM-based informalization; deterministic
formal-to-natural-language rendering; math word problems as narrative
wrappers around equations; robustness and contamination studies of
surface-form sensitivity; and evaluation of formalization outputs. It closes
with a gap analysis.

---

## 1. Autoformalization and informalization

Autoformalization — translating informal mathematics into a formal language
such as Lean or Isabelle — predates LLMs: Wang, Kaliszyk & Urban trained
neural MT models on aligned Mizar/LaTeX data (arXiv:1805.06502;
arXiv:1912.02636), already noting that generating *informal* text from
formal statements is the easier direction and useful for creating synthetic
parallel data. Wu et al. (NeurIPS 2022, arXiv:2205.12615) showed that large
models (Codex, PaLM) can few-shot-autoformalize competition problems into
Isabelle, establishing the modern LLM framing. A 2025 survey
(arXiv:2505.23486) covers the field end-to-end.

The direct ancestor of "render formal statements as informal text" is the
**informalization / back-translation** line:

- **ProofNet** (Azerbayev et al., arXiv:2302.12433) introduced distilled
  back-translation: informalize formal statements with an LLM, then train
  the reverse direction.
- **MMA** (Jiang et al., arXiv:2311.03755; NeurIPS 2024 version
  "Multi-language Diversity Benefits Autoformalization") informalized all of
  Mathlib4 and the Isabelle AFP with GPT-4, producing 332K informal–formal
  pairs, on the explicit argument that informalization is much easier than
  formalization. Fine-tuning on MMA lifted acceptable-with-minor-corrections
  autoformalization from 0% to ~16–31% on miniF2F/ProofNet.
- **"Lean-ing on Quality"** (arXiv:2502.15795) found that *quality* of
  back-translated pairs beats volume: curated back-translation with 1/150th
  of MMA's tokens outperformed it, i.e. noise in the informal side is a real
  cost.

**Key contrast with storyform:** in all of this work the informal side is
*LLM-generated*, so fidelity is probabilistic and must be filtered or
audited after the fact; the informal register is standard "textbook
mathematical English," heavily represented in pretraining data. Storyform
inverts both properties: the informal text is produced by a pure function
(fidelity holds by construction, invertibility is enforced by a
back-parser), and the register is deliberately *non-mathematical* narrative,
outside the textbook distribution.

## 2. Deterministic rendering of formal content into natural language

The mechanics of the idea — template/grammar rendering of machine-generated
formal objects, with the formal object retained as ground truth — has a long
lineage in synthetic reasoning benchmarks:

- **bAbI** (Weston et al., arXiv:1502.05698) generated stories from a world
  simulation and verbalized them with templates ("stringify"), so every
  question's answer is grounded in the simulation state.
- **RuleTaker** (Clark et al., arXiv:2002.05867) and **ProofWriter**
  (Tafjord et al., Findings ACL 2021) generated logic programs, computed
  their consequences with a theorem prover, and rendered facts/rules to
  English through simple natural-language templates. The English is
  synthetic but the label provenance is exact — the same "fidelity by
  construction" property storyform relies on.
- **Unigram** (EMNLP 2024, "Scaling Synthetic Logical Reasoning Datasets
  with Context-Sensitive Declarative Grammars") generalizes this to
  declarative two-sided grammars binding FOL and natural language, with
  solver-checked labels — the closest engineering analogue of a rendering
  grammar designed to be injective.
- **MuSR** (Sprague et al., ICLR 2024, arXiv:2310.16049) is the
  LLM-in-the-loop contrast: reasoning trees are symbolic, but GPT-4 writes
  the ~1000-word narrative. The stories are far more natural than templated
  text, at the price of exactly the fidelity risk storyform's invariants
  ban ("do not improve stories with LLM rewriting inside the pipeline").

A second, older tradition renders formal mathematics specifically:

- **Controlled natural languages** for mathematics — Attempto Controlled
  English (Fuchs et al.) and **Naproche** (Cramer et al.) — define a
  bidirectional, machine-checkable fragment of mathematical English. They
  target the *formal-looking* register (variables, formulas embedded in
  prose), the opposite design point from storyform's "no formal leakage."
- **Grammatical Framework** work — the GF Mathematical Grammar Library and
  the ongoing **Informath** project (Ranta) — translates between formal
  systems (Dedukti, OpenMath) and natural language via abstract syntax
  trees, in both directions, deterministically. Informath explicitly notes
  that *verbal* (word-based) renderings are needed to make informalization
  failure-free. This is the closest formal ancestor of a
  provably-invertible math verbalizer, though again its output is
  mathematical prose, not themed narrative.
- Data-to-text surface realization (e.g. rule-based realizers for knowledge
  graphs) shares the machinery but not the math focus.

**Gap:** none of these render *research-grade formal statements* (as opposed
to toy rulebases) as *fully de-mathematized stories*, and none use the
rendering as a stress test for autoformalization specifically.

## 3. Math word problems: stories wrapping equations, and the reverse task

Math word problems are the canonical "story whose meaning is an equation."
The classic formalization-direction result is Kushman et al. (ACL 2014),
which mapped algebra word problems to systems of equations via templates and
alignments — literally the storyform eval in miniature, for linear algebra
rather than universal-algebra implications. Later robustness work is
instructive:

- **SVAMP** (Patel et al., NAACL 2021, arXiv:2103.07191) showed MWP solvers
  collapse under simple variations that preserve the underlying equation —
  early evidence that models bind to surface patterns, not the formal
  content.
- **GSM-Symbolic** (Mirzadeh et al., ICLR 2025, arXiv:2410.05229) rebuilt
  GSM8K from symbolic templates; performance varies across instantiations
  of the *same* template, degrades when only numbers change, and drops up
  to 65% when an irrelevant clause (GSM-NoOp) is added.

The generation direction (equation → word problem) also has a literature
(MWP generation from equations/templates), but there the goal is
pedagogical fluency, with fidelity checked heuristically — not guaranteed
injectivity.

**Contrast:** storyform stories are not word problems — the model is not
asked to *solve* anything, only to recover the statement. This isolates the
translation/reading step that MWP benchmarks confound with arithmetic
and planning.

## 4. Robustness, contamination, and surface-form sensitivity

The motivation for testing formalization on non-standard phrasings is
well-supported:

- **"Reasoning or Reciting?"** (Wu et al., NAACL 2024, arXiv:2307.02477):
  across 11 counterfactual task variants that preserve the reasoning but
  change default conditions, performance degrades consistently — models
  partly overfit to the default surface form of a task.
- **Putnam-AXIOM** (ICML 2025, arXiv:2508.08292) and **MATH-Perturb**
  (ICML 2025, arXiv:2502.06453): programmatic perturbations of competition
  problems cause large accuracy drops (o1-preview −46.8% relative on
  Putnam variations; o1-mini −16.5% on MATH-P-Hard), including a failure
  mode of *blindly applying memorized templates* to modified contexts.
- GSM-Symbolic/GSM-NoOp (above) show the same at grade-school level.

Storyform occupies a distinctive point in this space: instead of perturbing
a known benchmark item, it re-skins the entire *register* while keeping the
formal semantics exact and machine-checkable, and it can generate unlimited
items from the ETP's 22M implications — contamination-resilient in the same
sense as the functional benchmarks, but with the ground truth trivially
exact rather than re-derived.

## 5. Grading formalization outputs

How to score a model's formal output is a recognized open problem when the
informal side comes first:

- Typechecking/compile success is necessary but nowhere near sufficient.
- **BEq** (Liu et al., ICLR 2025, "Rethinking and Improving
  Autoformalization") checks bidirectional definitional equivalence between
  candidate and reference formalizations with neuro-symbolic proving;
  **BEq+ / ProofNetVerif** (Poiroux et al., EMNLP 2025, arXiv:2406.07222)
  makes this deterministic and CPU-only and contributes 3,752
  human-annotated equivalence labels for metric benchmarking.
- **"The Faithfulness Gap"** (arXiv:2606.16541) frames informal↔formal
  *semantic drift* directly and certifies equivalence via bidirectional
  provability fingerprints, with a drift-detection theorem — the same
  "drift" vocabulary as this repo, approached from the certification side.

Because storyform's informal text is generated *from* the formal statement
by an injective grammar, its grader (`checkform.py`) can be purely
syntactic — parse, canonicalize, compare modulo a small, explicitly
enumerated symmetry group (renaming, side swap, uniform dualization). No
prover, no LLM judge, no annotated equivalence labels. This sidesteps the
entire metric problem that BEq/BEq+ exist to solve, at the cost of only
covering statements the grammar can express.

## 6. The substrate: the Equational Theories Project

The ETP (Tao et al.; report arXiv:2512.07087) determined all 22,028,942
implications among the 4,694 simplest magma laws, Lean-verified, finishing
14 April 2025. As a corpus source it has unusual properties: statements are
(a) formally exact and uniform in shape (one binary op, ≤4 operations per
side), (b) essentially absent from natural-language pretraining data (they
were never written up informally at scale), and (c) available in the
millions with known truth values that the storyform pipeline deliberately
ignores. Prior ETP-adjacent ML work uses it as a theorem-proving/ATP
benchmark; using it as an *informalization substrate* appears to be novel.

## 7. Positioning and gaps

Combining the threads, the storyform idea sits at an unoccupied
intersection:

| Prior work | Informal side | Fidelity | Register |
|---|---|---|---|
| MMA / ProofNet back-translation | LLM-generated | probabilistic, filtered | textbook math English |
| RuleTaker / ProofWriter / Unigram | templates/grammar | by construction | stilted synthetic English, toy logic |
| MuSR | LLM narrative over symbolic spec | audited, not guaranteed | rich narrative |
| CNL / GF Informath | deterministic grammar | by construction, invertible | mathematical prose with symbols |
| MWP benchmarks (SVAMP, GSM-Symbolic) | human or template | exact answers | word problems, solving conflated with reading |
| **Storyform** | **deterministic themed narrative** | **by construction + round-trip enforced** | **pure story, zero formal leakage** |

Specific claims of novelty the literature supports:

1. **Fidelity by construction for *narrative* informalization.** Grammar
   determinism and invertibility are known techniques (ProofWriter,
   Informath), but no prior work applies them to produce fully
   de-mathematized stories paired with research-grade formal statements.
2. **Formalization-from-story as an eval.** NL→FOL benchmarks (FOLIO;
   MALLS, ACL 2024) and autoformalization benchmarks (miniF2F, ProofNet)
   all use mathematical or quasi-formal register. An eval that requires
   recovering exact term trees, argument order, and quantifier structure
   from deliberately non-mathematical prose — graded syntactically against
   ground truth — does not exist in the surveyed literature.
3. **Contamination-proof by register, not just by perturbation.**
   Functional benchmarks (GSM-Symbolic, Putnam-AXIOM) fight contamination
   by regenerating instances; storyform additionally moves to a phrasing
   distribution that cannot have been memorized because it never existed
   before rendering.

Risks/limitations the literature flags for this approach:

- **Templated text is its own distribution.** RuleTaker-style English is
  recognizably synthetic; models may learn the rendering grammar itself
  rather than general story-reading (an issue if the corpus is used for
  training rather than eval). Theme diversity mitigates but does not
  eliminate this.
- **Narrow mathematical scope.** One binary operation and equational
  implications cover a tiny fragment of what autoformalization benchmarks
  target; conclusions about "formalization robustness" transfer only to
  the statement-reading skill, not to definition grounding or library
  alignment (the RAutoformalizer/dependency-retrieval problem).
- **Distance from deployment.** BEq+/ProofNetVerif-style evaluation exists
  precisely because real autoformalization targets Lean against Mathlib;
  the storyform eval's `op(...)` mini-language deliberately removes that
  dimension. This is a feature for isolation, a caveat for external
  validity.

---

## Bibliography

- Wang, Kaliszyk, Urban. *First Experiments with Neural Translation of
  Informal to Formal Mathematics.* arXiv:1805.06502; and
  *Exploration of NMT in Autoformalization of Mathematics in Mizar.*
  arXiv:1912.02636.
- Wu, Jiang, Li, Rabe, Staats, Jamnik, Szegedy. *Autoformalization with
  Large Language Models.* NeurIPS 2022. arXiv:2205.12615.
- Azerbayev, Piotrowski, Schoelkopf, Ayers, Radev, Avigad. *ProofNet.*
  arXiv:2302.12433.
- Jiang, Li, Jamnik. *Multilingual Mathematical Autoformalization / 
  Multi-language Diversity Benefits Autoformalization.* NeurIPS 2024.
  arXiv:2311.03755.
- *Lean-ing on Quality: How High-Quality Data Beats Diverse Multilingual
  Data in AutoFormalization.* arXiv:2502.15795.
- *Autoformalization in the Era of Large Language Models: A Survey.*
  arXiv:2505.23486.
- Weston, Bordes, Chopra, Rush, van Merriënboer, Joulin, Mikolov. *Towards
  AI-Complete Question Answering: A Set of Prerequisite Toy Tasks (bAbI).*
  ICLR 2016. arXiv:1502.05698.
- Clark, Tafjord, Richardson. *Transformers as Soft Reasoners over Language
  (RuleTaker).* IJCAI 2020. arXiv:2002.05867.
- Tafjord, Dalvi, Clark. *ProofWriter.* Findings of ACL 2021.
- Sileo. *Scaling Synthetic Logical Reasoning Datasets with
  Context-Sensitive Declarative Grammars (Unigram).* EMNLP 2024.
- Sprague, Ye, Bostrom, Chaudhuri, Durrett. *MuSR.* ICLR 2024.
  arXiv:2310.16049.
- Cramer et al. *The Naproche Project: Controlled Natural Language Proof
  Checking of Mathematical Texts.* CNL 2009. Fuchs et al., *Attempto
  Controlled English.*
- Ranta et al. *GF Mathematical Grammar Library*; *Informath* (GitHub:
  GrammaticalFramework/informath).
- Kushman, Artzi, Zettlemoyer, Barzilay. *Learning to Automatically Solve
  Algebra Word Problems.* ACL 2014.
- Patel, Bhattamishra, Goyal. *Are NLP Models Really Able to Solve Simple
  Math Word Problems? (SVAMP).* NAACL 2021. arXiv:2103.07191.
- Mirzadeh, Alizadeh, Shahrokhi, Tuzel, Bengio, Farajtabar. *GSM-Symbolic.*
  ICLR 2025. arXiv:2410.05229.
- Gulati, Miranda, et al. *Putnam-AXIOM.* ICML 2025. arXiv:2508.08292.
- Huang, Guo, et al. *MATH-Perturb.* ICML 2025. arXiv:2502.06453.
- Wu, Qiu, et al. *Reasoning or Reciting? Exploring the Capabilities and
  Limitations of Language Models Through Counterfactual Tasks.* NAACL 2024.
  arXiv:2307.02477.
- Yang, Xiong, Payani, Shareghi, Fekri. *Harnessing the Power of LLMs for
  NL-to-FOL Translation (MALLS, LogicLLaMA).* ACL 2024. arXiv:2305.15541.
- Liu et al. *Rethinking and Improving Autoformalization: Towards a
  Faithful Metric and a Dependency Retrieval-based Approach (BEq,
  RAutoformalizer, Con-NF).* ICLR 2025.
- Poiroux et al. *Reliable Evaluation and Benchmarks for Statement
  Autoformalization (BEq+, ProofNetVerif, ProofNet#, RLM25).* EMNLP 2025.
  arXiv:2406.07222.
- *The Faithfulness Gap: Certifying Semantic Equivalence Between
  Natural-Language and Formal Mathematical Statements.* arXiv:2606.16541.
- Bolan et al. (Equational Theories Project). *The Equational Theories
  Project: Advancing Collaborative Mathematical Research at Scale.*
  arXiv:2512.07087. Repo: github.com/teorth/equational_theories.
