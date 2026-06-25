## LADR dataset (Linear Algebra Done Right)

This folder contains material gathered from my other project based on the undergraduate textbook “Linear Algebra Done Right.” It organizes key pedagogical elements — definitions, theorems, examples, and exercises — as JSONL files to support experiments in natural-language-to-formal proof translation for Lean 4.

### Contents
- LADR_defs_136.jsonl — 136 definitions
- LADR_thms_256.jsonl — 256 theorems, each with an accompanying informal (natural language + math notation) proof
- LADR_examples_134.jsonl — 134 worked examples
- LADR_exercises_733.jsonl — 733 exercises (problem statements only)
- LADR_pilot_27.jsonl — a small pilot subset of theorems (with informal proofs) for quick iteration

All files are JSON Lines; one JSON object per line.

### JSON schema (observed)
Common fields:
- name: string identifier (e.g., "LADR_thm_chap_1_14", "LADR_ex_1.1A.1")
- domain: always "LADR" in this dataset
- nl_statement: the natural-language statement (may include TeX-style math)

Additional fields by file:
- Theorems (LADR_thms_256.jsonl, LADR_pilot_27.jsonl)
  - informal_proof: natural-language proof (often with TeX/align environments)
- Examples (LADR_examples_134.jsonl)
  - title: short descriptive title

Note: TeX math and environments appear inline; your preprocessing may need to strip or preserve them depending on the modeling approach.

### Motivation
Theorems are stated in natural language, and their proofs are given in natural language with math notation. Many LLMs struggle to directly produce verified Lean 4 code from a raw theorem statement, but leveraging the accompanying informal proof may simplify the translation task.

### Proposed experiments
We propose three experiments to measure the effect of providing informal proofs and of a multi-stage translation strategy:

1) NL theorem → formal Lean 4 (baseline)
   - Input: nl_statement
   - Task: produce Lean 4 theorem statement and a verified proof.

2) NL theorem + informal_proof → formal Lean 4
   - Input: nl_statement + informal_proof
   - Task: produce Lean 4 theorem statement and a verified proof, explicitly using the informal proof as guidance.

3) Multi-stage pipeline using informal_proof
   - Stage A: Translate informal_proof into an “informal Lean-style proof sketch” (structured Lean tactics/comments or a mathlib-style skeleton without full details).
   - Stage B: Combine nl_statement + the proof sketch to produce a fully verified Lean 4 proof.

### Evaluation metrics
Report metrics per experiment and compare across them:
- Formalization success (statement): percentage where the generated Lean statement typechecks.
- Proof verification success: percentage where Lean accepts the full proof (build/compile succeeds).
- pass@k (optional): success when sampling k independent generations.
- Delta vs. baseline: improvement of (2) and (3) over (1) on verification success.
- Time/steps (optional): tokens used, average proof length, number of tactic steps.
- Optional semantic checks:
  - When a canonical formal statement exists, compare normalized forms (pretty-printed AST or definally-equal variants). Otherwise, use human review on a sampled subset.

Success criteria should primarily be Lean verification; string exact match is not required if the statement is equivalent and the proof verifies.

### Prompt templates (sketch)
These are minimal examples to standardize prompting; adapt as needed.

Experiment 1 (baseline):
```text
System: You are a Lean 4 formalization assistant. Produce self-contained Lean 4 code that typechecks in mathlib4. Prefer structured tactic proofs when helpful.
User:
THEOREM (natural language):
<nl_statement>

Output:
-- theorem name derived from `name`
-- required imports
theorem <name_or_reasonable_snake_case> : <Lean statement> := by
  <proof that verifies>
```

Experiment 2 (append informal_proof):
```text
System: You are a Lean 4 formalization assistant. Use the provided informal proof to guide the Lean proof. Produce self-contained Lean 4 code that verifies.
User:
THEOREM (natural language):
<nl_statement>
INFORMAL PROOF:
<informal_proof>

Output:
theorem <name> : <Lean statement> := by
  -- Follow this high-level plan from the informal proof:
  -- <bullet points auto-extracted>
  <verified Lean proof>
```

Experiment 3 (multi-stage):
Stage A (proof sketch):
```text
System: Translate the informal proof into a Lean-flavored proof sketch: tactic outline, key lemmas, and subgoals. Do not try to fully formalize.
User:
<informal_proof>

Output:
- goal structure
- lemma references (by name when known; otherwise description)
- tactic outline (pseudo-Lean)
```
Stage B (formalization using sketch):
```text
System: You are a Lean 4 formalization assistant. Use the provided theorem and proof sketch to produce a verified Lean 4 proof.
User:
THEOREM:
<nl_statement>
PROOF SKETCH:
<structured outline from Stage A>
```

### Practical notes
- Start with the pilot set (LADR_pilot_27.jsonl) to iterate quickly.
- Ensure the Lean environment (Lean 4 + mathlib4 + lake) is reproducible across runs.
- Normalize LaTeX (e.g., replace \mathbf{F} by `𝔽` or `F` consistently) and names to match mathlib conventions.
- Consider providing canonical names (e.g., add a mapping file) if you later derive gold Lean statements.

### Attribution and usage
This dataset was compiled from my own notes and project derived from “Linear Algebra Done Right” for research and evaluation purposes. The natural language content may reflect textbook phrasing or standard mathematical exposition. Please use respectfully and for research/educational evaluation only; consult the textbook for authoritative statements and proofs.

