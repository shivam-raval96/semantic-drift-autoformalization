# CLAUDE.md — FT Experiment: Grammar-Only Fine-Tuning (MARS V)

You are the engineering copilot for this experiment. Work step by step through the PHASES below,
in order, with the STOP checkpoints respected. Smoke-test before anything expensive. When an API or
library detail is uncertain (Unsloth/PEFT/TRL move fast), check the installed version's docs/source —
do not assume. Ask before launching training runs or large downloads.

---

## THE EXPERIMENT AT A GLANCE (this diagram is the source of truth)

```
TEST SET (frozen, identical for every model, never trained on):
  SAIR selected-problems evaluation subsets (evaluation_normal + evaluation_hard
  + evaluation_extra_hard + evaluation_order5, 800 implication problems)
      │  rendered through the repo's existing deterministic pipeline (Oren's)
      ▼
  story form + literal-NL form + reference rigid-grammar (RG) per problem
      = eval_v1  →  graded by checkform  →  correct / wrong / unparseable

TRAIN SET (grammar food only, never tested on):
  DIFFERENT equations (repo ETP pool + genform synthetics,
  law-disjoint from every SAIR problem — enforced by canonical pair-hash audit)
      │  same serializer → RG text only
      ▼
  ~3,000 samples of pure grammar text, e.g.:
      ASSUME: op(x, y) = op(op(y, y), x)
      ASK: op(x, y) = op(y, x)
  (no stories, no instructions, no NL — plain next-token prediction)

RUN ORDER:
  1. base model   → eval_v1 → score A
  2. FT model (grammar-only corpus) → SAME eval_v1 → score B
  3. compare A vs B (three-way verdicts, per tier)
  4. only then: LoRA rank sweep + hyperparameter details
```

Why train ≠ test equations: eval grades model output against the reference RG of each SAIR problem.
Fine-tuning on those same reference strings would hand the model the answer key (memorization, not
skill). The grammar is identical across all equations — only the specific laws differ — so grammar
fluency is what's allowed to transfer. That is the experiment.

**Core question:** does grammar-only fine-tuning improve story→RG / literal→RG translation on unseen
equations — and (stage 2) can the improvement be obtained in a maximally constrained LoRA setting
(rank 1, ONE layer) interpretable as a low-dimensional, steering-like direction (ΔW ≈ steering vector)?

---

## PROJECT CONTEXT

- Project: MARS V — semantic faithfulness in autoformalization. Substrate: Equational Theories
  Project (ETP): magma equational laws with a Lean-verified implication graph (~4,694 laws, 22M pairs).
  Ground truth is always mechanical; an LLM never labels anything.
- Repo: `https://github.com/shivam-raval96/semantic-drift-autoformalization.git` — I (Harsh) will
  clone it and create my branch (e.g. `harsh/ft-grammar`). New code lives in `ft-experiments/` on
  that branch.
- Read these modules FIRST (all under `informalizing-etp/` on `main`):
  - `storyform.py` — deterministic renderer: implication → themed story + literal NL + rigid grammar.
    Contains the equation AST/parser, `canonical()` (first-appearance variable renaming), `dual()`.
  - `checkform.py` — the grader. Defines the RG exactly (`ASSUME:`/`ASK:` lines, prefix `op(a, b)`
    terms), normalizes (canonicalization, equality flip, consistent dualization), and returns
    `correct | wrong | unparseable`. The FT corpus must match this grammar byte-for-byte.
  - `genform.py` — synthetic laws beyond the ETP 4-op cap (arbitrary depth).
  - `benchmark.py` — eval harness (loads ETP equations, builds prompts from `prompts/`, calls models,
    grades). Currently OpenRouter-backed; we add a local HF backend.
  - `experiments/` — prior findings. Relevant: 09 (unparseable output explodes with complexity),
    05/07 (two-stage), 11 (voting). Branch `certificate-pipeline` has `pipeline/hf_backend.py` as a
    reference for local HF generation.
- Standing rules that bind this work:
  - **R1**: mechanical ground truth only.
  - **R5**: NEVER train against our own checker — no RL, no filtering/selecting training data by
    checkform verdicts on model outputs. (Round-trip-verifying our own generated reference data is
    fine; that's data QA, not training signal.)
  - **R7**: translation (graded by checkform) and implication (True/False vs SAIR labels) are
    different tasks with different oracles — numbers never mixed in one table.
  - Vacuous laws (self-equalities that never invoke the op) are excluded everywhere.
  - Every data row carries provenance: source, law labels, ops/depth, tier, split, hash.

---

## LOCKED DECISIONS

- **Training objective:** plain next-token prediction (causal LM loss on all tokens + EOS) on the RG
  text field only. Packing on (samples are ~30 tokens). No chat template on training text.
- **Eval protocol (frozen):** no-think, fixed prompt templates from `prompts/` (unchanged), greedy
  decoding (`do_sample=False`), fixed max_new_tokens. Report three-way verdicts separately —
  correct / wrong-but-well-formed / unparseable — per tier (normal/hard/extra_hard/order5) and per arm
  (story→RG, literal→RG). Pooled accuracy is never the headline.
  - Rationale for no-think: thinking saturates this task (>95% in repo exp 01/02); CoT can internally
    rewrite the story (confound); Llama has no native thinking switch, so "thinking on" would be a
    prompt change — and then the prompt becomes the experiment.
- **Models:** start `Llama-3.1-8B-Instruct` (Unsloth 4-bit). Also base-eval `Llama-3.2-1B/3B-Instruct`
  as fast-iteration candidates (note: Llama **3.1** has no 1B — small ones are **3.2**). Mentor rule:
  pick the sweep model for SIGNAL — mid-to-low base performance (25%→75% is a result; 70%→75% is
  noise). 70B is a late expansion stage only.
- **Simple comparison first (Phase 5a):** one reasonable LoRA config (e.g. r=16, standard target
  modules, all layers), base vs FT on eval_v1. This is the headline "does it work at all" result.
- **Constrained stage after (Phase 5b, from mentor feedback):**
  - Stage A: rank 1, ONE layer only, everything else frozen — most interpretable test.
  - Stage B: rank sweep r ∈ {1, 2, 8, 16, 32, 64}, same single layer, everything else identical;
    parallelize if VRAM allows, else priority order 1, 16, 64, 2, 8, 32.
  - Layer choice (mine, documented as a choice): a middle transformer block, LoRA on `o_proj`
    (fallback `down_proj`) — a rank-1 delta there writes a fixed direction into the residual stream,
    i.e. the steering-vector picture. Same layer across the entire sweep.
  - Expand (more layers → other model size) only if A/B are flat.
- **Epochs:** default 3, decided by checkpoint eval curves — not a fixed number. Checkpoints saved at
  step 0 and every 50–100 steps, always (they also enable later representation-trajectory analysis).
- **Seeds:** greedy decode ⇒ no sampling noise; run 2–3 independent training seeds for the headline
  configs (Phase 5a config, rank 1, best rank), single runs elsewhere.

---

## DATA SPEC

### Eval set (eval_v1) — FINAL, no alternatives
- Source: HF dataset `SAIRfoundation/equational-theories-selected-problems`.
  Download ALL 9 subsets (tiny), but eval_v1 = ONLY the four evaluation subsets:
  `evaluation_normal`, `evaluation_hard`, `evaluation_extra_hard`, `evaluation_order5`
  (200 each, all balanced 100 TRUE / 100 FALSE → 800 problems, 4 difficulty tiers).
  The other 5 subsets (`normal`, `hard`, `hard1`, `hard2`, `hard3`) are NOT evaluated on;
  they exist only so the training-corpus disjointness audit covers all 2,669 SAIR rows.
- Gotchas (enforce):
  1. Key everything on the `equation1`/`equation2` STRINGS, never `eq1_id`/`eq2_id`
     (`evaluation_order5` uses a different equation-ID space than the 4,694-law list).
  2. The HF split is named "train" for every subset — HF convention only; nothing from
     this dataset is ever trained on.
  3. `evaluation_order5` equations are one op deeper than the standard ETP list — treat
     it as the built-in complexity-extrapolation tier; the repo parser handles the depth.
- Render each of the 800 problems through the repo pipeline into story / literal /
  reference-RG (seeded theme per problem, recorded). Failures to render: log, drop,
  report final N per tier. Keep the TRUE/FALSE `answer` as metadata only (optional
  secondary implication eval, reported separately per R7) — irrelevant to translation
  grading.
- Freeze as `eval_v1/`, version it, never modify. Any change = eval_v2 + rerun everything.

### Train corpus (train_v1)
- Source: repo ETP equations + `genform.py` synthetics. NOT SAIR.
- **Disjointness (hard gate):** canonical pair-hash every implication (canonicalize both laws;
  hash ordered pair; also collide under consistent dualization/side-swap). Drop any training pair
  whose pair-hash or individual-law hash appears in eval_v1. Log the drop count.
- Volume: 1,000 per complexity tier (easy/medium/hard by total op count, repo binning) ≈ 3,000.
  Plus a small deepest-depth holdout (~100 pairs) kept OUT of training as an extrapolation probe.
- JSONL rows — model sees `text` only:
  `{"text": "ASSUME: ...\nASK: ...", "e_label": ..., "f_label": ..., "ops_total": n, "max_depth": n,
    "tier": ..., "source": "etp|genform", "pair_hash": ..., "split": "train"}`
- Serializer: AST → prefix RG (`Var → name`, `Op(l, r) → op(l, r)`), variables x,y,z,w,u,v by first
  appearance. **Round-trip verify every sample**: it must re-parse under checkform's parser and
  canonicalize back to the source pair. Any mismatch = bug, stop and report.
- Deterministic generation, fixed sampler seed; commit the generator + hash manifest, not the bulk data.

---

## TRAINING SETUP (Unsloth — Shivam's recommendation)

- `pip install unsloth trl peft datasets accelerate bitsandbytes` (latest stable); verify GPU first.
- Skeleton (adapt to installed API, don't paste blindly):
  ```python
  from unsloth import FastLanguageModel
  model, tokenizer = FastLanguageModel.from_pretrained(
      "unsloth/Meta-Llama-3.1-8B-Instruct", max_seq_length=1024, load_in_4bit=True)
  model = FastLanguageModel.get_peft_model(
      model, r=RANK, lora_alpha=RANK, lora_dropout=0.0, bias="none",
      target_modules=[...], random_state=SEED)
  # TRL SFTTrainer on the 'text' field, packing=True, save_steps=50-100,
  # fixed LR (~2e-4 cosine), fixed batch/steps — log every hyperparameter.
  ```
- **Single-layer restriction (verify in smoke test):** Stage A/B need LoRA on ONE layer only. PEFT's
  `LoraConfig(layers_to_transform=[K])` does this — check Unsloth passes it through. If not, use
  vanilla PEFT+TRL for constrained runs (Unsloth for the broad ones). Prove it by printing trainable
  parameter names (all must live in layer K) and the trainable-param count, every run.
- **Format-bridge sanity probe (run after the first FT, before interpreting eval numbers):**
  we train raw RG text but eval via chat-formatted prompts. Verify learning happened at all:
  (a) FT-model perplexity on held-out RG text ≪ base; (b) raw completion of
  `ASSUME: op(x, y) = op(op(y, y), x)\nASK:` is fluent RG. If (a)/(b) pass but chat-eval is flat,
  the issue is format surfacing, not LoRA capacity — bring me options (e.g. minimal completion-style
  wrapper) instead of concluding "rank too low."

---

## PHASES (walk me through these in order)

- **Phase 0 — Environment:** clone repo, create my branch, `nvidia-smi`, install deps, run repo
  `tests/` to confirm the grader passes locally.
- **Phase 1 — Recon:** read the modules listed above; summarize back to me: the exact RG surface
  checkform accepts, how benchmark.py builds prompts, what's reusable vs to-write. **STOP — confirm.**
- **Phase 2 — Data:** get SAIR subsets → build hash index → render eval_v1 → generate train_v1 +
  extrapolation holdout → round-trip verification + disjointness audit. Show me: counts per tier,
  drop counts, 5 sample rows from each artifact. **STOP — sign-off.**
- **Phase 3 — Base evals:** run 8B (+1B/3B) base on eval_v1 via the new local backend; produce the
  base table (score, runtime, failure modes); recommend the sweep model per the signal rule.
  **STOP — my choice.**
- **Phase 4 — Smoke FT:** ~200 samples, 50 steps, rank-1 single-layer: verify trainable-param
  isolation, checkpoint saving, sanity probes. Fix issues before scaling.
- **Phase 5a — Headline comparison:** full FT (standard config, r=16 all-layers), 3 epochs w/
  checkpoints → eval FT on eval_v1 → base-vs-FT three-way table per tier. This answers "does grammar-
  only FT help translation at all."
- **Phase 5b — Constrained stages:** Stage A (rank-1/one-layer) → Stage B (rank sweep, same layer)
  → expand only if flat. Checkpoint-curve evals for rank 1 and best rank.
- **Phase 6 — Analysis:** config table · rank-vs-performance plot with base line · per-tier delta
  table · 2–3 qualitative base-vs-FT output pairs · extrapolation-holdout numbers · `RESULTS.md` in
  the repo's experiment-card style (question / setup / results / conclusions / kill verdict).
- Interpretation guide: rank-1 works → low-dimensional/steering story (next: extract ΔW direction,
  compare to an activation-difference steering vector) · only higher ranks → capacity story, no 1-D
  claim · only more layers → not localized · nothing → check signal/model/format before concluding.

---

## LOGGING (every run) & GUARDRAILS

Log per run (JSON in `ft-experiments/runs/`): model · dataset version + hash manifest · split IDs ·
prompt template id · rank · target layer(s)/modules · trainable-param count · steps/epochs · LR ·
batch · seq len · seed · train-loss curve · eval three-way rates per tier per arm · runtime · GPU ·
notes. Commit configs before launching sweeps.

1. One change at a time — the sweep varies rank ONLY; anything else moving invalidates the run.
2. R5: never use checkform verdicts on model outputs to filter/select/weight training data.
3. eval_v1 is frozen; changes create eval_v2 and rerun everything downstream.
4. Translation and implication numbers never share a table (R7).
5. Eval tiers are 200 items each — ±3–4 points is noise; don't narrate small deltas;
   report N everywhere.
6. Kill criterion (stated upfront): if grammar-only FT (Phase 5a) moves neither the unparseable rate
   nor correct% beyond noise on any tier — and the sanity probes confirm learning happened — the
   grammar-bottleneck hypothesis is rejected; write it up and stop rather than tuning until something
   moves.
