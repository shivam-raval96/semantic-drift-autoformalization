# CLAUDE.md — MARS V (Harsh's threads)

You are the engineering copilot for Harsh Soni's work on **MARS V: semantic
faithfulness in autoformalization**. This file is the handoff: what the project
is, what has been run, what the numbers are, how Harsh works, and what breaks.
Read it fully before acting.

Branch: `harsh/experiments` (everything below is committed and pushed there).
Last updated 2026-08-23.

---

## THE PROJECT

Autoformalization = turning maths stated in natural language into machine-checkable
formal statements. Standard evaluation is broken: people check whether output
*compiles*, or ask an LLM to judge it. Neither detects **semantic drift** — a
formalization that is well-formed and provable but means something different from
the source. Measuring that honestly is the whole point.

Substrate: the **Equational Theories Project** — 4,694 magma equational laws with a
Lean-verified implication graph. Ground truth is mechanical; an LLM never labels
anything.

Pipeline runs *backwards* from usual practice: take a formal implication → render
deterministically into a themed story (all notation stripped) → ask a model to
recover the formal statement → grade with a deterministic parser. Because the story
was generated from a known formal object, the right answer is known exactly.
Verdicts are three-way: **correct / wrong-but-well-formed / unparseable**.

### Standing rules (bind all work)

- **R1** mechanical ground truth only; no LLM judging.
- **R5** never train against our own checker — no RL, no filtering training data by
  checkform verdicts on model outputs. (Round-trip-verifying our own generated
  reference data is fine; that's QA, not training signal.)
- **R7** translation (checkform-graded) and implication (True/False) are different
  tasks with different oracles — numbers never share a table.
- Vacuous laws excluded everywhere.
- `eval_v1` is frozen. Any change = eval_v2 + rerun everything downstream.
- Every data row carries provenance: source, law labels, ops/depth, tier, split, hash.
- One change at a time. Report N everywhere; tiers are 180–200 items so ±3–4 points
  is noise.

### Repo modules owned by others (read, don't edit)

Under `informalizing-etp/` on `main` (Oren's pipeline):
- `storyform.py` — deterministic renderer: implication → themed story + literal NL.
  Contains the equation AST (`Var`, `Op` frozen dataclasses), `canonical()`, `dual()`.
- `checkform.py` — THE grader. Defines the rigid grammar (`ASSUME:`/`ASK:`, prefix
  `op(a, b)`), normalizes under the symmetry group (variable renaming, side swap,
  uniform dualization), returns correct/wrong/unparseable.
- `genform.py` — synthetic laws beyond the ETP 4-op cap.
- `benchmark.py` — eval harness, prompt wrappers, `bucket_of`.
- `themes/` — 4 story themes: **paint, graft, signal, tea**.

---

## STATE: everything is complete and pushed. No jobs running.

Three finished threads. All numbers below are independently re-verified (see QC).

### Thread A — FT v1: grammar-only fine-tuning (KILLED, written up)

Question: is notation fluency the bottleneck? Train on bare `ASSUME:/ASK:` text only.

**Answer: no.** Syntax became perfect, semantics did not move, and in the capable
model it collapsed:

| model / arm | correct base → FT | unparseable base → FT |
|---|---|---|
| 8B story | 0.1% → 0.0% | 12.2% → 32.3% (runaway) |
| 8B literal | 0.4% → 0.0% | 22.8% → 0.0% |
| 32B literal | **34.4% → 3.6%** | 5.7% → 0.1% |
| 32B two-stage | **34.9% → 3.3%** | 6.3% → 0.5% |

Verdict: grammar-only continuation training is a **behavioral override, not skill
injection**. It buys syntax, adds no semantics, and displaces existing translation
ability. Representation check: law-disjoint probe 0.52–0.53 before AND after —
training changed how the model writes, not what it represents.

### Thread B — FT v2: task-pair fine-tuning + cross-grammar generalization (DONE)

Mentor design (Oren/Denis): *"train on the pairs then test on a different rigid
grammar — then we're really seeing if the fine-tuning task generalizes."*

Train on story→answer pairs in **grammar A only**, completion-only loss, then eval
the same frozen 777 problems in grammars the model never trained on:
- **A** (trained): `ASSUME:/ASK:`, `op(a, b)`
- **B-near**: `GIVEN:/SHOW:`, `f(a, b)` — surface re-skin, same structure
- **B-far**: `LAW:/DERIVE:`, `(a ∘ b)` — parenthesized infix, different structure

Pooled correct% / unparseable%, n=777 per cell:

| arm | 8B base → FT | 14B base → FT | 32B base → FT |
|---|---|---|---|
| story A | 0.1 → **96.8** | 14.0 → **89.4** | 18.8 → **99.7** |
| literal A | 0.4 → **97.7** | 26.8 → **99.4** | 34.4 → **99.9** |
| story B-near | 0.3 → **96.3** | 15.2 → **91.9** | 13.1 → **99.4** |
| literal B-near | 0.5 → **93.7** | 27.3 → **97.7** | 31.7 → **100.0** |
| story B-far | 0.3 → 37.8 | 10.0 → 52.5 | 12.6 → 77.5 |
| literal B-far | 0.3 → 36.7 | 30.5 → 47.6 | 20.1 → 81.5 |

**Findings:**

1. **Task pairs teach translation, not a format.** Near-full transfer to the
   re-skinned grammar and to the literal input arm, neither of which was trained.
2. **The structural limit is a capacity limit.** B-far scales 37% → 50% → 80% with
   model size, and its failures are *syntactic*: of the 14B's 152 order5 B-far
   failures, **119 are unbalanced parentheses, 5 are semantically wrong**. Grammar A
   order5 is its best tier (97%).
3. **Transfer peaks early then decays.** 14B: tied across grammars at step 100
   (61%/61%), B-far peaks **74.5% at step 300**, decays to 59% by step 900 while A
   climbs 81→93%. 8B gives back ~6 points. **32B does not decay at all** (A 99% by
   step 100; B-far 77.5→84% flat). Specialization-away-from-generality shrinks with
   capacity. An endpoint-only design would have under-reported 14B transfer by ~15 pts.
4. **Not SFT narrowing.** Format-following control (60 items, 3 output formats,
   unrelated content) intact: 8B 100/90/100, 14B 100 on informative families,
   32B 100/100/100 base AND FT.
5. **Held-out theme — input-side overfitting (found 2026-08-23).** train_v2 uses 3 of
   4 themes; **`tea` never trained on**. eval_v1 has 190 tea problems of 777.
   Story arm, correct%, Wilson 95% CI:

   | model | trained themes (n=587) | tea, unseen (n=190) |
   |---|---|---|
   | 8B base | 0.2 | 0.0 |
   | **8B FT** | **99.8** [99.0, 100] | **87.4** [81.9, 91.4] |
   | 14B base | 14.1 | 13.7 |
   | **14B FT** | **100** [99.3, 100] | **56.8** [49.7, 63.7] |
   | 32B base | 19.4 | 16.8 |
   | **32B FT** | **100** [99.3, 100] | **98.9** [96.2, 99.7] |

   Base rows are the control: tea is NOT intrinsically harder (gap 0 to −2.6, CIs
   overlap). The FT gap is *created by fine-tuning*. NOTE: this is a **group
   comparison within one eval run**, not a before/after drop — do not write it with
   an arrow.

   **Mechanism (diagnosed, not speculated):** the 3 trained themes all phrase the
   operation as *verb A short-prep B* — "pours {a} into {b}", "grafts {a} onto {b}",
   "feeds {a} through {b}". Tea alone nests a noun phrase: **"pours {a} over the
   leaves of {b}"**. The 14B's tea errors are 42% wrong / only 2% unparseable, and
   **77 of 79 wrong answers have a different op-count than the reference** — it
   inserts operations, turning bare variables into `op(v, v)`:
   ```
   ref : x = op(x, op(op(op(x, x), y),        z))
   got : x = op(x, op(op(op(x, x), op(y, y)), op(z, z)))
   ```
   Reading: "the leaves of rooibos" looks like *something derived from* rooibos, and
   the only way to express that here is an operation — so it hallucinates one. The
   14B learned the surface pattern; the 32B understands the phrase.

6. **Task FT installs the internal representation.** Law-disjoint correctness probe
   on contrast_v1: 8B **0.520 → 0.949**, 14B **0.607 → 0.984**, 32B **0.599 → 0.921**.
   On a subset restricted to the 533 problems whose laws the FT model provably never
   saw: 0.942 / 0.984 / 0.904. Shuffled controls 0.48–0.53 throughout. v1's
   grammar-only FT left this untouched at 0.52. Strongest *natural* representation
   ever measured here was Qwen3.6-27B at 0.815.

**The v1/v2 contrast is a controlled pair** — same equations, models, LoRA config;
only the training distribution differs. Grammar was never the bottleneck; the task was.

### Thread C — probing & steering (DONE, shareable doc built)

1. **Correctness is encoded in a verbally silent direction.** Qwen3-32B: probe reads
   correctness in-distribution ~0.81, but that direction is orthogonal to the model's
   own yes/no axis at all 63 layers (max |cos| 0.019 vs random p95 0.027; replicated
   on Qwen3.6-27B with J-lens AND R-lens, 0/63 blocks exceed random). Vocabulary
   readout 0.114 vs 0.112 random (positive control reading literal "yes" = 0.950).
   Causally inert: injecting it moves margin −0.07 vs −0.78 for a norm-matched random
   vector, on a harness where the yes/no direction swings +24 logits and flips
   yes-rate 6% → 100%.
2. **Asking the question routes it.** Real verification question → yes/no axis becomes
   correctness-predictive at blocks 48–57 (0.689 @L53). Placebo question at the same
   token position stays at chance there (0.483; max gap +0.212 @L54).
3. **Capability and scale contribute separately.** At matched behavioral capability
   (0.626 vs 0.629), Llama-3.3-70B reads 0.634 [0.622, 0.647] vs Qwen3.5-4B 0.549
   [0.537, 0.562] — non-overlapping. Ladder with σ above permutation null:
   8B 0.5204 (1.91σ), 4B 0.549 (3.89σ), 32B 0.5985 (8.50σ), 70B 0.6337 (8.32σ),
   Qwen3.6-27B 0.8146 (18.68σ).

Seven of my own earlier claims were falsified by controls I commissioned against my
own work; that ledger is deliberately kept in `RESEARCH_LOG.md`.

---

## DATA (frozen, sha-pinned — never regenerate casually)

- **eval_v1**: 777 problems, 4 tiers (normal 180 / hard 197 / extra_hard 200 /
  order5 200), 757 distinct pair classes. Derived from SAIR
  `equational-theories-selected-problems` 4 evaluation subsets (800) minus 23
  vacuous/render drops. Fields include `story`, `literal`, `reference_rg`,
  `canonical_e`, `canonical_f`, `theme`, `pair_hash`, `tier`.
  Theme mix: graft 213, signal 198, tea 190, paint 176.
- **train_v1**: 2,772 grammar-only samples (easy 772 / medium 1,000 / hard 1,000)
  + 100 beyond-length holdout.
- **train_v2**: the SAME 2,772 pairs re-rendered as stories + 100 holdout. Zero
  render drops. Themes: signal 984, graft 896, paint 892 — **tea held out**.
  Row: `{story, completion, canonical_e, canonical_f, theme, tier, pair_hash, ...}`.
- **contrast_v1** (probe-experiments): 1,000 problems → 2,000 texts; correct RG vs
  one meaning-changing AST edit, checkform-gated, exact 50/50.

**Disjointness is enforced under the grader's symmetry group** (renaming, side swap,
dualization) via `data-gen/ftlib.py` hashing: zero train/eval pair overlap, zero
law-class overlap with eval_v1. Re-verified from scratch during QC.

Theme assignment is **deterministic** — `sha256(canonical_e => canonical_f)` mod
n_themes — so the same equation always gets the same theme.

---

## REPO LAYOUT (after the 2026-08-23 restructure)

`v2/` is gone; modules live in the stage that owns them. Every `.py` must stay
**exactly one level** under `ft-experiments/` — files resolve siblings via
`Path(__file__).parent.parent`.

```
ft-experiments/
├── config.py                     # THE registry: models, paths, eval protocol, guardrails
├── DESIGN.md                     # v2 spec: grammars, predictions, kill criteria (pre-registered)
├── RESULTS.md                    # full results, both FT experiments
├── prep.sh  watch.sh  requirements.txt
├── train-pairs-8b.ipynb          # fine-tune the 8B end to end (Modal SDK in-process)
├── data-gen/    build_train.py build_train_v2.py build_eval.py build_sair_index.py
│                verify_artifacts.py ftlib.py build-data.ipynb io/
├── training/    train_pairs.py (v2) train_lora.py (v1) config.py
├── eval/        modal_eval.py grammars.py curve_eval.py format_control.py
│                test_grammars.py base_table.py compare_table.py
├── analysis/    make_figures.py export_checkpoints.py upload_checkpoints.py
│                CHECKPOINTS.md results.ipynb floor_samples.py
├── assets/      figures + shareable HTML reports
├── eval_v1/ train_v1/ train_v2/ data/ runs/     # FROZEN artifacts
```

**Notebooks are the readable path** and all three execute cleanly:
`data-gen/build-data.ipynb` (inspect data + disjointness, laptop),
`train-pairs-8b.ipynb` (full fine-tune; uses `with train_pairs.app.run():` so the
record comes back as a live object), `analysis/results.ipynb` (tables, curves, raw
outputs).

Also: `probe-experiments/` (capture, probing, steering, lens analysis),
`RESEARCH_LOG.md` (31 entries — decisions, dead ends, falsifications),
`MORNING_REPORT.md`, `DASHBOARD.md`, `ONEPAGER.md`.

Docs to share: `probe-experiments/REPORT.html`, `ft-experiments/assets/REPORT_v2.html`
(self-contained; open in Chrome → Cmd+P → Save as PDF for Slack).

---

## INFRASTRUCTURE

- **Modal** workspace `order-evaluation` — SHARED with Idhant (his volumes are
  visible). Denis is NOT in it.
  - Volume `harsh-ft-grammar-weights` → `/models`, checkpoints at
    `checkpoints/<run>/step-N/`.
  - Volume `harsh-probe-activations` → `/acts`.
  - Guardrails: min_containers=0, max_containers=1, scaledown 60s, retries=0.
  - Eval: A10G for 8B, A100-80GB for 14B/32B. Training: A100-80GB, timeout 21600.
  - Capture GPU comes from env `CAPTURE_GPU` (default A10G) — **not** the registry.
- **HuggingFace**: authenticated as `SoHarshh`. Checkpoints published at
  **https://huggingface.co/SoHarshh/mars-v-ft-checkpoints** (public, 52 checkpoints,
  163 files, verified end-to-end by downloading and reconstructing ΔW).

### Checkpoints (52 total, all on HF)

| folder | model | trained on | n |
|---|---|---|---|
| `llama-3.1-8b_grammar-only` | Llama-3.1-8B | bare grammar text | 9 (step-0…75, every 10) |
| `llama-3.1-8b_task-pairs` | Llama-3.1-8B | story→grammar pairs | 12 (step-0…1041, every 100) |
| `ministral-3-14b_task-pairs` | Ministral-3-14B | story→grammar pairs | 12 |
| `qwen3-32b_grammar-only` | Qwen3-32B | bare grammar text | 9 |
| `qwen3-32b_task-pairs` | Qwen3-32B | story→grammar pairs | 10 (step-0…900) |

All r=16/alpha=16, dropout 0, all 7 projections, all layers; lr 2e-4 cosine, 3%
warmup, seed 0, 3 epochs. Every run has step-0. `final/` is byte-identical to
`step-1041`. `dW = (alpha/r)·B@A`. Each folder has `trajectory.npz`
(`steps`, `modules`, `norms` = ‖dW‖ per module per checkpoint).

**Useful fact for trajectory work**: weights stop moving well before training ends —
8B mean ‖dW‖ 0 → 0.263 → 0.410 → 0.460, flat from step 500; 32B 0.4855 at step 800
vs 0.4856 at 900. The interesting geometry is the first third of training. This is
also why the 32B's missing final steps (see caveat) cost nothing.

---

## HOW HARSH WORKS (respect this — it is most of the job)

- **Verify everything before reporting.** He trusts output blindly once it comes from
  me, and has said so explicitly. Never assert a number I have not re-derived from
  raw artifacts. Run controls against my own claims. He values a caught error far
  more than a confident wrong answer.
- **No fake data, no reward hacking, no silent caps.** State drops, skips, and
  limitations plainly.
- **Money is real.** He has flagged spend (~$190 in the first two days). Don't idle
  GPUs, don't re-run what's cached, prefer one-engine/hot-swap patterns over N
  container loads, size the GPU to the model. Report cost when it matters.
- **Use GPUs fully and in parallel.** He does not want serial queues; he wants every
  available slot busy — and asked specifically for full utilization, not just many
  containers. Queued Modal jobs cost nothing; launch wide.
- **Autonomy, then report.** He hands over whole experiments ("do it end to end and
  only tell me when done"). Don't ask permission for reversible in-scope work. DO
  confirm before publishing externally, spending big, or deleting anything.
- **He reads the dashboards himself** and will catch discrepancies. Keep
  `runs/ft-v2/live.log` + `STATUS.md` truthful and current.
- **Plain language.** He asks "explain like I know nothing" often — and those
  questions have twice uncovered real findings. Show examples rather than jargon.
  Avoid arrows for group comparisons (see the tea note above).
- **Slack style**: lowercase, casual, short bullets, no em-dashes, one figure
  explained in a line. He posts to Oren/Denis/Shivam and needs paste-ready text.
- **Code must look human-run, not AI-generated.** He compared us unfavourably to
  `github.com/idhantgulati/vlm-alignment` — notebooks per experiment, 5-line READMEs,
  flat configs with short present-tense comments, casual commits ("plotting", "minor
  update to ft nb"). Avoid: long defensive docstrings, dated decision citations in
  comments, orchestration shell scripts, combinatorial function generation, comments
  that argue with an imagined reviewer.
- Commits under his identity only, no Claude mentions in messages (memory:
  `git-commit-style`).

---

## PITFALLS (all hit; all fixed — do not repeat)

- `modal run` ties the remote app to the local client: a network blip kills training.
  **Always `--detach` for long jobs**, and persist run records to the volume, not
  just the return value.
- HF `upload_folder` commits **atomically** — a 5 GB folder shows nothing until it
  finishes. Push small files (npz) separately so a repo is never blank.
- Never let two writers append to the same `results.jsonl`: the positional resume
  miscounts. (Cost me a dedupe: 800 rows / 500 unique.)
- `pkill -f` patterns can kill unrelated background jobs. Be specific.
- HF pre-norms the LAST hidden state — do not re-norm capture row n−1 (this bug
  produced a 28-logit error before a validation gate caught it).
- Gemma-3 overflows float16 at capture; use fp32 (currently excluded for this).
- Ministral-3 is **multimodal** (`Mistral3ForConditionalGeneration`): use
  `AutoModelForImageTextToText`, scope LoRA to `language_model` layers, assert no
  vision leakage. No text-only variant exists.
- Python 3.13 + path-based import of a module defining dataclasses needs
  `sys.modules[name] = mod` before `exec_module`.
- Grader leniency must match across grammars or the unparseable comparison is unfair
  (B cleaner is now byte-identical to checkform's).
- 32B training needs >3h; timeout is 21600 now.

---

## KNOWN CAVEAT TO DISCLOSE

`qwen3-32b_task-pairs` stops at **step-900**, not a 3-epoch final — its container was
lost to a network failure at ~2.6 epochs. **Every 32B number in RESULTS.md is from
the step-900 adapter.** Its own curve shows grammar A at 100% from step 500 and B-far
flat from step 300, and ‖dW‖ is identical at steps 800/900, so nothing was left to
move. Documented in RESULTS.md under "Checkpoint provenance (v2)".

---

## QUALITY CONTROL PRECEDENT

A six-auditor adversarial workflow re-derived **19,425 stored verdicts** from raw
model responses — exact match on every one; summary arithmetic 0.0pp; row order clean
in 25 run dirs; training loss provably completion-only with a sha-verified prompt
byte-bridge to eval; disjointness recomputed from scratch; grammar-B instrument
fuzz-tested for leniency parity (0% label confusion); probe pipeline attacked for
leakage (none). Two minor defects found and fixed: a preemption-corrupted 14B loss
curve, and a "law-disjoint" claim scoped too broadly (fixed with FT-unseen subset
probes). **Run this kind of audit before any headline claim leaves the lab.**

New instruments must self-validate before their numbers count — e.g. `curve_eval.py`
had to reproduce the audited runner row-for-row (200/200 identical verdicts) before
its curve was admitted.

---

## OTHER PEOPLE'S WORK

- **Denis** — branch `denislim/mech-interp`, `team-update-exp01-exp03.md`
  (Qwen3-4B): law identity decodable form-invariantly mid-stack (86% vs 1.9%
  chance); steering along story→literal is a clean null; accuracy is governed by
  **thinking-token budget** (story needs ~2× literal's). His read: the bottleneck is
  serial compute, not representation style. Corroborates our
  represented-but-not-read family. He is asking for our checkpoints (now on HF).
- **Luiza** — wants checkpoints for PCA / trajectory comparison.
- **Idhant + Shivam** — `github.com/idhantgulati/vlm-alignment`, published
  (arXiv 2602.16931): narrow FT erodes VLM safety alignment, misalignment scales
  monotonically with LoRA rank, harmful behaviour occupies ~10 principal components.
  Their repo is the style reference for ours.

---

## NEXT EXPERIMENTS (designed, not run — need Harsh's go)

1. **LoRA rank sweep** (was Phase 5b, shelved when v1 failed; meaningful now that v2
   works). r ∈ {1, 2, 8, 16, 32, 64} on the 8B, everything else frozen, checkpoints
   every 100 steps. Measure against all three generalization axes (trained grammar,
   unseen grammar, unseen theme) — does higher rank buy trained-task accuracy at the
   cost of generality? Gives Luiza checkpoints across time AND capacity. ~6 A100-hours
   + eval. Mirrors the vlm-alignment rank-sweep result shape.
2. **4-fold leave-one-theme-out.** Current theme result is ONE held-out theme, so it
   cannot separate "the 14B overfits to theme vocabulary" from "tea is odd for the
   14B". Rotate all four. Data-side change only.
3. **Phrasing-vs-vocabulary probe.** A new theme with *fresh words but the trained
   sentence shape* ("verb A prep B") should barely hurt the 14B; one with a nested
   noun phrase should hurt badly. Separates unfamiliar words from unfamiliar
   structure. Cheap, renderer-side, no retraining for the diagnostic.
4. Thinking-on arm for Qwen3-32B (Denis's budget result predicts a large jump).
5. Optional: Gemma-3-27B fp32 recapture; Qwen3.6-27B steering (confounded by hybrid
   linear attention).
