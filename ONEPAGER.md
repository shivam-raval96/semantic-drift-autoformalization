# Status one-pager: grammar FT + correctness probing (Harsh, 2026-08-16)

Two experiment threads on the ETP substrate, both on branch `harsh/experiments`.
All grading is mechanical (checkform); all artifacts frozen with sha manifests;
all GPU runs on Modal.

## Thread A: grammar-only fine-tuning (ft-experiments/)

Question: does fine-tuning ONLY on rigid-grammar text (no stories, no
instructions) improve story-to-RG / literal-to-RG translation on unseen equations?

Done:
- eval_v1 frozen: 777 problems (4 SAIR tiers: normal 180 / hard 197 /
  extra_hard 200 / order5 200; 757 distinct pair classes), rendered to
  story + literal + reference RG.
- train_v1 frozen: 2,772 grammar-only samples (easy 772 / medium 1,000 /
  hard 1,000) + 100 beyond-length holdout; pair-disjoint from all 2,669 SAIR
  rows, law-disjoint from eval_v1.
- Base evals (vLLM, greedy, no-think, A10G): Llama-3.2-1B 0/777 correct in all
  three arms; Llama-3.1-8B 1/777 (story), 3/777 (literal), 0/777 (two-stage).
  Signal lives in the unparseable rate (8B: story 6.0-23.5%, literal 13.3-45.0% by tier).
- Phase 5a training complete: LoRA r=16, all layers, 41.9M trainable params,
  75 steps; holdout perplexity 9.72 -> 2.49; format probes passed.

FT evals complete (runs/ft-v1/comparison.md). 8B: unparseable eliminated in
literal (45->0%) and two-stage (39->0%), inverted in story (9-24% -> 18-52%,
runaway generation); correct 0% everywhere. 32B (same recipe, fresh adapter):
grammar perfected, but correct COLLAPSED - literal 32-61% -> 3-6%, two-stage
29-62% -> 2-6%, story 13-39% -> 3-10% with runaway returning (32-60% unparse).
Kill verdict: grammar-only continuation training is a behavioral override, not
a skill injection - it adds nothing where semantics is absent (8B) and
displaces translation where semantics exists (32B). Still open: Phase 5b rank
sweep (moot under the kill verdict unless reframed).

## Thread B: is translation correctness linearly represented? (probe-experiments/)

Setup: contrast_v1 frozen: 1,000 problems (334 easy / 333 medium / 333 hard) ->
2,000 texts; correct answer = reference RG, wrong = one checkform-gated AST edit
(arg_swap 226 / var_sub 274 / prune 247 / grow 253; 193 grader-symmetric edits
rejected). Probes: logistic regression (+ MLP twin) per layer, out-of-fold,
grouped splits; controls: law-disjoint split, char-TF-IDF lexical floor, answer
length, layer 0, shuffled labels.

Results:
1. Llama-3.1-8B (reader mode, 33 layers x 2 sites): in-distribution 0.623 peak,
   law-disjoint 0.520, within 0.02 of the like-for-like lexical floor (0.503).
   Pre-registered H1 refuted. MLP twin also collapses (0.504-0.512 law-disjoint).
2. Behavioral verification gate, 300 balanced texts, one-word answers:
   8B 0.500 / Qwen2.5-7B 0.520 / Qwen3-32B 0.537 / Llama-3.3-70B 0.540 - all
   fail ~0.65; conservative pattern (8B emits zero "yes"; the others' rare "yes"
   answers are 0.79-0.88 precise). Few-shot arm: 32B 0.590 (easy tier 0.67);
   7B and 70B moved slightly down; 8B not run.
3. Logit-margin gate (threshold-free): 8B 0.522 / 7B 0.564 / 32B 0.669 /
   70B 0.629. Qwen3-32B is the smallest passer and beats the 70B.
4. Qwen3-32B probes (65 layers x 2 sites): in-distribution 0.705 (layer 61),
   law-disjoint 0.599 (all folds 0.584-0.641) vs lexical floor 0.503 under the
   same split; easy tier 0.793. MLP shows no consistent gain (below linear at the
   reported site; 0.600 vs 0.570 at the other).

5. Direction analysis (local): probes fit on independent law-disjoint folds
   share a dominant direction (mean pairwise cosine 0.70 in 5,120-d; range
   0.26-0.94, the low pair excluding the largest law cluster). Probe score vs
   the model's own yes/no margin: Spearman 0.30 (n=300) - the representation
   and the behavioral readout are partially decoupled.

Headline: a law-general, linearly decodable correctness signal exists in the
capable model's deep layers and is absent in the incapable 8B; representation
and behavioral competence emerge together, tier for tier.

Round 4 (complete): (a) FT-8B probes - representation unchanged by grammar FT
(law-disjoint 0.52-0.53 = base). (b) Steering, 300 texts x 8 conditions:
baseline margin AUROC 0.6694 reproduced exactly; direction injection at
0.25-1.0x residual norm (both signs) moves AUROC < 0.006; norm-matched random
shifts margins more than the direction does. The correctness direction is
readable but causally inert - the lab's fourth independent
represented-but-not-read result. (c) 32B base translation: story 13-39%,
literal 32-61%, two-stage 29-62%, order5 <=1.5% (depth cliff); the first
signal-zone model on eval_v1.

## Setup details (at hand for the meeting)

| | |
|---|---|
| Models evaluated | Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct (base evals); + Qwen2.5-7B-Instruct, Qwen3-32B, Llama-3.3-70B-Instruct (verification gates) |
| Models trained | Llama-3.1-8B-Instruct (LoRA r=16 all layers; rank sweep pending) |
| Models probed | Llama-3.1-8B (33 layers, d=4096), Qwen3-32B (65 layers, d=5120); residual stream, float16, 2 sites (last answer token / mean over answer tokens), 2,000 texts each |
| Tasks | translation (story->RG, literal->RG, two-stage; checkform-graded), verification (yes/no on candidate; word-level and logit-margin), probing (correct vs wrong reading) |
| ETP data | source list 4,694 laws; eval_v1 777 SAIR problems; train_v1 2,772+100 (ETP pool + 240 genform synthetic laws, 5-8 ops/eq); contrast_v1 1,000 pairs (ETP + 240 fresh synthetics, ops_total 2-12; 92 law-connected components) |
| Representations | themed stories (4 themes: paint, tea, graft, signal), literal NL, rigid grammar (ASSUME/ASK prefix op(a,b)); probing uses story + RG answer as bare text |
| Protocol constants | greedy/no-think everywhere, seeds fixed, frozen prompt templates (sha-pinned), vacuous laws excluded, R1 (mechanical ground truth only), R5 (no training against the checker), R7 (translation and implication never mixed) |
| Infra | Modal: A10G (1B/7B/8B), A100-80GB (32B), H100x2 (70B); vLLM for evals, HF transformers for capture/gates; run records committed per run |

Full numbers and provenance: `probe-experiments/RESULTS.md`, `runs/`,
`ft-experiments/runs/`.
