# Grammar-only fine-tuning: does rigid-grammar fluency training improve translation?

## Experiment

Fine-tune ONLY on raw rigid-grammar text (bare `ASSUME:/ASK:` lines, no stories,
no instructions, plain next-token loss) and test whether story-to-RG /
literal-to-RG translation improves on unseen equations. The grammar is identical
across all equations; only the specific laws differ, so grammar fluency is what
is allowed to transfer. Run on two models: Llama-3.1-8B-Instruct (near 0%
correct at base) and Qwen3-32B (13-61% correct at base), so effects are readable
in both directions. All grading is mechanical (checkform); kill criterion stated
before any run.

## Data

```
train_v1 (grammar text only, frozen, sha-pinned)
  easy 772 (ops 2-4, ETP laws) | medium 1,000 (5-8, ETP) | hard 1,000 (10-12, synthetic)
  + 100 beyond-length holdout (ops 14-16, never trained)
        v
  pair-disjoint from ALL 2,669 SAIR rows, law-disjoint from every evaluated
  problem, both under the grader's symmetry group (renaming, side swap,
  dualization); every sample round-trip verified through the grader's parser
        v
eval_v1 (frozen): 777 problems, 4 tiers (normal 180 / hard 197 / extra_hard 200
/ order5 200), rendered to story + literal + reference RG
```

## Setup

| | |
|---|---|
| Models trained | Llama-3.1-8B-Instruct (41.9M trainable), Qwen3-32B (134M trainable) |
| LoRA | r=16, alpha=16, all layers, q/k/v/o/gate/up/down projections, dropout 0 |
| Objective | causal LM on raw `text` field, packed 1024-token blocks, loss on all tokens, no chat template |
| Schedule | LR 2e-4 cosine, 3% warmup, micro-batch 1 x grad-accum 4, 3 epochs = 75 steps, seed 0 |
| Eval protocol (frozen) | vLLM bf16 greedy, no-think, fixed templates (sha-pinned), max_tokens 4096; chunked resumable runner |
| Checkpoints | every 10 steps + step 0, on the Modal volume |

## Training-side results (both models learn the grammar)

| Model | Loss (start -> end) | Holdout RG perplexity | Raw RG completion after FT |
|---|---|---|---|
| 8B | 0.71 -> 0.47 | 9.72 -> 2.49 | fluent, parseable RG |
| 32B | 0.69 -> 0.45 | 12.12 -> 2.41 | fluent, parseable RG |

## Behavioral results (translation on eval_v1, base vs FT)

![Grammar-FT effect](ft_effect.png)

| Model / arm | Correct, base -> FT | Unparseable, base -> FT |
|---|---|---|
| 8B story | 0.1% -> 0.0% | 12.2% -> 32.3% (runaway) |
| 8B literal | 0.4% -> 0.0% | 22.8% -> 0.0% |
| 8B two-stage | 0.0% -> 0.0% | 19.2% -> 0.1% |
| 32B story | 18.8% -> 5.3% | 6.7% -> 43.2% (runaway) |
| 32B literal | **34.4% -> 3.6%** | 5.7% -> 0.1% |
| 32B two-stage | **34.9% -> 3.3%** | 6.3% -> 0.5% |

Pooled over all 777 problems; per-tier tables in `runs/ft-v1/comparison.md`.

Three findings:

1. **Syntax is fully trainable.** On literal and two-stage prompts the FT models
   emit perfectly parseable grammar (unparseable to ~0% from up to 45% per tier).
2. **Story prompts trigger a non-stopping pathology.** The FT models stream RG
   until the token cap (nearly all unparseables are length-capped truncations);
   generation volume roughly triples, which is also why FT evals needed
   escalated timeouts and chunked runs.
3. **No semantic gain where capability is absent; displacement where it exists.**
   The 8B stays at 0% correct. The 32B, which translated at 34% pooled, drops to
   3-4%: it learned "always produce fluent RG" and that habit overrode
   "translate the input".

## Representation check

The FT-8B checkpoint was captured and probed identically to the base model
(probe-experiments, contrast_v1): law-disjoint AUROC 0.52-0.53 before AND after
FT, at the lexical floor. Grammar training changed how the model writes, not
what it represents about correctness. (Full probing/steering program:
`probe-experiments/RESULTS.md`.)

## Kill verdict

The grammar-bottleneck hypothesis is rejected in the strongest available form:
grammar-only continuation training is a **behavioral override, not a skill
injection**. It buys syntax, adds no semantics behaviorally or internally, and
displaces existing translation ability in a model that had it.

## Learnings for a v2 recipe

1. Put the TASK in the training distribution: instruction-formatted story->RG
   pairs (or completion-style with the story in context), never bare notation.
2. Disjointness must be computed under the grader's symmetry group or you leak
   answer keys (`data-gen/ftlib.py` has the canonical hashing).
3. Verify the format bridge before interpreting chat evals: perplexity + raw
   completion probes (both passed here, which is what makes the null clean).
4. Expect FT models to generate longer; budget eval timeouts accordingly (or
   use the chunked resumable runner).

## Artifacts

`train_v1/` + `eval_v1/` (frozen, manifests), `training/` (configs, presets),
`runs/ft-v1/` (train records, per-arm evals, `comparison.md`),
`runs/base-v1/` (base evals incl. Qwen3-32B). Repro:
`data-gen/` -> `training/run_train.sh` -> `eval/run_eval.sh <model> <arm>
[--adapter ...]` -> `eval/compare_table.py`.
