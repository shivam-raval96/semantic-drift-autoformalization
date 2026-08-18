# Overnight dashboard — 2026-08-18/19

Status at a glance. `MORNING_REPORT.md` = the writeup to read first.
`RESEARCH_LOG.md` = full reasoning, every hypothesis and failure.

---

## THE RESULT OF THE NIGHT

**Qwen3-32B encodes translation correctness in a direction it has no way to
say — and we have the positive controls to prove the measurement works.**

| Evidence | Number | Control it beats |
|---|---|---|
| The information is really there | probe 0.599 law-disjoint, 8.5σ above null | lexical floor 0.503 |
| NOT on the yes/no verbalization axis, at ANY layer | cos 0.011; **0 of 63 blocks** exceed random | random \|cos\| p95 0.027 |
| Replicated: 2nd model, 2 independent lenses | **0/63** for J-lens AND R-lens | same, on a model with a *stronger* representation (0.82) |
| NO verbal expression at all | readout max **0.1143** | random 0.1121; **positive control 0.9503** |
| Steering it does nothing | Δmargin **−0.07** | random −0.78 |
| ...and the harness demonstrably works | J-lens dir **+24.16**, yes-rate 6%→**100%** | same code, same texts |

Figure: `probe-experiments/steering_mechanism.png`

## SECOND RESULT: co-emergence, with the specificity control

Pre-registered `mean` site, paired clustered bootstrap, permutation null under
the law-disjoint grouping (`runs/analysis-v1/probe_uncertainty.json`):

| Model | capability | representation | 95% CI | σ above null |
|---|---|---|---|---|
| Llama-3.1-8B | 0.522 | 0.5204 | [0.511, 0.530] | 1.91 (borderline) |
| Qwen3.5-4B | 0.626 | 0.5490 | [0.537, 0.562] | 3.89 |
| Qwen3-32B | 0.669 | 0.5985 | [0.586, 0.611] | 8.50 |
| **Qwen3.6-27B** | **0.826** | **0.8146** | [0.801, 0.829] | **18.68** |

All adjacent pairs separate at P(>0) = 1.0. **Specificity control:** a
task-irrelevant label (story theme) does NOT track capability — at matched
capacity it *anti*-correlates (Spearman −0.4). So this is not "bigger models
represent everything better."

---

## HONESTY LEDGER — four of my own claims died tonight

Each was killed by a control or audit I commissioned to attack it.

1. **"The model knows much more than it says"** — leaky CV. Under law-disjoint
   splits: probe 0.6708 vs model's readout 0.6701. It expresses what it knows.
2. **"Peaks at L53 then declines"** — bug artifact; Δ=0.0156, CI straddles 0.
3. **"Task framing routes correctness into the verbal channel"** — **a placebo
   question ("Is the story written in English?") recovers 0.6366 of the
   0.6701 effect.** It is the chat format and answer position that open the
   channel, not the question's content.
4. **"Pearson r=0.95 co-emergence" / "absent in the 8B"** — the relation is
   ordinal not linear (slope through the low models under-predicts the top by
   0.16); the fold SDs we reported were invalid (the law-disjoint folds are
   1112/666/74/74/74 — effectively a tier split); the 8B is borderline
   (1.9σ), not absent.

## RUNNING (self-completing)

| Lane | Job | Decides |
|---|---|---|
| H100×2 | Llama-3.3-70B capture+probe | **decisive**: same capability as Qwen3.5-4B (0.629 vs 0.626) at 17× the size. Co-emergence predicts ~0.55; scale-quality predicts 0.60+ |
| A100 | Gemma-3-27B probe | fills the empty middle band; n=4→6, P(perfect order) 0.042→0.0014 |
| local | position control 2×2 | the last confound on the reader-vs-asked contrast |

## COMPLETED

margin lens (reader + asked + placebo) · lens validation gate · J-space
alignment · vocabulary readout with positive controls · steering positive
control · two-directions comparison · knows-vs-says under law-disjoint splits ·
7 behavioral gates · 5 model captures · replication on Qwen3.6-27B ·
co-emergence + specificity + uncertainty analyses · literature recon

## FAILED / FIXED (diagnosed, never blindly rerun)

bf16 via numpy → torch · gate-sample leak → full 2000 · missing git in image →
apt · nested multimodal weights → key autodetect · Neuronpedia lens is a
fit-checkpoint → reconstruct mean exactly · **double-normalized final layer →
caught by validation gate** · volume read-after-write races → settle+verify+retry

## NEXT (for tomorrow)

1. Read the 70B result — it settles scale vs capability.
2. Is the answer-position signal the same direction under placebo and real
   question? (cheap; both captures exist)
3. FT v2 with the task in-distribution, measured on all three axes: behavior,
   representation, verbal accessibility.
