# Overnight dashboard — 2026-08-18/19

Live status. Full reasoning in `RESEARCH_LOG.md`. Every number traces to a
committed file under `probe-experiments/runs/`.

---

## THE RESULT OF THE NIGHT

**Qwen3-32B encodes translation correctness in a direction that is verbally
silent, and we now have the full mechanism with positive controls.**

| Evidence | Number | Control it beats |
|---|---|---|
| The information is really there | probe 0.599 law-disjoint | lexical floor 0.503 |
| It is NOT on the yes/no verbalization axis (at ANY layer) | cosine **0.011**; 0 of 63 blocks exceed random | random \|cos\| p95 0.027 |
| It has NO verbal expression at all | readout max **0.1143** | random 0.1121; positive control **0.9503** |
| Steering it does nothing | Δmargin **−0.07** | random −0.78 |
| ...and the harness demonstrably works | J-lens dir Δmargin **+24.16**, yes-rate 6%→**100%** | same code, same texts |

So the earlier "readable but causally inert" null is now *explained*: we were
pushing a direction with essentially zero component on the axis that controls
what the model says. The knowledge is real, decodable, and mute.

**Bonus nuance:** steering the J-lens direction shifts the model's *bias*
(answers yes to everything) but barely dents its *discrimination*
(AUROC 0.669 → 0.647). The verbal axis controls what it says, not what it knows.

---

## HONESTY LEDGER (claims I withdrew tonight)

An independent adversarial agent reviewed the headline claim and found real
problems. Recorded because these corrections matter more than the wins:

1. **"The model knows much more than it says" — WITHDRAWN at the answer
   position.** My probe used problem-grouped CV (leaky across shared laws)
   against a fit-free lens. Under law-disjoint splits: probe 0.6708 vs model's
   own readout 0.6701. Essentially identical. The honest version is the
   opposite and more interesting: *when asked, the model's verbal readout
   captures essentially everything a linear probe can extract.*
2. **"Peaks at L53 then declines to the output" — DEAD.** The endpoint was a
   bug artifact; corrected Δ = 0.0156 with bootstrap CI [−0.004, +0.036].
3. **"Task framing performs late-depth routing" — CONFOUNDED, controls
   running.** Reader vs asked differ in capture *position* and in whether
   yes/no is even a licensed continuation. Two controls in flight (below).
4. **A real bug caught by validation:** HF pre-norms the last hidden state; our
   lens normalized it twice. Fixed; after the fix the final-layer lens (0.6701)
   matches the independently measured behavioral gate (0.6694) to 0.0007.

---

## RUNNING

| Lane | Job | Decides |
|---|---|---|
| A100 | asked capture with `ansend` site | the position confound (C1) — is the effect the question or the readout position? |
| A100 | placebo-question capture ("Is the story written in English?") | the licensing confound (C2) — question-specific or format-driven? |
| local | Qwen3.6-27B reader probes | co-emergence point at the top capability (0.826) |
| watchdog | 5-min heartbeat | early failure detection |

## COMPLETED TONIGHT

- margin lens reader + asked (Qwen3-32B), bug found → fixed → revalidated
- lens-formula validation harness (now a permanent gate on all lens work)
- J-space alignment: probe direction vs verbalization direction
- vocabulary readout with two positive controls
- steering positive control (J-lens direction) — harness proven to work
- two-directions comparison (reader vs asked representations)
- knows-vs-says recomputed under law-disjoint splits
- behavioral margin gates: qwen3.5-4b 0.626, gemma-3-27b 0.602, qwen3.6-27b 0.826
- literature recon (J-lens/J-space, R-lens, logit/tuned lens) + verified lens
  inventory; discovered Qwen3-32B has a free pre-fitted J-lens

## CAPABILITY LADDER (behavioral margin AUROC, 300 balanced texts)

| Model | Margin AUROC |
|---|---|
| Llama-3.1-8B | 0.522 |
| Qwen2.5-7B | 0.564 |
| gemma-3-27b-it | 0.602 |
| Qwen3.5-4B | 0.626 |
| Llama-3.3-70B | 0.629 |
| Qwen3-32B | 0.669 |
| **Qwen3.6-27B** | **0.826** |

## FAILED / FIXED (all diagnosed, none blindly rerun)

- numpy can't read bf16 → read via torch
- asked capture inherited the 300-text gate sample → full 2000 by default
- `debian_slim` has no git for pip-from-github → apt git
- 3.6-27B weights nested under `model.language_model.*` → key autodetection
- Neuronpedia lens is a raw `fit()` checkpoint, not a saved lens → reconstruct
  the mean exactly as `jacobian_sum / n_done`
- **margin lens double-normalized the final layer** → caught by the validation
  gate, diagnosed with a discriminating test, all numbers recomputed
- volume read-after-write races on large downloads → settle + verify + retry

## OPEN QUESTIONS

1. Do the two in-flight controls survive? If asked@answer-end ≈ 0.50 and the
   placebo margin ≈ 0.50, the routing story is clean; if the placebo tracks
   correctness, the effect is format, not verification.
2. Are there genuinely two correctness representations? (cos 0.053, asymmetric
   cross-application — needs the full 2000-text asked set to confirm.)
3. Does the verbal-silence result replicate on Qwen3.6-27B (0.826 verifier,
   free J+R lenses)?

## NEXT

1. Fold the two controls into the claim as soon as they land.
2. Replicate the orthogonality + vocabulary-silence result on Qwen3.6-27B.
3. Co-emergence scatter: capability vs law-disjoint probe strength across the
   roster.
