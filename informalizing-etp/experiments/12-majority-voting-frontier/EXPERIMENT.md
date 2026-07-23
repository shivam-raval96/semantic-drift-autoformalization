# 12 — Majority voting for frontier closed models on the deepest pairs

Run 2026-07-23.

## Question

Experiment 11 found that K-sample self-consistency barely helps weak
open-weight models (+2.3 points pooled at vote@7), because their
errors are not sampling noise: in the deepest complexity bins they sat
at ~0% and seven samples found nothing to amplify. This experiment
points the same machinery at the opposite corner: **frontier closed
models on only the hardest pairs** (per-equation ops bin 10, the bin
where every open model scored 0), with the vote ladder extended to
**k = 1..10** and the sampling run at **two temperatures (0.7 and
1.0)** to test whether more sampling diversity changes the voting
gain. Are frontier models above the bin-10 floor at all — and where
they are, is the residual error the kind of noise majority voting can
remove?

## Setup

Same pipeline, prompts, grading, and regime machinery as experiments
09/11; the changed variables are the models, the pair set (bin 10
only), the two temperatures, and K = 10.

- **Equations**: experiment 09's `data/synthetic-equations.txt`
  (sha256 `7f11f36c…`), as in experiment 11.
- **Models** (3, closed, via OpenRouter):
  - `anthropic/claude-opus-4.8`
  - `openai/gpt-5.5`
  - `x-ai/grok-4.3`
  Chosen so the off regime is *fully* clean — recorded
  `native_reasoning`: opus `{"enabled": false}`, gpt-5.5 and grok
  `{"effort": "none"}`. Zero reasoning tokens in all 1,260 calls; no
  per-vendor floors (gemini-3.1-pro / gemini-2.5-pro / grok-4.5 were
  rejected for mandatory reasoning).
- **Temperature caveat (gpt-5.5)**: OpenRouter's metadata lists no
  `temperature` support for gpt-5.5, so the parameter is dropped and
  its passes vary only by the model's internal default sampling. Its
  "t0 baseline" is therefore not greedy decoding, and its t0.7 and
  t1.0 arms are draws from the same distribution — a built-in A/A
  control (observed: 5.5 vs 5.4 mean correct-of-20 per pass). The
  temperature comparison rests on opus and grok.
- **Sampling** (seed 0): `--stratify-eq-ops 20 --eq-bins 10:10`,
  `--label-prefix S` → 20 pairs, both laws carrying exactly 10
  operations. Note this is a *fresh* deterministic draw from the
  bin-10 pool, not experiment 11's bin-10 subset: the balanced
  sampler consumes one RNG stream across bins, so dropping bins 1–9
  changes which bin-10 pairs seed 0 yields. Comparisons with
  experiments 09/11 are at bin level only.
- **Form / regime**: `--form story`, `--reasoning off`, max_tokens
  4096, `--concurrency 8`.
- **Passes**: 10 at temperature 0.7
  (`runs/run-eqstrat20-bin10-s0-story-t0.7-pass{1..10}`) and 10 at
  temperature 1.0 (`…-t1-pass{1..10}`), each a normal resumable
  benchmark run; `samples.jsonl` byte-identical across all 21 run
  dirs. Temperature verified live where supported: opus emits 197–198
  distinct (pair, response) cells of 200 per arm, grok 195–200;
  gpt-5.5 only 95–100, consistent with the dropped parameter.
- **Voting** (`voteform.py`): per arm, vote@k over prefix passes 1..k
  for every k = 1..10 (even k allowed; ties fall back to the earliest
  pass). New `--condition-prefix` flag stamps arm-specific condition
  labels ("t0.7 vote@3" vs "t1.0 vote@3") so both ladders share one
  charts.py report.
- **Baselines**: a fresh temperature-0 run of the same command
  (`runs/…-t0-baseline`) — experiment 09's baseline lacks these
  models — plus its voteform-labeled subset copy
  (`runs/…-t0.7-t0-baseline`, "temp0 single-pass"), which is what the
  report consumes.
- **Scale**: 20 pairs × 3 models × 21 runs = 1,260 calls,
  OpenRouter-reported cost $17.61 (2.54M tokens; opus median response
  ~1.6k chars vs ~300 for gpt-5.5/grok).

## Prompts

Story arm template byte-identical to experiments 02–11:
`formalize_prompt.md` at sha256
`ad33f6de859156b81be0d889abd3c56e4d9275bd855eb6d804d4e8ebcfe4983c`.
Nothing in any prompt mentions repeated attempts or voting.

## Reproduce

```sh
M3=anthropic/claude-opus-4.8,openai/gpt-5.5,x-ai/grok-4.3
EQ=experiments/09-synthetic-complexity/data/synthetic-equations.txt
OUT=experiments/12-majority-voting-frontier/runs

python3 benchmark.py --seed 0 --stratify-eq-ops 20 --eq-bins 10:10 \
    --equations-path $EQ --label-prefix S --form story --reasoning off \
    --concurrency 8 --models "$M3" \
    --out-dir $OUT/run-eqstrat20-bin10-s0-story-t0-baseline

for T in 0.7 1.0; do
  TT=$([ $T = 0.7 ] && echo t0.7 || echo t1)
  for P in 1 2 3 4 5 6 7 8 9 10; do
    python3 benchmark.py --seed 0 --stratify-eq-ops 20 --eq-bins 10:10 \
        --equations-path $EQ --label-prefix S --form story --reasoning off \
        --temperature $T --concurrency 8 --models "$M3" \
        --out-dir $OUT/run-eqstrat20-bin10-s0-story-$TT-pass$P
  done
done

python3 voteform.py $OUT/run-eqstrat20-bin10-s0-story-t0.7-pass{1,2,3,4,5,6,7,8,9,10} \
    --ks 1,2,3,4,5,6,7,8,9,10 --out-root $OUT --condition-prefix "t0.7 " \
    --baseline $OUT/run-eqstrat20-bin10-s0-story-t0-baseline \
    --baseline-label "temp0 single-pass"
python3 voteform.py $OUT/run-eqstrat20-bin10-s0-story-t1-pass{1,2,3,4,5,6,7,8,9,10} \
    --ks 1,2,3,4,5,6,7,8,9,10 --out-root $OUT --condition-prefix "t1.0 "

python3 charts.py \
    $OUT/run-eqstrat20-bin10-s0-story-t0.7-t0-baseline \
    $OUT/run-eqstrat20-bin10-s0-story-t0.7-vote{1,3,5,10} \
    $OUT/run-eqstrat20-bin10-s0-story-t1-vote{1,3,5,10} \
    --title "ETP story formalization · majority voting, frontier closed models, deepest-nesting pairs (bin 10), t0.7 vs t1.0" \
    --out experiments/12-majority-voting-frontier/report/report.html --pdf
```

## Results

Artifacts: the 21 benchmark run dirs, 21 voteform-derived dirs
(vote@1..10 per arm plus the labeled baseline subset) under `runs/`,
and `report/report.html` / `.pdf` (correct rate by model and
condition, accuracy by nesting depth 6–8 within the bin, verdict
composition per condition). All 21 real runs: 60/60 rows graded, 0
api-errors, compliance fully clean (0 reasoning tokens everywhere).

Correct% over the 20 bin-10 pairs (t0 = the temp-0 baseline):

| model | t0 | t0.7 vote@ 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 / 10 | t1.0 vote@ 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 / 10 |
|---|---|---|---|
| claude-opus-4.8 | 80 | 80 / 80 / 95 / 100 / 100 / 100 / 100 / 100 / 100 / 100 | 80 / 85 / 95 / 100 / 100 / 100 / 100 / 100 / 100 / 100 |
| gpt-5.5 | 25 | 25 / 25 / 25 / 30 / 30 / 35 / 35 / 35 / 35 / 35 | 30 / 30 / 30 / 30 / 30 / 30 / 30 / 30 / 30 / 30 |
| grok-4.3 | 0 | 0 everywhere | 0 everywhere |
| **pooled** | 35.0 | 35.0 → 45.0 (from k=6) | 36.7 → 43.3 (from k=4) |

- **Flips, t0 baseline → vote@10**: t0.7 arm +6 / −0 (opus +4,
  gpt-5.5 +2); t1.0 arm +5 / −0 (opus +4, gpt-5.5 +1). No cell
  anywhere flipped correct → wrong: on this pair set voting is
  strictly non-destructive, unlike experiment 11 (22 gained / 9 lost).
- **Per-pass single-sample rates**: opus 80–95 (t0.7), 75–95 (t1.0);
  gpt-5.5 25–35 / 20–30; grok 0 in all 21 runs.
- **Vote diagnostics at k = 10** (votes-for-winner → correct%):
  - t0.7: 0 votes (all 10 passes unparseable) 6 cells;
    1 vote 21 cells → 0%; 2–3 votes 8 cells → 25%;
    6–10 votes 25 cells → **100%** (unanimous 10/10: 12 cells).
  - t1.0: 0 votes 8 cells; 1 vote 23 cells → 0%; 3–6 votes 6 cells →
    33%; 7–10 votes 23 cells → **100%** (unanimous: 10 cells).
  - As a confidence gate: answer only at ≥6/10 consensus → 100%
    precision on 42% coverage (t0.7); ≥7/10 → 100% on 38% (t1.0);
    overall correct rate is 43–45%. The bimodal shape of experiment
    11 reproduces exactly — cells are either high-consensus (all
    correct) or every-answer-different (all wrong) — but for a
    frontier model the high-consensus mode is much bigger.
- **Voting again fixes parse failures cheaply**: pooled unparseable
  drops 22 → 6 (t0.7 vote@1 → vote@10) and 25 → 8 (t1.0); for opus it
  reaches 0. gpt-5.5's persistent unparseables are arity-malformed
  `op(…)` terms emitted at finish_reason `stop`; grok's include
  repetition-loop collapses — `ASSUME: op(a, op(b, op(a, …` repeated
  until the 4096-token cap (15 of 420 pass rows) — the same
  output-grammar collapse experiment 09 documented.
- **Temperature made no material difference** where it was real:
  opus reaches 100% by vote@4 in both arms; grok stays at 0 in both.
  The gpt-5.5 A/A arms differ by ≤1 cell at every k, sizing the
  sampling noise floor.

## Conclusions

- **Near its capability edge, a frontier model's residual errors are
  sampling noise — and majority voting removes them completely.**
  Opus 4.8 goes 80% → 100% on the hardest bin by vote@4 in both
  arms, with zero correct→wrong flips; every one of the 20 deepest
  pairs has a correct modal answer. This is the regime experiment
  11's weak models never reached: for them deep pairs failed by
  grammar collapse (nothing to amplify), while for a strong model
  the same pairs fail by occasional slips that other samples
  outvote.
- **The capability cliff just moves, it doesn't disappear.**
  Grok 4.3 is 0/20 in all 21 runs — a frontier-branded model can sit
  exactly where the weak open models sat, complete with the same
  repetition-loop failure mode — and gpt-5.5's +10/+5 points come
  from just 2–3 flipped cells, its errors mostly stable malformed
  grammar rather than noise. Voting only pays where a model is
  already mostly right; it is a polish for near-mastery, not a
  ladder out of incapacity.
- **Sampling temperature (0.7 vs 1.0) doesn't matter here.** Both
  arms saturate at the same ceiling at nearly the same k. The choice
  of 0.7 in experiment 11 was not load-bearing.
- **Consensus stays a near-perfect, free confidence signal at the
  frontier**: ≥6/10 agreement identified a subset with 100%
  precision at 38–42% coverage against a 43–45% base rate — the
  strongest version yet of experiment 11's calibration finding, and
  the practical recipe this suggests: sample K times, answer on
  consensus, escalate (to a stronger model or a human) on
  disagreement.
- Methodological: `voteform.py --condition-prefix` (new) lets several
  vote ladders share one charts.py report with unambiguous condition
  labels; the report consumes the voteform-labeled baseline subset so
  the temp-0 arm is labeled "temp0 single-pass" rather than the
  regime fallback "no-think".
