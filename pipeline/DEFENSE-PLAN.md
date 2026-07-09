# DEFENSE-PLAN.md — building the confirmatory infrastructure

Status: PROPOSED (nothing here is built; this file is the plan for the defending
layer). Everything measured so far is exploratory; this plan converts the five
headline results into preregistered, killable claims with enforced variation
tracking and per-instrument scope cards. Principle throughout: an experiment that
cannot fail is not evidence; an instrument that won't say "outside my bounds" is
not an instrument.

---

## 0. The research question (primary, falsifiable)

> **RQ: When an instruction-tuned LM translates a natural-language statement of an
> equational law into formal form, is upcoming certified semantic drift linearly
> decodable from the model's residual stream — before any output token — beyond
> what the input surface alone predicts?**

Operationalized as H1 below with a frozen probe site, frozen layer, frozen split,
and a kill condition. Everything else (migration, gradient, judge-swap, telephone)
is secondary: each gets its own kill condition and Holm correction across the set.
If H1 dies, the paper's monitor claim dies with it — that is the point.

### Hypotheses and kill conditions

| id | claim (confirmatory form) | kill condition |
|---|---|---|
| **H1** (primary) | On a FRESH never-used ETP sample (law-hash split), a logistic probe at the frozen site/layer predicts certified drift with ΔAUROC(probe − bag-of-words) > 0, 95% cluster-bootstrap CI excluding 0. Preregistered secondary bar: same vs max over the full gauntlet baseline set (BoW, register one-hot, surface stats, other-model encoders) | primary CI includes 0 |
| H2 | Silent-failure share (VBU / all failures) increases with capability rank across a preregistered 6-model ladder; index = METR 50%-task-completion time horizon (snapshot at prereg date), Arena-Elo fallback slots for uncovered models, mixing declared in prereg | Spearman ρ ≤ 0, or CI includes 0 |
| H3 | Mean-pooled intent decodability decreases monotonically A1→A2→A3 (held-out-rung protocol), per-equation permutation test; A2 admitted only if reworked renderer passes L-fail < 20% pilot gate, else drops out automatically | non-monotone ordering in the confirmatory sample |
| H4 | Certificate-trained monitor beats judge-trained monitor on the judge-error stratum: ΔAUROC > 0, CI excluding 0, both preregistered judges | CI includes 0 for both judges |
| H5a | Sampled telephone chains (temp 0.7, 5 seeds/origin): survival plateaus (survival₆ − survival₃ ≥ −10pts) | survival keeps falling |
| H5b | Terminal-state (attractor) distributions differ across models (permutation test on attractor histograms) | distributions indistinguishable |

Effect-size predictions (from exploratory data, recorded so shrinkage is visible):
H1 probe ≈ 0.96 vs BoW ≈ 0.75; H2 silent share haiku ~0.28 → sonnet ~1.0;
H3 0.98/0.78/0.32; H4 Δ ≈ 0.87 on the error stratum; H5 plateau by hop 3.
Confirmatory numbers may shrink; the kill conditions are about sign and CI, not
about reproducing the exploratory magnitudes.

### What the experiments do NOT test (registered non-claims)

- Nothing beyond **single-binary-operation equational laws**. No claim about
  Lean/Isabelle, code, or natural-language entailment. (Second domain = separate
  future prereg, not this one.)
- Nothing beyond the **named models**. The monitor is per-model; H1 passing on
  Llama-3.1-8B licenses no claim about any other checkpoint. (Cross-model
  transfer is exploratory arm X2, reported as such.)
- `equivalent*` is **not proven equivalence** — it is "no countermodel ≤ 4."
  Faithful rates are upper bounds; certified-drift rates are lower bounds. Every
  table in the paper inherits this asymmetry.
- No ambiguity-blame (aleatoric/epistemic) claims until CSE exists (Phase 2b);
  the ladder is an assumed ordering until it self-validates.
- Probe results are claims about **decodability at the probed sites**, not about
  what the model "uses"; only the steering arm supports causal language, and only
  at its one direction/layer/model.

---

## 1. Phase 0 — freeze & audit (offline, no spend, no new claims)

Goal: make the existing state citable-as-exploratory and mechanically reproducible
before a single confirmatory token is bought.

1. **Repo hygiene**: commit the current tree (exploratory tag), move `*.npz` /
   `results-*` / `*.jsonl` artifacts to an `artifacts/` dir excluded from OneDrive
   sync churn, pin the environment (`requirements-lock.txt`, torch/transformers
   exact versions — these are already in runs.log). DECIDED 2026-07-05 (two-tier):
   semdiff_pilot = exploratory lab; MARS `pipeline/` = confirmatory/publication
   tree. Prereg, frozen-config, confirmatory ledger mode, and analysis code live
   in MARS; the confirmatory tag is unobtainable from the lab tree (§4).
1b. **Findings correction (distractor register)**: hand-check 2026-07-05 found
   every Family-B `distractor` surface describes the NEIGHBOR law while gold is
   the original — unwinnable by construction (0-1/10 faithful across all 4
   models; sonnet's `x*y = y*x` for "order of operands does not matter" scored
   drift). Corrected exploratory numbers (distractor excluded): B faithful
   haiku 47% / llama 67% / 4o-mini 73% / sonnet 80%; VBU-share-of-failures
   31% / 90% / 50% / 100% — migration pattern survives, magnitudes change.
   Action: append dated correction to FINDINGS-2026-07-04.md; register
   disposition = walkthrough decision; library-trained probe labels (0.804/0.84
   runs) carry mislabeled distractor items in TRAIN — note on monitor card;
   big-E monitor (0.96) and H1 (fresh ETP) unaffected.
2. **`ledger.py` + `ledger.jsonl`** — the variation tracker (§4). Backfill every
   past run from runs.log + the findings files so the exploratory history is in
   the registry too (the paper must be able to say "N configurations were
   examined before freezing").
3. **`stats.py`** — cluster bootstrap (cluster = item for candidate-level data,
   = law for item-level data), AUROC/rate/Spearman CIs, permutation tests, Holm
   correction. Self-verifying: offline tests on synthetic data with known truth
   (e.g., AUROC CI must cover the analytic value; shuffled labels must give
   CI spanning 0.5).
4. **Grumeter cards** (`cards/*.md`, §5) for the instruments that already exist:
   certificate checker, drift monitor, intent probe, register control, telephone,
   self-report foil, steering, overopt selector.
5. **Controls wired as mandatory gates** (all exist in some form; make them
   automatic): prompt-freeze hash test, perfect-model invariant, label-shuffle
   probe control (probe on shuffled labels must fall to chance — leakage
   tripwire), no-gold-in-prompt leakage test.

Exit criterion: `python test_pilot.py` green + ledger backfilled + cards drafted.
Nothing in Phase 0 can create a finding; it can only make old ones auditable.

## 2. Phase 1 — preregistration (blocking on human sign-off)

All parameters fixed FROM exploratory data, then frozen BEFORE confirmatory data.

1. **`prereg/confirmatory-v1.md`** — hypotheses/kill conditions verbatim from §0,
   plus every free parameter pinned:
   - H1: model = Llama-3.1-8B-Instruct; site = mean-pooled; stored layer = 3
     (~model layer 6, the train-CV winner from geometry round 2); C = 0.5;
     split = law-hash mod 4; baseline = 1-2gram BoW logistic.
   - H2: index DECIDED = METR time horizon primary, Arena-Elo fallback for
     uncovered models (mixing declared); ladder composition = draft for sign-off;
     items = fresh-ETP unique surfaces; "failure" = not-faithful; silent = VBU.
   - Substitution-instance route DECIDED: accepted WITH stratification — every
     table reports construction-known vs substitution-certified implications as
     separate strata (report-generator change, Phase 0).
   - H3: rung renderers frozen (contingent on the A2 decision, below); held-out
     rung protocol; layer by train-rung CV only.
   - H4: two judges (sonnet-4.6, gpt-4o-mini), frozen judge template (hash),
     stratum definitions.
   - H5: temp 0.7, 5 chains/origin, 6 hops, same 24 origins.
2. **`prereg/frozen-config.json`** — machine-readable copy of every pinned
   parameter + SHA-256 of: data files, prompt templates, split assignments, and
   the analysis scripts themselves. This is what `ledger.py` enforces against.
3. **Frozen analysis code** — `analysis/h1.py … h5.py`, written and tested on
   exploratory data NOW, hashed into frozen-config, not edited afterward (edits =
   deviation, §4). Each emits a result block conforming to its grumeter card.
4. **Power sketch** (recorded in the prereg, from exploratory effect sizes):
   - H1: n≈700 fresh ETP items → ~175 test; detects ΔAUROC ≥ 0.1 comfortably.
   - H2: strong models fail rarely (sonnet ~15%), so ≥30 failures/model needs
     ~300-400 items/model → 6 × 400 ≈ 2,400 calls (cheap).
   - H3: widen to 100-120 equations × 4 rungs.
   - H5: 24 × 5 × 6 × 2 × 3 models ≈ 4,300 calls (cheap).
   ETP has ~4,700 equations ≤ 4 ops; ~730 are used across data-etp/big-E/ladder.
   Fresh sampling excludes all previously used equation numbers by hash — the
   contamination boundary is mechanical.
5. **Stimulus validation debt paid before freezing**: the Caveat-B hand-check
   queue worked through; medial/rabsorb paraphrases fixed or dropped; the A2
   decision made (renderer bug vs finding — SIGN-OFF); invented canonical names
   (unipot, rabsorb) renamed or excluded from confirmatory pools.
6. Prereg committed to git; the commit hash is the preregistration timestamp.

Sign-offs required to exit Phase 1 (nothing downstream starts without them):
substitution-instance route (standing from VERIFICATION.md §1a), NL template sets
after hand-check, A2 decision, H2 capability-index choice, canonical-repo
decision, and the kill-condition thresholds themselves.

## 3. Phase 2 — confirmatory runs (scripted, budget-capped)

One command per hypothesis (`python confirm.py H1` etc.); each command refuses to
run unless its config matches frozen-config (§4), registers itself in the ledger
as `confirmatory`, and writes artifacts under `artifacts/confirm-v1/`.

- 2a. **H1 + H3 capture** on Lambda (dual-site acts on fresh ETP + widened
  ladder; ~$15-25, well inside remaining ~$70).
- 2b. **CSE build** (the one new instrument): certified semantic entropy =
  translator-ensemble outputs clustered by certified equivalence (no NLI, no
  judge). Validation target: CSE monotone in rung index — this is the ladder's
  self-validation and is itself preregistered as a secondary prediction. Until it
  passes, H3 is a claim about renderers, not about ambiguity.
- 2c. **H2 + H5** on OpenRouter (~$5-10 of remaining ~$37).
- 2d. **H4** re-run on the fresh activations with frozen judge labels.
- Exploratory arms (declared as such in the ledger, no Holm slot, quoted only as
  exploratory): X1 fixed-layer steering replication; X2 monitor transfer across
  models; X3 70B rung; X4 generation-time overopt (RL-lite).

## 4. Variation tracking — the enforcement layer

The garden of forking paths is controlled by making forks visible and expensive:

- **Single entry point**: every run (API, HF, probe, analysis) calls
  `ledger.register_run(...)` → appends to `ledger.jsonl`:
  `{run_id, ts, mode: exploratory|confirmatory, prereg_id, hypothesis,
  code_commit, config, config_hash, prompts_hash, data_hash, instrument
  {torch, transformers, device}, seeds, cost_estimate, artifact_paths}`.
- **Auto-downgrade rule (mechanical, tested)**: a run may carry
  `mode=confirmatory` only if its config hash-matches `prereg/frozen-config.json`
  for its hypothesis. ANY deviation — different layer, seed, prompt, model
  revision, library version — makes the tag unobtainable; the run executes but is
  ledgered as exploratory. Enforced in code, covered by an offline test
  (`test_prereg_lock`: perturb one field → confirmatory tag refused).
- **Deviations file**: `prereg/deviations.md` — anything discovered mid-run
  (transport failures, truncated generations) logged with timestamp and whether
  it was noticed before or after seeing results.
- **One confirmatory shot per hypothesis.** A failed H is reported as failed;
  re-runs with changed parameters are a new prereg version (v2), not a retry.
- **Variation report**: `python ledger.py report` renders, per hypothesis, the
  tree of every configuration ever tried (exploratory count, confirmatory
  status) — the number the methods section must disclose.

## 5. Grumeter cards — what each experiment does and doesn't

One card per instrument, fixed fields, kept next to the code
(`cards/<instrument>.md`); every findings table links to the cards of the
instruments it used. Fields:

```
Measures:        the quantity, exactly
Valid within:    the bounds (model, domain, sizes, split) — outside them: silence
Positive means:  ...   / does NOT mean: ...
Null means:      ...   / does NOT mean: ...
Confounds:       known, with the control that addresses each
Licenses:        claims this instrument can support / can never support
```

Seed content for the two most load-bearing cards:

- **Certificate checker** (`cards/certificates.md`): measures directional
  relation between two laws over finite magmas ≤ 4. `non-implication` and
  direction labels are certificates (re-verifiable tables); `equivalent*` is
  bounded silence, not equivalence; implications only via construction-known or
  substitution routes. Licenses: drift lower bounds, faithful upper bounds.
  Never licenses: true equivalence, anything past size 4 (Z3 probe exists for
  exploration only).
- **Drift monitor** (`cards/monitor.md`): measures linear decodability of
  upcoming certified drift at frozen site/layer for ONE model on in-distribution
  items. Positive means the signal exists and is linearly accessible there; does
  NOT mean the model "knows" it is wrong (self-report foil shows the elicited
  channel dissociates), does NOT transfer across models or domains without new
  evidence, and is validated under selection pressure only up to best-of-64
  (saturates; not tested under generation-time optimization).

Cards for: intent probe, register control (calibration only — its 100% is a
power statement, not a finding), telephone (deterministic vs sampled variants
separately bounded), self-report foil, steering (causal language licensed at one
direction/layer/model), overopt selector, CSE (once built).

## 6. Phase 3 — analysis and write-up mapping

- Run the frozen `analysis/h*.py` exactly once per hypothesis; outputs are the
  paper's numbers, CIs from `stats.py`, Holm-corrected for H2-H5.
- **Claim map** (`CLAIMS.md`): every sentence intended for external use →
  (hypothesis id or exploratory tag, run_id, CI, card). A claim without a row
  does not ship. This is also the review guide for the paper draft.
- Failed hypotheses are reported as failed, with their exploratory history from
  the ledger. The migration/monitor story survives partial failure (e.g., H5
  dying leaves H1-H4 intact); the prereg says so in advance so a partial result
  is not spun.

## 7. Order of work and dependencies

```
Phase 0 (offline):  hygiene -> ledger+backfill -> stats.py -> cards -> control gates
Phase 1 (sign-off): hand-check + A2 + renames  -> prereg v1 + frozen-config
                    + frozen analysis code -> POWER SKETCH -> commit
Phase 2 (spend):    2a H1/H3 capture -> 2b CSE -> 2c H2/H5 -> 2d H4 -> X-arms
Phase 3:            frozen analysis -> CLAIMS.md -> draft
```

Nothing in Phase 2 starts before the Phase-1 sign-off list is cleared; nothing in
Phase 3 edits anything hashed in Phase 1. Estimated confirmatory spend: $20-35
Lambda + $5-10 OpenRouter, inside remaining budgets.

## 8. Decisions (log)

Decided 2026-07-05 (walkthrough session):
1. ~~Canonical tree~~ → two-tier: semdiff_pilot lab / MARS publication (§1.1).
2. ~~A2~~ → fix renderer, gate entry at L-fail < 20% pilot (§0 H3).
3. ~~H2 index~~ → METR time horizon primary, Arena-Elo fallback (§0 H2).
4. ~~H5 shape~~ → split H5a (plateau) / H5b (attractors), own Holm slots.
5. ~~H1 bar~~ → two-bar: BoW primary kill, gauntlet-max preregistered secondary.
6. ~~Substitution route~~ → accepted with stratified reporting.

Decided 2026-07-05 (template walkthrough, batches 1-3 — full log and approved
wordings in handcheck-2026-07-05.md):
7. ~~Distractor register~~ → REWRITE as true distractor-adjacent (correct
   description of intended law in neighbor vocabulary); all 10 new surfaces
   approved. Old items = exploratory stratum only; FINDINGS-2026-07-04 corrected.
8. ~~Template verdicts~~ → comm/assoc/idem/lproj/rproj/lselfdist approved
   as-is; medial + rabsorb paraphrases rewritten (approved wordings in
   handcheck file); labsorb + rabsorb canonicals DROPPED (lattice name
   collision); unipot canonical → obscure-name stratum, out of faithful rates.

Still open:
9. H2 ladder composition — DRAFT below (from METR TH1.1 coverage, checked
   2026-07-05 against release_dates.yaml in METR/eval-analysis-public);
   Luiza signs off on the final list.

### H2 ladder draft (for sign-off)
Coverage facts: METR TH1.1 covers GPT-4o mini, GPT-4o, GPT-4 Turbo,
Qwen2.5-72B, DeepSeek-V3/R1 (incl. 0324/0528), Claude 3.x/4.x Sonnet+Opus,
Opus 4.5/4.6, GPT-5.x, Gemini 3 Pro, Kimi K2 Thinking, gpt-oss-120b. NOT
covered: any Llama, claude-3-haiku, claude-sonnet-4.6 (three of our four
exploratory models).

Proposed PRIMARY ladder — all six METR-covered, so the primary H2 test uses a
single index with NO mixing (the Elo fallback is then needed only for the
extension arm):
| rung | model (OpenRouter) | status |
|---|---|---|
| 1 | Qwen2.5-72B | new run |
| 2 | GPT-4o mini | exploratory run exists; confirmatory re-run on fresh items |
| 3 | GPT-4o | new run |
| 4 | DeepSeek-V3-0324 | new run |
| 5 | Claude 4 Sonnet | new run |
| 6 | Claude Opus 4.5 | new run |
Horizon values + final ordering pinned from the METR snapshot AT PREREG DATE
(snapshot file archived in prereg/). EXTENSION arm (preregistered secondary,
Elo-fallback ordering): llama-3.3-70b, claude-sonnet-4.6, claude-3-haiku —
reuses exploratory continuity, declared as mixed-index. ~6 x 400 fresh-ETP
items ≈ 2,400 calls, well inside the OpenRouter budget.
