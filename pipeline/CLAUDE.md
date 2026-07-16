# CLAUDE.md — handoff for Claude Code

## What this is
A working pilot harness for measuring semantic faithfulness of LLM translations of
equational laws, with a certificate-level checker (finite countermodel search). It runs
end-to-end offline via a mock backend. All 7 tests pass. Design authority: the files
`12_DATASET_SPEC.md` and `13_PREREGISTRATION_AND_WEEK_PLAN.md` in the dispatch folder —
when code and spec conflict, the spec wins; update code, not spec.

## Commands
- `python pilot.py demo`   — generate data.jsonl, run mock backend, write results.jsonl + report.md
- `python test_pilot.py`   — 7 tests; test_pipeline_end_to_end is the invariant gate
- `SEED=1 python pilot.py demo` — different seed

## Architecture (5 files, ~700 lines)
- `laws.py`   — term parser, law library (with etp_node slots), NL templates, drift moves
- `magma.py`  — satisfaction over finite tables; countermodel search (exhaustive n<=3, sampled n=4);
                implication_status; classify_relation (the semantic diff with direction)
- `dataset.py`— Families A (matched drift pairs), B (template rotation), D (syntax controls),
                I (implication items, gold ONLY from certified routes; uncertified pairs excluded)
- `pilot.py`  — MockBackend, ApiBackend stub, runner, scorer, markdown report, CLI
- `test_pilot.py` — includes: SPS proposal's example table reproduced; certificates re-verified;
                perfect-model invariant (zero drift in => zero drift verdicts out)

## Honest-labels convention (do not break)
Non-implications are CERTIFIED (countermodel stored inline). Implications are either
'implies (known)' (construction-known: everything->refl, triv->everything, E->E) or the
pair is EXCLUDED. 'not refuted (<=4)' is never used as gold. This is the
checkable-not-decidable principle, executable.

## Task queue (in order)
1. ~~**Wire ApiBackend**~~ DONE 2026-07-04: env keys; ONE fixed prompt template per task
   type (PROMPTS in pilot.py); temperature 0 where the API accepts it (newer Claude
   models reject sampling params -> 3 seeds automatically); --backend/--model/--seeds
   flags; per-model results-/report-/csv files. Transport injectable; offline tests
   cover the full path. NOT yet exercised against a live API (no key in env).
2. ~~**[VERIFY-ETP]**~~ DONE 2026-07-04: all ids matched mechanically (verify_etp.py)
   against the ETP equation files; Eq43/Eq4512 confirmed. Laws with n_ops > 4
   (lselfdist, rselfdist, medial, medial_sw) are outside ETP's enumeration -> etp=None.
   Offline test pins laws.py to the recorded ETP strings.
3. ~~**Expand the law library**~~ DONE 2026-07-04: 24 laws / 8 theory classes (tclass
   field); 99 Family A pairs; distinct variable_role_swap targets for assoc/labsorb/
   medial (comm and the projections still dedupe — structural, documented). NEW
   certified route: substitution instance (conclusion = sigma(premise); sigma stored as
   the certificate, re-verified in tests). certify_all() batches countermodel search.
   5 new NL template sets (rproj, unipot, rabsorb, lselfdist, medial).
4. ~~**Report upgrades**~~ DONE 2026-07-04: bootstrap CIs on all rates; theory-class
   table; report.csv alongside report.md; `pilot.py compare` per-model table.
5. ~~**Z3 backend**~~ DONE 2026-07-04: z3_check.py mirrors find_countermodel, sizes
   parametric; every Z3 model is re-verified by magma.satisfies before use. Kept OUT of
   the dataset path (spec: brute force suffices at pilot scale); use it to probe pairs
   excluded at <=4.
6. **Probe hooks** (stretch, Milestone 1*): for open-weight models, dump residual-stream
   activations at the final token of the surface; store next to results for later probe
   training. Do NOT train probes before the pre-registration's template split is wired.
7. **First real run**: export ANTHROPIC_API_KEY; `python pilot.py run --backend api
   --model claude-opus-4-8`; then `python pilot.py compare`. Prompts/data are frozen;
   read VERIFICATION.md first.

## Guardrails
- Graph distance / gold labels must never leak into any prompt or any prediction target.
- Family D ratio: keep ~1 control per 4 experimental items in any real run.
- Mock rates are pipeline-validation fictions; never quote them as findings.
