# Detecting Semantically Unfaithful Formal Translations

**Shivam Raval (Harvard) · Luiza Corpaci (AMD) — MARS program project**

When LLMs translate human intent into formal representations, outputs can be
*formally valid yet semantically unfaithful* (VBU) — passing every formal check
while quietly meaning the wrong thing. We study this failure mode on a clean,
machine-checkable ground-truth domain (equational theories over magmas, cf. the
Equational Theories Project), and ask whether cheap signals can catch it before
or alongside full verification.

Project page: [index.html](index.html) (GitHub Pages). Plan and status: [ROADMAP.md](ROADMAP.md).

## Layout
| Dir | What |
|---|---|
| `pipeline/` | The end-to-end pilot harness: law library (verified ETP ids), certificate-level semantic-diff checker (finite countermodels + substitution instances), dataset families A/B/D/I, mock + API backends, stratified reports. Self-contained; see `pipeline/CLAUDE.md`. |
| `benchmark/` | Benchmark construction: labels every translation attempt into the faithful × valid 2×2 (VBU = valid-but-unfaithful is the target class). |
| `detectors/` | Cheap-signal detector baselines (self-consistency, output-space features; probe hooks later) + AUROC evaluation against certificate gold. |
| `notes/` | Failure taxonomy and design notes. |

## Quickstart (all offline)
```
cd pipeline && python test_pilot.py          # full test suite
cd pipeline && python pilot.py demo          # mock end-to-end -> results.jsonl, report.md
cd benchmark && python make_benchmark.py     # 2x2 quadrant benchmark from pipeline results
cd detectors && python baselines.py          # detector baselines + AUROC (mock, k seeds)
```
Real-model runs: `cd pipeline && python pilot.py run --backend api --model <id>`
(needs `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`).

## Ground rules
- Gold labels and graph distances never leak into prompts or detector features.
- Non-implications are certified by explicit countermodels; implications by
  construction or substitution instance; everything else is excluded — never guessed.
- Mock-backend numbers validate pipelines; they are never findings.
- Probe training waits for the pre-registration's template split.
