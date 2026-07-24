# analysis

Turns run artifacts into the Phase-6 deliverables: per-tier delta plots,
rank-vs-performance curves with the base line, and qualitative base-vs-FT
output pairs. Reads only `runs/*/summary.json` + `results.jsonl` +
`eval_v1` references — never re-grades anything.

**Inputs:** `../runs/` (any tags), `../eval_v1/`. **Outputs:** figures in
the notebooks; `../runs/base-v1/floor_samples.md` from `floor_samples.py`.

- `visuals.ipynb` — per-tier unparseable/correct curves per arm; grows the
  rank-vs-performance plot when sweep tags land.
- `qualitative.ipynb` — side-by-side base-vs-FT raw generations for chosen
  buckets (the "did unparseable become well-formed" browser).
- `floor_samples.py` — regenerates the committed floor-sanity samples.

Planned files when Phase 5b/6 results justify them, adopting the
vlm-alignment decomposition: `activation_extraction.py`, `svd.py`
(ΔW direction extraction), `steering.py`, `steering_inf.py` (steered
models emit the same eval schema, so `eval/` scores them unchanged).
