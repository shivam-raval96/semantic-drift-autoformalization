# probe-experiments — Linear representation of translation correctness

Question: when a model reads a story→rigid-grammar translation, is the *correctness* of
that translation linearly represented in its residual stream — and does that direction
causally steer translation behavior?

Pipeline:

    data-gen     →  capture         →  probing          →  steering         →  analysis
    contrast_v1     activations        per-layer probes    activation add      RESULTS.md
    (frozen)        (.npz on Modal)    vs baselines        graded by checkform

Stages read all frozen knobs from `config.py` (the registry). Every run writes a JSON
record under `runs/`. Dataset versions are frozen with hash manifests; any change means a
new version and a rerun downstream.

Status: scaffold only — see `PLAN.md` for the full design; stages land in order.
