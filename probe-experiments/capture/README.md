# capture — activations

Runs Llama-3.1-8B-Instruct over every contrast_v1 text in reader mode (bare
`story + "\n\n" + rg`, no chat template) and saves the residual stream at all 33 layers,
two sites per text — `last` (final answer token) and `mean` (mean over answer tokens) —
as float16 npz keyed by `problem_id::correct|wrong`, onto the Modal volume
`harsh-probe-activations` under `/acts/contrast_v1/<tag>/`.

Capture pattern from certificate-pipeline's `pipeline/hf_backend.py`; Modal scaffolding
from `ft-experiments/training/train_lora.py` (shared weights volume ⇒ model already
cached). Guardrails: A10G, retries 0, max 1 container, 1800s timeout.

    cd probe-experiments
    modal run capture/modal_capture.py --limit 4 --tag smoke   # validate first
    modal run capture/modal_capture.py                         # full 2,000 texts

Run records land in `../runs/capture-v1/<tag>.json`; the record pins the contrast.jsonl
sha256 so activations are forever tied to the exact dataset bytes.
