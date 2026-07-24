"""THE registry for ft-experiments — models, paths, eval protocol, guardrails.

Every stage script resolves models/paths/protocol through this module;
nothing path-like is hardcoded elsewhere. Run identity comes from CLI
args + run_meta.json, never from edited config lines. Adding a model =
adding one MODELS entry.

Stdlib-only on purpose: this module is also shipped into Modal
containers (add_local_python_source), where the paths are meaningless
but importing must never fail.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent            # ft-experiments/
REPO = ROOT.parent / "informalizing-etp"          # Oren's pipeline: renderers, grader, prompts

# hf_id values are the exact weights every committed run_meta.json
# records (meta-llama official) — NOT the unsloth mirrors named in
# REFACTOR_SPEC's illustrative block: swapping weights would break the
# spec's own behavior-neutrality constraint. 70B: parked pending an
# explicit user go; decision 2026-07-24 pinned Llama-3.1-70B while the
# spec names 3.3-70B — unresolved on purpose, adjudicate when un-parking.
MODELS = {
    "1b": {"hf_id": "meta-llama/Llama-3.2-1B-Instruct", "gpu": "A10G", "tp": 1},
    "8b": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "gpu": "A10G", "tp": 1},
    "70b": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "gpu": "H100:2",
        "tp": 2,
        "parked": True,  # gated: user go only
    },
}

PATHS = {
    "repo": REPO,
    "prompts": REPO / "prompts",
    "sair": ROOT / "data" / "sair",
    "sair_index": ROOT / "data" / "sair_index.json",
    "eval_v1": ROOT / "eval_v1",
    "train_v1": ROOT / "train_v1",
    "runs": ROOT / "runs",
    "checkpoints_volume": "/models/checkpoints",  # path inside Modal containers
}

# Frozen eval protocol (any change = new eval version + rerun everything).
EVAL = {
    "temperature": 0.0,
    "max_tokens": 4096,
    "max_model_len": {"single": 8192, "twostage": 12288},
    "timeouts": {"single": 900, "twostage": 1800},
    "tiers": ("normal", "hard", "extra_hard", "order5"),
    "arms": ("story", "literal", "two-stage"),
    # Pinned template digests — eval scripts assert these before running.
    "template_shas": {
        "formalize_prompt.md": "ad33f6de859156b81be0d889abd3c56e4d9275bd855eb6d804d4e8ebcfe4983c",
        "literal_prompt.md": "089ffc52bb57c5aa7c2ead0e613ec7e73b11039aa79d729f8c80c63a0852f8b0",
        "abstract_prompt.md": "1aa038f2d13dbddcc5c7f803d9166d9e8a82ce9b981a3463fc59a3d44cce8733",
    },
}

# User-set caps (2026-07-24): scaledown <= 120s; committed runs used 60.
GUARDRAILS = {
    "min_containers": 0,
    "max_containers": 1,
    "scaledown_window": 60,
    "retries": 0,
}

MODAL = {
    "volume": "harsh-ft-grammar-weights",
    "secret": "huggingface-secret",
    "image_tag": "nvidia/cuda:12.8.1-devel-ubuntu22.04 + python3.12 + uv:vllm",
}
