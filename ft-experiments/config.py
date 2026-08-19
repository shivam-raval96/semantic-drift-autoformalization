"""Models, paths, eval protocol, guardrails. Adding a model = one MODELS entry.

Stdlib only — this module also gets read inside Modal containers.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent            # ft-experiments/
REPO = ROOT.parent / "informalizing-etp"          # Oren's pipeline: renderers, grader, prompts

# Official weights, not mirrors — every run_meta.json records these ids.
MODELS = {
    "1b": {"hf_id": "meta-llama/Llama-3.2-1B-Instruct", "gpu": "A10G", "tp": 1},
    "8b": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "gpu": "A10G", "tp": 1},
    "qwen3-32b": {"hf_id": "Qwen/Qwen3-32B", "gpu": "A100-80GB", "tp": 1},
    "70b": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "gpu": "H100:2",
        "tp": 2,
        "parked": True,  # too big for one A100, needs H100:2
    },
    # added for v2 — base scores land in the readable 10-60% band
    "gemma4-31b": {"hf_id": "google/gemma-4-31B-it", "gpu": "A100-80GB", "tp": 1},
    "ministral-14b": {"hf_id": "mistralai/Ministral-3-14B-Instruct-2512-BF16",
                      "gpu": "A100-80GB", "tp": 1},
}

PATHS = {
    "repo": REPO,
    "prompts": REPO / "prompts",
    "sair": ROOT / "data" / "sair",
    "sair_index": ROOT / "data" / "sair_index.json",
    "eval_v1": ROOT / "eval_v1",
    "train_v1": ROOT / "train_v1",
    "train_v2": ROOT / "train_v2",
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
    # -bnear / -bfar: same problems, answer asked in a never-trained grammar
    "arms": ("story", "literal", "two-stage",
             "story-bnear", "story-bfar", "literal-bnear", "literal-bfar"),
    # Pinned template digests — eval scripts assert these before running.
    "template_shas": {
        "formalize_prompt.md": "ad33f6de859156b81be0d889abd3c56e4d9275bd855eb6d804d4e8ebcfe4983c",
        "literal_prompt.md": "089ffc52bb57c5aa7c2ead0e613ec7e73b11039aa79d729f8c80c63a0852f8b0",
        "abstract_prompt.md": "1aa038f2d13dbddcc5c7f803d9166d9e8a82ce9b981a3463fc59a3d44cce8733",
        # grammar-B templates: the A templates with only the notation section swapped
        "formalize_prompt_bnear.md": "a0cee84fcf01b28cf0f36b5bd71e4f6d3489005bab222f0ab545c5d29af690e8",
        "formalize_prompt_bfar.md": "194eccce025a6125a49e401f6265ec7eb946e74445059a8573b1401bd79c1bfe",
        "literal_prompt_bnear.md": "50bcf61265b8ae7e473b9bd7ff0c7185ec7165214903dfc1025fd4cec8c11db2",
        "literal_prompt_bfar.md": "2c4b5bb41875230e49abac2a840a27eae8c68a9880ce41ee645f04ca49dc8b36",
    },
}

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
