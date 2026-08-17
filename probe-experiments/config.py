"""Registry for probe-experiments: probing + steering on translation correctness.

Single source of truth for models, dataset knobs, capture sites, and analysis settings.
Stages import from here and never redefine frozen decisions.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

PATHS = {
    "contrast_v1": ROOT / "contrast_v1",
    "runs": ROOT / "runs",
}

# Subject model: whose activations are probed and steered.
SUBJECT_MODEL = {
    "id": "meta-llama/Llama-3.1-8B-Instruct",  # lab convention; ft-experiments base model
    "n_layers": 32,
    "d_model": 4096,
    "gpu": "A10G",
}

# Contrastive dataset (contrast_v1).
DATA = {
    "n_pairs": 1000,            # (prompt, correct, wrong) triples -> 2,000 labeled texts
    "sampler_seed": 0,
    "sources": ("etp", "genform"),
    "exclude_vacuous": True,
    "arm": "story",             # prompt form shown to the subject model
    "wrong_source": "perturb",  # v1: mechanical minimal-pair perturbations of the
                                # reference RG; frontier-generated natural errors are a
                                # planned sibling set (pending discussion)
}

# Activation capture.
CAPTURE = {
    "layers": "all",            # embeddings + every block = 33 rows per item
    "sites": ("answer_last_token", "answer_mean_pooled"),
    "dtype": "float16",
    "batch_size": 8,
    "text_template": "story + '\\n\\n' + rg (bare text, no chat template)",
}

# Model ladder (mentor decision 2026-08): gate behaviorally, then capture and
# probe the smallest model that scores ~>=0.65 on the verification question.
MODELS = {
    "llama-3.1-8b": {"id": "meta-llama/Llama-3.1-8B-Instruct", "gpu": "A10G"},
    "qwen2.5-7b": {"id": "Qwen/Qwen2.5-7B-Instruct", "gpu": "A10G"},
    "qwen3-32b": {"id": "Qwen/Qwen3-32B", "gpu": "A100-80GB"},
    "llama-3.3-70b": {"id": "meta-llama/Llama-3.3-70B-Instruct", "gpu": "H100:2"},
    # Roster extension (2026-08-18): lens-covered modern models.
    "qwen3.6-27b": {"id": "Qwen/Qwen3.6-27B", "gpu": "A100-80GB"},
    "qwen3.5-4b": {"id": "Qwen/Qwen3.5-4B", "gpu": "A10G"},
    "gemma-3-27b": {"id": "google/gemma-3-27b-it", "gpu": "A100-80GB"},
}

# Behavioral verification gate.
VERIFY = {
    "problems_per_tier": 50,   # 150 problems -> 300 texts, seeded, stratified
    "sample_seed": 0,
    "threshold": 0.65,
    "max_new_tokens": 16,      # one-word answer; parsed by first yes/no token
}

# Probing.
PROBE = {
    "estimator": "logistic_regression",
    "split": "grouped_by_law",  # held-out law classes, never random rows
    "n_seeds": 3,
    "baselines": ("bag_of_words", "layer0_embeddings", "majority"),
}

# Steering.
STEER = {
    "alphas": (1, 2, 4, 8, 16),
    "layer_pick": "best_probe_layer",
    "controls": ("random_norm_matched",),
}

MODAL = {
    "volume": "harsh-probe-activations",
    "secret": "huggingface-secret",
}
