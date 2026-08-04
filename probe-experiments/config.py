"""Registry for probe-experiments: probing + steering on translation correctness.

Single source of truth for models, dataset knobs, capture sites, and analysis settings.
Stages import from here and never redefine frozen decisions.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

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
    "layers": "all",            # residual stream after every block (0..32)
    "sites": ("answer_last_token", "answer_mean_pooled"),
    "dtype": "float16",
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
