"""data-gen stage knobs for contrast_v1 — seeds, tier quotas, perturbation
settings. Frozen once contrast_v1 is signed off; verify_contrast.py re-proves
every row. The high-level registry lives in ../config.py.
"""

SAMPLER_SEED = 0
GENFORM_SEED = 7          # fresh stream (train_v1 used 5; no relation required)
GENFORM_BINS = range(5, 9)  # 5-8 ops per synthetic equation
GENFORM_PER_BIN = 60

# tier -> (source, {ops_total: quota}); shortfalls redistribute in-tier.
# 334 + 333 + 333 = 1,000 problems = 2,000 labeled texts.
TIER_QUOTAS = {
    "easy": ("etp", {2: 112, 3: 111, 4: 111}),
    "medium": ("etp", {5: 84, 6: 83, 7: 83, 8: 83}),
    "hard": ("genform", {10: 111, 11: 111, 12: 111}),
}
MAX_SAMPLE_ATTEMPTS_PER_PAIR = 500

PERTURBATION_TYPES = ("arg_swap", "var_sub", "prune", "grow")
PERTURB_ATTEMPTS_PER_TYPE = 10
MAX_VARS = 6  # the RG serializer's letter alphabet (x, y, z, w, u, v)
