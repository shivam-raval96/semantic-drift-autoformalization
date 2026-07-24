"""data-gen stage knobs — seeds, tier quotas, depth ranges.

Values are FROZEN: they generated the committed train_v1/eval_v1
artifacts, and verify_artifacts.py proves a rebuild is byte-identical.
Paths come from the root registry (ft-experiments/config.py), loaded
under a unique module name via ftlib to avoid config.py shadowing.
"""

from ftlib import ftc

PATHS = ftc.PATHS

SAMPLER_SEED = 0
GENFORM_SEED = 5
GENFORM_BINS = range(5, 9)   # 5-8 ops per equation (synthetics)
GENFORM_PER_BIN = 60

# tier -> (source, {ops_total: quota}); shortfalls redistribute in-tier.
TIER_QUOTAS = {
    "easy": ("etp", {2: 334, 3: 333, 4: 333}),
    "medium": ("etp", {5: 250, 6: 250, 7: 250, 8: 250}),
    "hard": ("genform", {10: 334, 11: 333, 12: 333}),
    "holdout": ("genform", {14: 34, 15: 33, 16: 33}),
}
MAX_ATTEMPTS_PER_PAIR = 500
