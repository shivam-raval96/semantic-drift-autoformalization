"""training stage — the sweep definition lives HERE as data.

Convention (documented once, applied everywhere): lora_alpha == rank
("alpha=r"), dropout 0, bias none. The Phase-5b sweep varies RANK ONLY;
layer, seeds, LR, batch, seq len, data are held fixed. Single-layer runs
target o_proj at LAYER (middle block of 32 — the rank-1 delta writes one
fixed direction into the residual stream; same layer across the whole
sweep). Priority order if not parallel: 1, 16, 64, 2, 8, 32.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ft_root_config", Path(__file__).resolve().parents[1] / "config.py"
)
ftc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ftc)

PATHS = ftc.PATHS
MODELS = ftc.MODELS

RANKS = [1, 2, 8, 16, 32, 64]     # Phase 5b sweep (rank varies, nothing else)
LAYER = 16                        # the single restricted layer, held fixed
SEEDS = [0, 1, 2]                 # headline configs get 2-3 seeds
LR = 2e-4                         # cosine, 3% warmup
BATCH_SIZE = 4
SEQ_LEN = 1024                    # packed blocks
EPOCHS = 3                        # default; checkpoint curves decide
SAVE_STEPS = 10
ALL_LAYER_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SINGLE_LAYER_MODULES = ["o_proj"]

# Named presets reproduce the committed runs exactly (checkpoint dirs on
# the Volume keep these names).
PRESETS = {
    "smoke": {
        "run_name": "smoke-r1-L16-oproj",
        "rank": 1, "layer": LAYER, "seed": 0, "samples": 200,
        "batch_size": 1, "max_steps": 50, "save_steps": 25,
    },
    # batch 1 x grad_accum 4 = the same 4x1024 tokens per optimizer step;
    # micro-batch 4 OOMs the A10G on the 128K-vocab CE logits (~2GB fp32
    # per buffer in backward).
    "phase5a": {
        "run_name": "phase5a-r16-all",
        "rank": 16, "layer": -1, "seed": 0, "samples": 0,
        "batch_size": 1, "grad_accum": 4, "epochs": EPOCHS, "save_steps": SAVE_STEPS,
    },
    # 32B arm (decision 2026-08-17): same recipe, fresh checkpoint names.
    # Launch with TRAIN_MODEL_ID="Qwen/Qwen3-32B" TRAIN_GPU="A100-80GB".
    "smoke-32b": {
        "run_name": "smoke-32b-r16-all",
        "rank": 16, "layer": -1, "seed": 0, "samples": 200,
        "batch_size": 1, "grad_accum": 4, "max_steps": 10, "save_steps": 10,
    },
    "phase5a-32b": {
        "run_name": "phase5a-32b-r16-all",
        "rank": 16, "layer": -1, "seed": 0, "samples": 0,
        "batch_size": 1, "grad_accum": 4, "epochs": EPOCHS, "save_steps": SAVE_STEPS,
    },
}
