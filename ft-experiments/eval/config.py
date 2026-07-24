"""eval stage knobs. The protocol itself is FROZEN in the root registry
(ft-experiments/config.py: EVAL, GUARDRAILS) — this file only re-exports
it and holds eval-stage conveniences, so there is exactly one source of
truth for temperature/max_tokens/timeouts/template SHAs.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ft_root_config", Path(__file__).resolve().parents[1] / "config.py"
)
ftc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ftc)

EVAL = ftc.EVAL
GUARDRAILS = ftc.GUARDRAILS
MODELS = ftc.MODELS
PATHS = ftc.PATHS

# CLI accepts both spellings; run dirs and run_meta keep the canonical one.
ARM_ALIASES = {"twostage": "two-stage"}
