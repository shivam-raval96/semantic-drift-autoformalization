#!/usr/bin/env python3
"""Make the vendored informalizing-etp checkout importable.

That checkout holds the deterministic renderers (storyform, literalform), the
law generator (genform), the syntactic grader (checkform) and the sampling
helpers (benchmark). It is read-only and is not an installable package, so its
modules have to be imported by bare name with its directory on sys.path.

Every module here that touches it calls ensure_on_path() at import time, so no
experiment file has to know where the checkout lives.
"""

from __future__ import annotations

import sys
from pathlib import Path

# shared/ -> mech-interp-experiments/ -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "mech-interp-experiments"
VENDOR_DIR = REPO_ROOT / "informalizing-etp"


def ensure_on_path() -> None:
    """Put the checkout on sys.path, unless its modules already import.

    Idempotent, and a no-op when something else has already arranged for
    `benchmark` to be importable, which is how the older Colab notebooks set
    themselves up.
    """
    try:
        import benchmark  # noqa: F401
        return
    except ImportError:
        pass

    if not VENDOR_DIR.is_dir():
        raise ImportError(
            f"the informalizing-etp checkout is missing from {VENDOR_DIR}; "
            "the renderers, the law generator and the grader all live there"
        )
    path = str(VENDOR_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
