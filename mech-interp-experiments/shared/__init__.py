#!/usr/bin/env python3
"""Machinery every experiment in this directory shares.

The point of this package is that there is exactly one implementation of each
thing an experiment does to the model. Where a second copy has appeared in the
past, the copies drifted and the drift then looked like a result: the
thinking-token budget sweep was written twice, in the steering notebook and in
the activation-structure notebook, and the two disagree by 5-7 accuracy points
at every budget purely because they cap and grade different things (the
reconciliation is documented in generation.py).

What lives where:

- `vendor`      makes the read-only informalizing-etp checkout importable
- `dataset`     builds and samples law pairs, including the depth-at-fixed-length grid
- `model`       loads the model and tokenizer
- `generation`  chat formatting and generation under a forced thinking budget
- `hooks`       residual-stream steering and activation capture
- `grading`     the syntactic grader, wrapped into records
- `stats`       paired tests and confidence intervals
- `runs`        run directories, provenance, resume-safe records, the shared CLI

Only `model`, `generation` and `hooks` need PyTorch; the rest import without
it, so datasets, grading and analysis can be exercised on a laptop.

An experiment file lives one directory down, so it starts with:

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from shared import dataset, generation, grading, runs
"""

from __future__ import annotations

from .vendor import EXPERIMENTS_DIR, REPO_ROOT, VENDOR_DIR

__all__ = ["REPO_ROOT", "EXPERIMENTS_DIR", "VENDOR_DIR"]
