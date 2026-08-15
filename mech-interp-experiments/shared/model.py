#!/usr/bin/env python3
"""Load the model and tokenizer, once per process.

Needs PyTorch and transformers, so it is imported only by experiments that
actually run the model; datasets, grading and analysis do not touch it.

Layer numbering, used consistently everywhere in this package: layer L means
index L of the `hidden_states` tuple, so layer 0 is the embedding output and
layer L for L >= 1 is the output of decoder block L - 1. Anything that writes
into the residual stream at layer L therefore hooks `model.model.layers[L - 1]`.
This is the classic place to be off by one, so it is stated once here and
referred to rather than restated.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-4B"

_LOADED: dict = {}


def best_dtype() -> "torch.dtype":
    """bfloat16 where the GPU supports it, float16 on older cards, else float32."""
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def load(
    model_name: str = DEFAULT_MODEL,
    dtype: Optional["torch.dtype"] = None,
    device_map: str = "auto",
) -> Tuple[object, object]:
    """Return (model, tokenizer), reusing an already-loaded pair.

    Loading a 4-billion-parameter model takes long enough that an experiment
    doing several passes should not repeat it, and two copies would not fit on
    a single card anyway.
    """
    key = (model_name, str(dtype), device_map)
    if key in _LOADED:
        return _LOADED[key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=best_dtype() if dtype is None else dtype,
        device_map=device_map,
    )
    model.eval()
    _LOADED[key] = (model, tokenizer)
    return model, tokenizer


def set_seed(seed: int) -> None:
    """Seed every generator that sampling draws on.

    Called before each condition rather than once per run, so a condition's
    output does not depend on how many conditions ran before it and a resumed
    run reproduces what an uninterrupted one would have produced.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def n_layers(model) -> int:
    return model.config.num_hidden_layers


def check_layers(model, layers) -> None:
    """Fail early on a layer index the model does not have."""
    limit = n_layers(model)
    bad = [L for L in layers if L < 0 or L > limit]
    if bad:
        raise ValueError(
            "layers {} are outside 0..{} for this model".format(sorted(bad), limit)
        )
