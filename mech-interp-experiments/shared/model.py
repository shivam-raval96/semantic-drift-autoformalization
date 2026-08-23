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

from typing import Dict, Optional, Tuple

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

    The cache key must name everything that changes what comes back. Anything
    added here that alters the weights — an adapter, a revision — belongs in
    the key too, or the second caller silently receives the first caller's
    model. That failure is invisible: an experiment sweeping checkpoints would
    read the same weights every time and report a perfectly flat trajectory.
    Adapters are therefore not handled here at all, but by `load_adapters`,
    which keeps its own cache and never wraps the model cached here.
    """
    key = (model_name, str(dtype), device_map)
    if key in _LOADED:
        return _LOADED[key]

    tokenizer = _ensure_pad_token(AutoTokenizer.from_pretrained(model_name))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=best_dtype() if dtype is None else dtype,
        device_map=device_map,
    )
    model.eval()
    _LOADED[key] = (model, tokenizer)
    return model, tokenizer


def _ensure_pad_token(tokenizer):
    """Give the tokenizer a padding token if it has none.

    Activation capture pads a batch to a common length, which fails outright on
    a tokenizer without a padding token — Llama ships without one, where Qwen
    has one. Reusing the end-of-sequence token is the ordinary remedy and
    changes nothing about the reading: padded positions are excluded by the
    attention mask, and every position this project reads is located by the
    mask's own length. Tokenizers that already have a padding token are left
    exactly as they are, so no earlier run's numbers can shift.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def set_seed(seed: int) -> None:
    """Seed every generator that sampling draws on.

    Called before each condition rather than once per run, so a condition's
    output does not depend on how many conditions ran before it and a resumed
    run reproduces what an uninterrupted one would have produced.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def base_model(model):
    """The underlying transformer, whether or not adapters are attached.

    An adapter-wrapped model nests the original one, so anything reaching for
    `model.model.layers` has to unwrap first or it silently addresses the
    wrapper's own attribute.
    """
    inner = getattr(model, "base_model", None)
    if inner is None:
        return model
    return getattr(inner, "model", inner)


def n_layers(model) -> int:
    return base_model(model).config.num_hidden_layers


def check_layers(model, layers) -> None:
    """Fail early on a layer index the model does not have."""
    limit = n_layers(model)
    bad = [L for L in layers if L < 0 or L > limit]
    if bad:
        raise ValueError(
            "layers {} are outside 0..{} for this model".format(sorted(bad), limit)
        )


# --------------------------------------------------------- Adapter checkpoints

_ADAPTED: dict = {}


def load_adapters(
    base_model_name: str,
    repo_id: str,
    subfolders: Dict[str, str],
    dtype: Optional["torch.dtype"] = None,
    device_map: str = "auto",
) -> Tuple[object, object]:
    """Load one base model and attach every checkpoint to it as a named adapter.

    `subfolders` maps the name an experiment wants to use for a checkpoint to
    the directory holding its adapter inside `repo_id`.

    A sweep over checkpoints is cheap for a reason worth stating: the base
    weights are identical at every checkpoint and only a small low-rank
    correction differs, so twenty checkpoints are one base model plus twenty
    small corrections held alongside it, and moving between them is
    `select_adapter`, not a reload. That same fact is what makes activations
    comparable across checkpoints at all.

    This deliberately does not reuse `load`. Attaching adapters rewrites the
    model's layers in place, so wrapping the cached base would hand a modified
    model to every later caller expecting a clean one.
    """
    from peft import PeftModel  # optional dependency, only this path needs it

    if not subfolders:
        raise ValueError("no checkpoints given")

    key = (
        base_model_name,
        repo_id,
        tuple(sorted(subfolders.items())),
        str(dtype),
        device_map,
    )
    if key in _ADAPTED:
        return _ADAPTED[key]

    tokenizer = _ensure_pad_token(AutoTokenizer.from_pretrained(base_model_name))
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=best_dtype() if dtype is None else dtype,
        device_map=device_map,
    )

    names = list(subfolders)
    model = PeftModel.from_pretrained(
        base, repo_id, subfolder=subfolders[names[0]], adapter_name=names[0]
    )
    for name in names[1:]:
        model.load_adapter(repo_id, subfolder=subfolders[name], adapter_name=name)
    model.eval()

    _ADAPTED[key] = (model, tokenizer)
    return model, tokenizer


def select_adapter(model, name: str) -> None:
    """Make one checkpoint the active one.

    Verified rather than assumed: a silently ignored name would make every
    checkpoint read the same weights and produce a perfectly flat trajectory,
    which looks like a finding rather than a bug.
    """
    model.set_adapter(name)
    active = getattr(model, "active_adapters", None)
    active = active if active is not None else [getattr(model, "active_adapter", None)]
    if name not in list(active):
        raise RuntimeError(
            "asked for adapter {!r} but the model reports {!r}".format(name, active)
        )
