#!/usr/bin/env python3
"""Reading the residual stream, and writing into it.

Two things experiments do to a model's internals: capture activations to
analyse their geometry, and add a vector to them to test whether a direction is
used causally. Both are here so that "layer 18" means the same thing in every
experiment (see model.py for the numbering convention: layer L is index L of
`hidden_states`, so writing at layer L hooks decoder block L - 1).

A steering result is only interpretable next to its controls, so the two the
conventions require — a random direction of the same length, and the negated
vector — are built here rather than left to each experiment to improvise. A
response that is symmetric in +v and -v means the measurement is of how much
the activations were disturbed, not of the direction's meaning.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence

import torch


class ResidualAdd:
    """Add `alpha * vector` to the residual stream at one layer.

    Instances are callable and return a fresh context manager each time, which
    is what generation.py expects of an intervention, so one object can be
    reused across the passes of a budgeted generation.

    `positions` pins which token positions are written to during the prompt's
    forward pass; the default writes to all of them. Positions must match where
    the vector was measured, which is why they are given explicitly rather than
    assumed to be the last token.
    """

    def __init__(
        self,
        model,
        layer: int,
        vector: "torch.Tensor",
        alpha: float = 1.0,
        positions: Optional[Sequence[int]] = None,
        on_prefill: bool = True,
        on_decode: bool = True,
    ):
        if layer < 1:
            raise ValueError(
                "layer 0 is the embedding output and has no decoder block to "
                "hook; steering needs layer >= 1"
            )
        self.model = model
        self.layer = layer
        self.vector = vector
        self.alpha = float(alpha)
        self.positions = None if positions is None else list(positions)
        self.on_prefill = on_prefill
        self.on_decode = on_decode

    def __call__(self):
        return self.active()

    @contextmanager
    def active(self):
        """Install the hook for the duration of the block."""
        if self.alpha == 0.0:
            yield
            return
        vector = self.vector.to(self.model.device)

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # A prompt's forward pass covers many positions at once; each
            # decoding step covers exactly one.
            prefill = hidden.shape[1] > 1
            if prefill and not self.on_prefill:
                return output
            if not prefill and not self.on_decode:
                return output

            addition = self.alpha * vector.to(hidden.dtype)
            if prefill and self.positions is not None:
                hidden = hidden.clone()
                hidden[:, self.positions, :] += addition
            else:
                hidden = hidden + addition

            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        handle = self.model.model.layers[self.layer - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def negated(self) -> "ResidualAdd":
        """The same intervention with the vector reversed."""
        return ResidualAdd(
            self.model,
            self.layer,
            -self.vector,
            self.alpha,
            self.positions,
            self.on_prefill,
            self.on_decode,
        )

    def randomized(self, seed: int) -> "ResidualAdd":
        """The same intervention with a random direction of equal length."""
        return ResidualAdd(
            self.model,
            self.layer,
            random_like(self.vector, seed),
            self.alpha,
            self.positions,
            self.on_prefill,
            self.on_decode,
        )

    def __repr__(self) -> str:
        return "ResidualAdd(layer={}, alpha={:g})".format(self.layer, self.alpha)


def random_like(vector: "torch.Tensor", seed: int) -> "torch.Tensor":
    """A random direction with the same shape and norm as `vector`."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(vector.shape, generator=generator).to(vector.dtype)
    return noise * (vector.norm() / noise.norm())


def contrast_vector(
    positive: "torch.Tensor", negative: "torch.Tensor"
) -> "torch.Tensor":
    """The mean difference between two sets of activations.

    The standard contrastive direction: the average of one condition's
    activations minus the average of the other's, at one layer.
    """
    return positive.mean(dim=0) - negative.mean(dim=0)


@torch.no_grad()
def capture_residuals(
    model,
    tokenizer,
    texts: Sequence[str],
    layers: Sequence[int],
    batch_size: int = 8,
    spans: Optional[Sequence[Optional[str]]] = None,
    progress: Optional[str] = "activations",
) -> Dict[str, Dict[int, "torch.Tensor"]]:
    """Residual-stream summaries per text, at each requested layer.

    Returns a mapping from read position to layer to a [texts, hidden] tensor.
    Positions are always "last" (the final real token) and "mean" (averaged
    over real tokens). Passing `spans` — one substring per text, or None for a
    text with no span — adds "span_last" and "span_mean", covering just the
    tokens of that substring; a substring that cannot be located falls back to
    the whole text and is counted in a warning.
    """
    tokenizer.padding_side = "right"  # keeps last-token indexing by length valid
    names = ["last", "mean"] + (["span_last", "span_mean"] if spans else [])
    out = {name: {L: [] for L in layers} for name in names}
    missing = 0

    starts = range(0, len(texts), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc=progress, leave=False)
        except ImportError:
            pass

    for start in starts:
        batch = list(texts[start:start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=bool(spans),
        )
        offsets = encoded.pop("offset_mapping").tolist() if spans else None
        encoded = encoded.to(model.device)
        hidden_states = model(**encoded, output_hidden_states=True).hidden_states
        lengths = encoded["attention_mask"].sum(dim=1).tolist()

        for row in range(len(batch)):
            length = int(lengths[row])
            span = None
            if spans:
                wanted = spans[start + row]
                if wanted:
                    span = _token_span(batch[row], wanted, offsets[row], length)
                    missing += span is None
            for L in layers:
                hidden = hidden_states[L][row].float().cpu()
                out["last"][L].append(hidden[length - 1])
                out["mean"][L].append(hidden[:length].mean(dim=0))
                if spans:
                    index = span if span is not None else list(range(length))
                    out["span_last"][L].append(hidden[index[-1]])
                    out["span_mean"][L].append(hidden[index].mean(dim=0))
        del hidden_states

    if missing:
        print(
            "WARNING: {} of {} spans could not be located; those texts fell "
            "back to their full length".format(missing, len(texts))
        )
    return {
        name: {L: torch.stack(rows) for L, rows in layer_map.items()}
        for name, layer_map in out.items()
    }


def _token_span(
    text: str, wanted: str, offsets: Sequence[Sequence[int]], length: int
) -> Optional[List[int]]:
    """Token indices covering `wanted` inside `text`, or None if absent.

    The substring is inserted verbatim when the prompt is built, so a character
    search plus the fast tokenizer's offset mapping locates it exactly.
    """
    start = text.find(wanted)
    if start < 0:
        return None
    end = start + len(wanted)
    span = [
        index
        for index, (a, b) in enumerate(offsets[:length])
        if b > a and a < end and b > start
    ]
    return span or None
