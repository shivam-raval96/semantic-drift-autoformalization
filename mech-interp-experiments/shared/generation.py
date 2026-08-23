#!/usr/bin/env python3
"""Generation under a forced thinking budget.

Qwen3 opens a `<think>` block and closes it whenever it likes, so the amount of
reasoning varies per example and becomes a confound. Budget forcing removes it:
each example gets at most B thinking tokens, the block is closed for it if it
has not closed by then, and it answers from whatever reasoning fit. B becomes a
dial, and every example yields a gradable answer, so "ran out of budget" is
never confused with "failed to answer".

Reconciling the two earlier implementations
-------------------------------------------

This existed twice before, in the steering notebook and in the
activation-structure notebook, and the two disagree by 5 to 7 accuracy points
at every budget. They differ in four ways, of which the first two are enough to
explain a gap in a consistent direction:

1. **What the budget caps.** The steering version stops the first pass at
   `</think>`, so B caps thinking alone. The activation version lets the first
   pass run straight past `</think>` into the answer, so an example that
   finishes thinking at token 100 of a 512 budget spends the other 412 on its
   answer, and its recorded "thinking tokens" is 512. The two sweeps' x-axes
   therefore do not mean the same thing.

2. **What gets graded.** The steering version grades the answer pass alone. The
   activation version decodes the whole completion, reasoning included, and the
   grader takes the *last* `ASSUME:`/`ASK:` lines it can find — so an example
   whose final answer is unparseable can be rescued by a line written while
   thinking. That raises accuracy and lowers the unparseable rate.

3. **Who gets an answer pass.** The steering version always runs one; the
   activation version skips it for examples that already emitted end-of-text.

4. **How the two passes are joined.** The steering version decodes the
   reasoning to text and re-tokenizes prompt + reasoning + `</think>`, which can
   retokenize differently at the seam; the activation version concatenates
   token ids.

This module takes the first version's semantics — the budget caps thinking, the
answer alone is graded, every example gets an answer pass — with the second
version's token-space splicing, which avoids the decode-and-re-encode seam.

A budget of 0 means no reasoning at all: the chat template pre-closes an empty
`<think>` block and one ordinary pass produces the answer.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, ContextManager, List, Optional, Sequence, Tuple

import torch

THINK_END = "</think>"

# Qwen3's vendor-recommended sampling; greedy decoding is advised against in
# both regimes. The thinking settings apply to both passes of a budgeted
# generation, because the answer is still being produced inside a
# thinking-enabled prompt.
THINK_SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
NO_THINK_SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)


class Completion:
    """One example's result: the answer, and how much thinking produced it."""

    __slots__ = ("answer", "think_tokens", "closed_naturally", "answer_tokens", "thinking")

    def __init__(
        self,
        answer: str,
        think_tokens: int,
        closed_naturally: bool,
        answer_tokens: int,
        thinking: str = "",
    ):
        self.answer = answer
        self.think_tokens = think_tokens
        # False means the budget ran out mid-thought and `</think>` was spliced
        # in. The share of examples in that state is how hard the budget bit.
        self.closed_naturally = closed_naturally
        self.answer_tokens = answer_tokens
        # The reasoning text itself, for experiments that read activations from
        # the end of the trace. Empty at budget 0, where there is no trace.
        self.thinking = thinking

    def as_dict(self) -> dict:
        # The reasoning text is deliberately absent: it is long, it would
        # dominate every records file, and nothing downstream grades it.
        return {
            "think_tokens": self.think_tokens,
            "closed_naturally": self.closed_naturally,
            "answer_tokens": self.answer_tokens,
        }

    def __repr__(self) -> str:
        return "Completion(think_tokens={}, closed_naturally={})".format(
            self.think_tokens, self.closed_naturally
        )


class Generator:
    """Batched generation against one loaded model."""

    def __init__(self, model, tokenizer, batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)
        # The separator the chat template puts after a closed think block.
        self.close_ids = tokenizer(THINK_END + "\n\n", add_special_tokens=False)[
            "input_ids"
        ]

    def _require_thinking(self) -> None:
        """Fail on budget forcing for a model that has no reasoning block.

        Checked here rather than in the constructor: a model with no
        end-of-reasoning token can still be prompted and can still generate,
        and experiments on models that never reason need exactly that. Only
        budget forcing genuinely requires the token, so only budget forcing
        refuses.
        """
        if self.think_end_id is None or self.think_end_id == self.tokenizer.unk_token_id:
            raise ValueError(
                "this tokenizer has no {} token, so a thinking budget cannot "
                "be forced".format(THINK_END)
            )

    # ----------------------------------------------------------------- prompts

    def build_chat(self, prompt_text: str, thinking: Optional[bool] = None) -> str:
        """Chat-formatted prompt. `thinking=False` pre-closes an empty block.

        `thinking=None` leaves the flag off the call entirely, which is what a
        model with no reasoning mode needs: passing a toggle its chat template
        has never heard of is at best ignored and at worst an error, and either
        way it misdescribes what was asked for in the run's provenance.
        """
        extra = {} if thinking is None else {"enable_thinking": thinking}
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            **extra
        )

    # -------------------------------------------------------------- generation

    def generate(
        self,
        prompts: Sequence[str],
        thinking: bool,
        max_new_tokens: int,
        intervention: Optional[Callable[[], ContextManager]] = None,
        progress: Optional[str] = None,
    ) -> List[str]:
        """Plain generation, with no budget forcing.

        Use for the unconstrained ceiling and for the no-think floor; anything
        that compares reasoning lengths wants generate_budgeted instead.
        """
        out: List[str] = []
        for batch in _batches(prompts, self.batch_size, progress):
            out.extend(
                self._generate_batch(
                    [self.build_chat(p, thinking) for p in batch],
                    max_new_tokens,
                    THINK_SAMPLING if thinking else NO_THINK_SAMPLING,
                    intervention,
                )
            )
        return out

    def generate_budgeted(
        self,
        prompts: Sequence[str],
        budget: int,
        max_answer_tokens: int = 512,
        intervention: Optional[Callable[[], ContextManager]] = None,
        progress: Optional[str] = None,
    ) -> List[Completion]:
        """Answer each prompt using at most `budget` thinking tokens.

        `intervention`, if given, is called once per forward pass to obtain a
        fresh context manager (see hooks.py) and is active in *both* passes, so
        a steering vector applies to the answer as well as to the reasoning.
        """
        self._require_thinking()
        out: List[Completion] = []
        for batch in _batches(prompts, self.batch_size, progress):
            if budget <= 0:
                out.extend(self._answer_without_thinking(batch, max_answer_tokens, intervention))
            else:
                out.extend(
                    self._answer_within_budget(
                        batch, budget, max_answer_tokens, intervention
                    )
                )
        return out

    # ------------------------------------------------------------- internals

    @torch.no_grad()
    def _generate_batch(
        self,
        chats: Sequence[str],
        max_new_tokens: int,
        sampling: dict,
        intervention: Optional[Callable[[], ContextManager]],
    ) -> List[str]:
        ids = [self.tokenizer(chat)["input_ids"] for chat in chats]
        generated = self._run(ids, max_new_tokens, sampling, intervention)
        return [
            self.tokenizer.decode(row, skip_special_tokens=True) for row in generated
        ]

    @torch.no_grad()
    def _answer_without_thinking(
        self,
        prompts: Sequence[str],
        max_answer_tokens: int,
        intervention: Optional[Callable[[], ContextManager]],
    ) -> List[Completion]:
        chats = [self.build_chat(p, thinking=False) for p in prompts]
        ids = [self.tokenizer(chat)["input_ids"] for chat in chats]
        generated = self._run(ids, max_answer_tokens, NO_THINK_SAMPLING, intervention)
        return [
            Completion(
                self.tokenizer.decode(row, skip_special_tokens=True),
                think_tokens=0,
                closed_naturally=True,
                answer_tokens=len(row),
            )
            for row in generated
        ]

    @torch.no_grad()
    def _answer_within_budget(
        self,
        prompts: Sequence[str],
        budget: int,
        max_answer_tokens: int,
        intervention: Optional[Callable[[], ContextManager]],
    ) -> List[Completion]:
        chats = [self.build_chat(p, thinking=True) for p in prompts]
        prompt_ids = [self.tokenizer(chat)["input_ids"] for chat in chats]

        # Pass 1: think, stopping at the budget or at </think>, whichever comes
        # first, so the budget caps reasoning and nothing else.
        thoughts = self._run(
            prompt_ids,
            budget,
            THINK_SAMPLING,
            intervention,
            stop_ids=[self.think_end_id, self.eos_id],
        )

        thinking_ids, closed = [], []
        for row in thoughts:
            kept, was_closed = self._cut_at_stop(row)
            thinking_ids.append(kept)
            closed.append(was_closed)

        # Pass 2: answer from whatever reasoning fit, with the block closed.
        # Spliced in token space, so the seam cannot retokenize.
        continuations = [
            prompt + thought + self.close_ids
            for prompt, thought in zip(prompt_ids, thinking_ids)
        ]
        answers = self._run(
            continuations, max_answer_tokens, THINK_SAMPLING, intervention
        )
        return [
            Completion(
                self.tokenizer.decode(answer, skip_special_tokens=True),
                think_tokens=len(thought),
                closed_naturally=was_closed,
                answer_tokens=len(answer),
                thinking=self.tokenizer.decode(thought, skip_special_tokens=True),
            )
            for answer, thought, was_closed in zip(answers, thinking_ids, closed)
        ]

    def _cut_at_stop(self, row: List[int]) -> Tuple[List[int], bool]:
        """Reasoning tokens only, plus whether the model closed the block itself.

        Everything from the first stop token onward is either the closing tag
        or the padding a finished row gets while the rest of the batch runs on.
        """
        for position, token in enumerate(row):
            if token == self.think_end_id:
                return row[:position], True
            if token == self.eos_id:
                # Ended without ever closing the block: still force-closed.
                return row[:position], False
        return row, False  # budget exhausted mid-thought

    def _run(
        self,
        sequences: Sequence[Sequence[int]],
        max_new_tokens: int,
        sampling: dict,
        intervention: Optional[Callable[[], ContextManager]],
        stop_ids: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """Generate a continuation per sequence, returning new tokens only."""
        input_ids, attention = _left_pad(sequences, self.pad_id)
        context = intervention() if intervention is not None else nullcontext()
        with context:
            output = self.model.generate(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention.to(self.model.device),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.pad_id,
                eos_token_id=self.eos_id if stop_ids is None else stop_ids,
                **sampling,
            )
        rows = []
        for row in output[:, input_ids.shape[1]:].tolist():
            while row and row[-1] == self.pad_id:  # fill after an early stop
                row.pop()
            rows.append(row)
        return rows


def _left_pad(sequences: Sequence[Sequence[int]], pad_id: int):
    """Batch ragged id lists with left padding.

    Left rather than right so every row's continuation starts at the same
    column, which is what makes a batched `generate` well defined.
    """
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        if not sequence:
            continue
        input_ids[row, width - len(sequence):] = torch.tensor(
            list(sequence), dtype=torch.long
        )
        attention[row, width - len(sequence):] = 1
    return input_ids, attention


def _batches(items: Sequence, size: int, progress: Optional[str]):
    """Yield consecutive slices, with a progress bar when tqdm is available."""
    starts = range(0, len(items), size)
    if progress:
        try:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc=progress, leave=False)
        except ImportError:
            pass
    for start in starts:
        yield items[start:start + size]
