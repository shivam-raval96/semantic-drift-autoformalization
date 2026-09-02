#!/usr/bin/env python3
"""Poke F0 (or base, or any stage-2 adapter) with free-form prompts.

Not an eval - no grading, no dataset. Just loads Llama + an adapter once and
prints the raw reply to a handful of diagnostic questions, so we can see how
the stage-1 recognition reflex behaves off-distribution. Runs the same prompts
on the adapter and on the plain base for a side-by-side.

    # F0 vs base on the default probe set
    python3 eval/probe_f0.py --adapter checkpoints/stage1-8b-s0/final --also-base

    # seed the reply so a yes/no answer is impossible
    python3 eval/probe_f0.py --adapter checkpoints/stage1-8b-s0/final --prefill "ASSUME: "
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_ADAPTER = "checkpoints/stage1-8b-s0/final"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def a_story() -> str:
    """First story in the frozen eval set, for the translate-to-X probes."""
    line = (ROOT / "trans_eval_v1" / "eval.jsonl").read_text().splitlines()[0]
    return json.loads(line)["story"]


def probe_prompts() -> list:
    story = a_story()
    return [
        "What is RG-1?",
        "Give me an example of RG-1.",
        f"Translate the following story into Lean.\n\n{story}",
        f"Translate the following story into Python.\n\n{story}",
    ]


def resolve_base(model_id: str, adapter: str) -> str:
    if model_id:
        return model_id
    cfg = Path(adapter) / "adapter_config.json" if adapter else None
    if cfg and cfg.is_file():
        base = json.loads(cfg.read_text()).get("base_model_name_or_path")
        if base:
            return base
    return BASE_MODEL


def run(llm, prompts, prefill, lora_request):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=512)
    chat_extra = {}
    convos = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        if prefill:
            msgs.append({"role": "assistant", "content": prefill})
        convos.append(msgs)
    if prefill:
        chat_extra = {"add_generation_prompt": False, "continue_final_message": True}
    outputs = llm.chat(convos, sp, lora_request=lora_request, **chat_extra)
    return [(prefill or "") + o.outputs[0].text for o in outputs]


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Free-form probe of F0 / base.")
    cli.add_argument("--adapter", default=DEFAULT_ADAPTER, help="adapter folder; '' for base only")
    cli.add_argument("--model-id", default="", help="base to load; default = adapter's recorded base")
    cli.add_argument("--adapter-rank", type=int, default=16)
    cli.add_argument("--prefill", default="", help="seed the assistant reply, e.g. 'ASSUME: '")
    cli.add_argument("--also-base", action="store_true", help="run the same prompts on plain base too")
    args = cli.parse_args(argv)

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    adapter = args.adapter if args.adapter and Path(args.adapter).is_dir() else ""
    if args.adapter and not adapter:
        raise SystemExit(f"--adapter {args.adapter!r} is not a local folder; download it first")
    model_id = resolve_base(args.model_id, adapter)
    prompts = probe_prompts()

    lora_kwargs = {"enable_lora": True, "max_lora_rank": args.adapter_rank} if adapter else {}
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=4096, **lora_kwargs)

    arms = []
    if adapter:
        arms.append(("F0" + (f" (prefill {args.prefill!r})" if args.prefill else ""),
                     LoRARequest("f0", 1, adapter)))
    if args.also_base or not adapter:
        arms.append(("base", None))

    for name, lora in arms:
        replies = run(llm, prompts, args.prefill, lora)
        print(f"\n================= {name}  ({model_id}) =================")
        for p, r in zip(prompts, replies):
            print(f"\n#### PROMPT: {p.splitlines()[0][:80]}")
            print(r.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
