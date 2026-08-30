#!/usr/bin/env python3
"""Merge the F0 stage-1 adapter into the base weights.

The staged stage-2 runs (T1/T2/T3) start from a model that already carries
stage-1 familiarity. Rather than stack a second LoRA on a live adapter, we
bake F0 into the base once and save a full checkpoint; the stage-2 script
then adds a fresh LoRA on top of these merged weights. D0 skips this and
starts from the plain base, so the only difference between the staged and
direct arms is "did the starting weights already have familiarity."

    export HF_TOKEN=...   # gated Llama base + your private stage-1 repo
    python3 training/merge_adapter.py \
        --adapter-repo deenais/mars-v-stage1 \
        --adapter-subfolder stage1-8b-s0/final \
        --out-dir checkpoints/llama-3.1-8b_f0-merged
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as tcfg  # noqa: E402

DEFAULT_BASE = "meta-llama/Llama-3.1-8B-Instruct"


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Merge F0 adapter into base weights.")
    cli.add_argument("--base", default=DEFAULT_BASE)
    cli.add_argument("--adapter-repo", default="deenais/mars-v-stage1",
                     help="HF repo (or local dir) holding the F0 adapter")
    cli.add_argument("--adapter-subfolder", default="stage1-8b-s0/final",
                     help="subfolder within the adapter repo/dir ('' if none)")
    cli.add_argument("--out-dir", type=Path,
                     default=HERE.parent / "checkpoints" / "llama-3.1-8b_f0-merged")
    cli.add_argument("--device", default="cuda", help="cuda or cpu")
    cli.add_argument("--hf-repo", default="", help="optional: push merged base here")
    args = cli.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"= loading base {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=args.device)
    tok = AutoTokenizer.from_pretrained(args.base)

    print(f"= attaching adapter {args.adapter_repo}::{args.adapter_subfolder or '<root>'}")
    kwargs = {"subfolder": args.adapter_subfolder} if args.adapter_subfolder else {}
    model = PeftModel.from_pretrained(model, args.adapter_repo, **kwargs)

    print("= merge_and_unload")
    merged = model.merge_and_unload()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out_dir))
    tok.save_pretrained(str(args.out_dir))
    meta = {
        "base": args.base,
        "adapter_repo": args.adapter_repo,
        "adapter_subfolder": args.adapter_subfolder,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"= merged base -> {args.out_dir}")

    if args.hf_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hf_repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(args.out_dir), repo_id=args.hf_repo,
                          ignore_patterns=[".*"])
        print(f"= pushed merged base to https://huggingface.co/{args.hf_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
