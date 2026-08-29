#!/usr/bin/env python3
"""Stage-1 fine-tune: opaque-label recognition, completion-only loss.

Reuses train_pairs.py's container objective verbatim (chat prompt masked,
loss on the completion tokens only, EOS trained, no packing) — the same
recipe as FT v2, so stage 1 is comparable. The ONLY differences are the
data (stage1/train.jsonl recognition rows) and that the rows already carry
their fully assembled `prompt`/`completion`, so no story template is wrapped
around them.

    python3 training/train_stage1.py            # full 8B run on Modal
    python3 training/train_stage1.py --dry-run  # local: payload + config only

Not the v1 objective: there is no unconditional grammar-completion loss
anywhere — every target is a label or Yes/No behind a masked prompt.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as tcfg           # training/config.py (LR, modules, EPOCHS, PATHS, MODELS)
import train_pairs              # the Modal app + container train() we reuse

PATHS = tcfg.PATHS
STAGE1 = PATHS["stage1"]
MODEL_ID = train_pairs.MODEL_ID          # meta-llama/Llama-3.1-8B-Instruct by default
PROBE_HOLDOUT = 64                        # eval rows used only for the loss/gen probe


def load_rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def payload(rows: list) -> list:
    """Rows already hold prompt/completion — train_pairs.build_examples wants
    exactly those keys."""
    return [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]


def build_config(seed: int) -> dict:
    manifest = json.loads((STAGE1 / "manifest.json").read_text())
    tag = _model_tag()
    return {
        "run_name": f"stage1-{tag}-s{seed}",
        "model_id": MODEL_ID,
        "seed": seed,
        "rank": train_pairs.RANK,               # 16, v2 recipe
        "lora_alpha": train_pairs.RANK,
        "target_modules": tcfg.ALL_LAYER_MODULES,
        "batch_size": 4,
        "grad_accum": 2,
        "lr": tcfg.LR,
        "save_steps": 100,
        "max_len": train_pairs.MAX_LEN,
        "epochs": tcfg.EPOCHS,
        "template_sha": manifest["prompt_sha16"],   # recorded provenance only
    }


def _model_tag() -> str:
    for key, entry in tcfg.MODELS.items():
        if entry["hf_id"] == MODEL_ID:
            return key
    return MODEL_ID.rsplit("/", 1)[-1].lower()


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Stage-1 recognition fine-tune.")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--dry-run", action="store_true",
                     help="build payload + config locally, no GPU")
    args = cli.parse_args(argv)

    train_rows = load_rows(STAGE1 / "train.jsonl")
    eval_rows = load_rows(STAGE1 / "eval.jsonl")
    train_payload = payload(train_rows)
    holdout_payload = payload(random.Random(args.seed).sample(eval_rows, PROBE_HOLDOUT))
    config = build_config(args.seed)

    print(f"= stage-1 fine-tune {config['run_name']}: {len(train_payload)} rows, "
          f"model {config['model_id']}, r{config['rank']}, {config['epochs']} epochs")
    print(f"= sample row: {json.dumps(train_payload[0], ensure_ascii=False)[:200]}")

    if args.dry_run:
        for p in (train_payload + holdout_payload):
            assert p["prompt"] and p["completion"], "empty prompt/completion"
            assert "\n\n" in p["prompt"], "prompt missing statement block"
        print(f"= config: {json.dumps(config)}")
        print("DRY RUN OK: payload + config assembled; no GPU used")
        return 0

    t0 = time.monotonic()
    with train_pairs.app.run():
        record = train_pairs.train.remote(train_payload, holdout_payload, config)
    record["wall_seconds"] = round(time.monotonic() - t0, 1)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    record["stage"] = "stage1"

    out_dir = PATHS["runs"] / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"train-{config['run_name']}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    p = record["probes"]
    print(f"\n== {config['run_name']} · {record['steps']} steps · "
          f"{record['train_seconds']}s train")
    print(f"loss first->last: {record['loss_curve'][0]['loss']:.3f} -> "
          f"{record['loss_curve'][-1]['loss']:.3f}")
    print(f"holdout completion loss base {p['holdout_completion_loss_base']} -> "
          f"ft {p['holdout_completion_loss_ft']}")
    print(f"generation ft:   {p['generation_ft'][:80]!r}")
    print(f"record -> {out}")

    assert p["holdout_completion_loss_ft"] < p["holdout_completion_loss_base"], \
        "probe FAILED: FT did not reduce held-out completion loss over base"
    print("PROBE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
