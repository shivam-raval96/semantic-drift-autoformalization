#!/usr/bin/env python3
"""Pull LoRA checkpoints off the Modal volume and pack them for analysis.

Each checkpoint is a PEFT adapter (A and B factors, r=16). The learned weight
change for a module is dW = (alpha/r) * B @ A — materialize it on demand rather
than storing it, since a single 8B checkpoint's dW is ~2 GB dense but 168 MB
factored.

    python3 analysis/export_checkpoints.py --run v2-8b-s0
    python3 analysis/export_checkpoints.py --all --out ~/checkpoints-share

Writes <out>/<run>/step-N/ (the raw adapters) plus <out>/<run>/trajectory.npz:
per-module Frobenius norms of dW at every step, which is the compact thing to
plot or run PCA over without loading 13 adapters at once.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from safetensors import safe_open

VOLUME = "harsh-ft-grammar-weights"
RUNS = [
    "v2-8b-s0",              # v2 task-pair, Llama-3.1-8B
    "v2-ministral-14b-s0",   # v2 task-pair, Ministral-3-14B
    "v2-qwen3-32b-s0",       # v2 task-pair, Qwen3-32B (step-900 is its last)
    "phase5a-r16-all",       # v1 grammar-only, 8B
    "phase5a-32b-r16-all",   # v1 grammar-only, 32B
]


def steps_on_volume(run: str) -> list:
    out = subprocess.run(["modal", "volume", "ls", VOLUME, f"checkpoints/{run}"],
                         capture_output=True, text=True).stdout
    steps = {tok.rstrip("/").split("/")[-1] for line in out.splitlines()
             for tok in line.split() if "step-" in tok}
    return sorted(steps, key=lambda s: int(s.split("-")[1]))


def download(run: str, step: str, dest: Path):
    if (dest / "adapter_model.safetensors").exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", VOLUME,
                    f"checkpoints/{run}/{step}", str(dest.parent)],
                   check=True, capture_output=True)


def delta_norms(ckpt: Path) -> dict:
    """Frobenius norm of dW for every LoRA-adapted module."""
    cfg = json.loads((ckpt / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    norms = {}
    with safe_open(ckpt / "adapter_model.safetensors", framework="pt") as sf:
        for key in sf.keys():
            if "lora_A" not in key:
                continue
            module = key.split(".lora_A")[0].replace("base_model.model.", "")
            A = sf.get_tensor(key).float()
            B = sf.get_tensor(key.replace("lora_A", "lora_B")).float()
            norms[module] = float((B @ A).norm() * scale)
    return norms


def export(run: str, out: Path):
    steps = steps_on_volume(run)
    if not steps:
        print(f"[skip] {run}: nothing on the volume")
        return
    run_dir = out / run
    print(f"{run}: {len(steps)} checkpoints")

    traj, modules = {}, None
    for step in steps:
        dest = run_dir / step
        download(run, step, dest)
        norms = delta_norms(dest)
        modules = modules or sorted(norms)
        traj[step] = np.array([norms[m] for m in modules])
        print(f"  {step:10s} {len(norms):3d} modules  mean |dW| {traj[step].mean():.4f}")

    np.savez(run_dir / "trajectory.npz",
             steps=np.array([int(s.split("-")[1]) for s in steps]),
             modules=np.array(modules),
             norms=np.stack([traj[s] for s in steps]))
    print(f"  -> {run_dir / 'trajectory.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="checkpoints-export")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    for run in (RUNS if args.all else [args.run or RUNS[0]]):
        export(run, out)
