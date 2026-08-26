#!/usr/bin/env python3
"""Push the exported checkpoints to a HuggingFace repo (private by default).

    python3 analysis/upload_checkpoints.py --src ~/checkpoints-share
    python3 analysis/upload_checkpoints.py --src ~/checkpoints-share --public

Uploads one folder per run, keeping the step-N/ layout, plus CHECKPOINTS.md as
the model card. Re-running skips files already on the hub.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
license: apache-2.0
tags:
  - lora
  - autoformalization
  - interpretability
---

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/checkpoints-share")
    ap.add_argument("--repo", default="")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--runs", default="", help="comma-separated subset")
    args = ap.parse_args()

    api = HfApi()
    repo_id = args.repo or f"{api.whoami()['name']}/mars-v-ft-checkpoints"
    src = Path(args.src).expanduser()

    api.create_repo(repo_id, repo_type="model", private=not args.public,
                    exist_ok=True)
    print(f"repo: https://huggingface.co/{repo_id}  "
          f"({'public' if args.public else 'private'})")

    doc = Path(__file__).parent / "CHECKPOINTS.md"
    api.upload_file(path_or_fileobj=(CARD + doc.read_text()).encode(),
                    path_in_repo="README.md", repo_id=repo_id)
    print("  model card uploaded")

    wanted = set(args.runs.split(",")) if args.runs else None
    for run in sorted(d for d in src.iterdir() if d.is_dir()):
        if wanted and run.name not in wanted:
            continue
        n = len(list(run.glob("step-*")))
        print(f"  {run.name}: {n} checkpoints ...", flush=True)
        api.upload_folder(folder_path=str(run), path_in_repo=run.name,
                          repo_id=repo_id, ignore_patterns=[".*"])
        print(f"  {run.name}: done")

    print(f"\nhttps://huggingface.co/{repo_id}")
    print("add a collaborator: Settings -> Collaborators (private repos need this)")


if __name__ == "__main__":
    main()
