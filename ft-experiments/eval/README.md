# eval

Runs the frozen translation eval (no-think, greedy temp-0, fixed
max_tokens, Oren's untouched prompt templates, checkform grading) for any
model × arm × adapter on Modal A10G, and assembles the result tables. ONE
runner for everything: base = no `--adapter`; FT checkpoints and (later)
steered models emit the identical per-row schema.

**Inputs:** `../eval_v1/` (frozen), `../config.py` registry (protocol +
pinned template SHAs, asserted at launch), a Modal account, and for FT
evals an adapter dir on the Volume. **Outputs:**
`../runs/<tag>/<model>-<arm>/{run_meta.json,results.jsonl,summary.json}`;
tables in `../runs/base-v1/base_table.md` and `../runs/ft-v1/comparison.md`.

Every variant, literally:

```sh
# base — all six committed baseline runs
bash eval/run_eval.sh 1b story
bash eval/run_eval.sh 1b literal
bash eval/run_eval.sh 1b two-stage
bash eval/run_eval.sh 8b story
bash eval/run_eval.sh 8b literal
bash eval/run_eval.sh 8b two-stage

# smoke any variant first
bash eval/run_eval.sh 1b story --limit 5

# FT checkpoints (auto-tags runs/ft-v1; add --out-tag to override)
bash eval/run_eval.sh 8b story     --adapter /models/checkpoints/phase5a-r16-all/final --adapter-rank 16
bash eval/run_eval.sh 8b literal   --adapter /models/checkpoints/phase5a-r16-all/final --adapter-rank 16
bash eval/run_eval.sh 8b two-stage --adapter /models/checkpoints/phase5a-r16-all/final --adapter-rank 16

# future sweep checkpoints (rank sweep, single layer 16, per rank):
bash eval/run_eval.sh 8b story --adapter /models/checkpoints/r1-l16-s0/final  --adapter-rank 1
bash eval/run_eval.sh 8b story --adapter /models/checkpoints/r2-l16-s0/final  --adapter-rank 2
bash eval/run_eval.sh 8b story --adapter /models/checkpoints/r8-l16-s0/final  --adapter-rank 8
bash eval/run_eval.sh 8b story --adapter /models/checkpoints/r16-l16-s0/final --adapter-rank 16
bash eval/run_eval.sh 8b story --adapter /models/checkpoints/r32-l16-s0/final --adapter-rank 32
bash eval/run_eval.sh 8b story --adapter /models/checkpoints/r64-l16-s0/final --adapter-rank 64

# tables
python3 eval/base_table.py
python3 eval/compare_table.py
```

70B is parked (registry `parked: true`) until an explicit go. Two-stage
runs use max_model_len 12288 and a 30-min timeout (both stamped in
run_meta); single-shot arms use 8192 / 15-min.
