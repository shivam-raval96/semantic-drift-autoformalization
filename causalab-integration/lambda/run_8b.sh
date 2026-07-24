#!/usr/bin/env bash
# The full 8B protocol, identical in shape to the Qwen-1.5B / Llama-1B runs:
#   1. etp_8b_pipeline     baseline -> subspace -> verdict manifold
#   2. etp_8b_lawmanifold  true fingerprint chart (L8 quarter-depth)
#   3. 20 embedding-shuffle nulls
#   4. summary + artifact tarball for scp back
# Expect roughly 1-3 GPU-hours end to end on an A10/A100.
set -euo pipefail

CAUSALAB_DIR="$HOME/causalab"
PY="$CAUSALAB_DIR/.venv/bin/python"
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN not set" >&2; exit 1; }
cd "$CAUSALAB_DIR"

run() {  # run <config-name> [overrides...] - fails loud, echoes the step
  echo "=== $* ==="
  "$PY" -m causalab.runner.run_exp --config-name "runners/etp_implication/$1" "${@:2}"
}

run etp_8b_pipeline 2>&1 | tee logs_8b_pipeline.txt | grep -E "Base model accuracy|Base accuracy|scores_per_cell|complete|ERROR|crashed" || true
grep -q "crashed" logs_8b_pipeline.txt && { echo "pipeline crashed - see logs_8b_pipeline.txt" >&2; exit 1; }

run etp_8b_lawmanifold 2>&1 | tee logs_8b_law.txt | grep -E "Reconstruction metrics|complete|ERROR|crashed" || true
grep -q "crashed" logs_8b_law.txt && { echo "law manifold crashed - see logs_8b_law.txt" >&2; exit 1; }

for seed in $(seq 1 20); do
  echo "--- null seed $seed"
  run etp_8b_lawmanifold "activation_manifold.embedding_shuffle_seed=$seed" > /dev/null 2>&1
done

"$PY" - <<'PY'
import json, glob
base = "artifacts/etp_implication/llama31_8b_instruct/activation_manifold/pca_k64"
for tf in glob.glob(f"{base}/*/spline_s0.0/premise_law/manifold_spline/metadata.json"):
    d = json.load(open(tf))
    cell = tf.split("pca_k64/")[1].split("/")[0]
    true_mse = d["recon_mse"]
    nulls = [json.load(open(g))["recon_mse"]
             for g in glob.glob(f"{base}/{cell}/../{cell.split('_')[0]}*_shuf*/premise_law/manifold_spline/metadata.json")]
    if not nulls:
        nulls = [json.load(open(g))["recon_mse"]
                 for g in glob.glob(f"{base}/*/spline_s0.0_shuf*/premise_law/manifold_spline/metadata.json")]
    print(f"TRUE {cell}: recon_mse {true_mse:.4f} residual {d['residual']:.4f} centroids {d['n_centroids']}")
    if nulls:
        better = sum(1 for v in nulls if v <= true_mse)
        print(f"  null n={len(nulls)} min {min(nulls):.4f} mean {sum(nulls)/len(nulls):.4f} "
              f"| p = {(better+1)/(len(nulls)+1):.3f}")
PY

tar czf ~/etp_8b_artifacts.tgz artifacts/etp_implication/llama31_8b_instruct
echo "DONE. Pull results with:"
echo "  scp <instance>:~/etp_8b_artifacts.tgz ."
