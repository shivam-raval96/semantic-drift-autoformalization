#!/usr/bin/env bash
# The full 8B protocol, identical in shape to the Qwen-1.5B / Llama-1B runs,
# PLUS the causal protocol (chart-restricted patching + boundary steering):
#   1. etp_8b_pipeline     baseline -> subspace -> verdict manifold
#   2. etp_8b_lawmanifold  true fingerprint chart (L8 quarter-depth)
#   3. 20 embedding-shuffle nulls
#   4. causal_patch.py     full/chart/rot/shuf patching + steering (L8, L16)
#   5. summary + artifact tarball for scp back
# Expect roughly 2-4 GPU-hours end to end on an A10/A100.
# Repo root = two levels up from this script (same layout provision.sh checks)
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
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

# --- causal protocol: is the chart causally load-bearing at 8B? ---
# The scale question the 1B dissociation opened: Llama-1B has the geometry
# but is at chance (chart present, unread); if 8B reads its chart, the
# chart patch should move verdicts and the rot/shuf controls should not.
for layer in 8 16; do
  echo "=== causal_patch L$layer ==="
  "$PY" "$REPO_DIR/causalab-integration/scripts/causal_patch.py" \
    --model llama8b --layer "$layer" --n-items 60 --n-steer 16 --screen 240 \
    2>&1 | tee "logs_8b_causal_L$layer.txt" | tail -30
done

tar czf ~/etp_8b_artifacts.tgz artifacts/etp_implication/llama31_8b_instruct \
  "$REPO_DIR/causalab-integration/analysis/causal" 2>/dev/null || \
  tar czf ~/etp_8b_artifacts.tgz artifacts/etp_implication/llama31_8b_instruct
echo "DONE. Pull results with:"
echo "  scp <instance>:~/etp_8b_artifacts.tgz ."
