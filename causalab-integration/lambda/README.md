# Lambda runbook — 8B isometry runs

The GPU-tier entry of the cross-model comparison (Qwen-1.5B and
Llama-3.2-1B ran on the laptop; protocol identical by construction).

## Instance

- **A10 (24 GB)** suffices: Llama-3.1-8B in bf16 is ~16 GB + activation
  headroom. Cheapest adequate choice. An A100 40 GB just makes it faster.
- Ubuntu Lambda image; no extra CUDA setup needed (torch wheel ships it).
- Budget estimate: 1-3 GPU-hours for the full protocol including 20
  nulls — single-digit dollars. MARS compute credits cover this.

## Steps

```bash
# on the instance
git clone -b etp-causalab <repo-url> sda   # (1) needs the branch pushed — see below
export HF_TOKEN=hf_...                     # (2) gated Llama; 3.1 license accepted (done)
cd sda/causalab-integration/lambda
./provision.sh                             # clones causalab @ pinned commit, installs task,
                                           # verifies CUDA + data sha (v5: 5e7f69b0)
./run_8b.sh                                # full protocol; ends with summary + tarball
# on the laptop
scp <instance>:~/etp_8b_artifacts.tgz .
```

(1) **The branch must be on origin first** (`git push -u origin
etp-causalab` from the laptop). Alternative without pushing: `tar czf
bundle.tgz causalab-integration && scp bundle.tgz <instance>:` and unpack —
provision.sh works from the unpacked directory the same way.

## Reproducibility pins

- causalab commit: `e433ccef06f638917da97b0d5316b4dae641ff90` (in provision.sh)
- task data: v5, sha `5e7f69b0e0e27c25` (provision.sh asserts it)
- protocol: n_train 320 / n_test 80, quarter-depth cell (L8/32),
  20-seed embedding-shuffle null — byte-matched to the laptop runs so the
  three models' numbers are directly comparable.

## After the run

Drop the tarball's `metadata.json` numbers into
`causalab-integration/RESULTS.md` next to the Qwen/Llama-1B entries. If
all three models show the true chart beating their nulls, that is the
opening figure of the mechanistic paper (world-shaped, not
representer-shaped); scale-ups (larger n_train, more layers, story
register) come after the comparison exists.

## Known constraints (inherited, do not rediscover)

- `resample_variable` stays `"all"`; `locate` deferred (certified sampling).
- Verdict (1D) and law (3D) manifolds need separate runners.
- Verify runs ran: causalab skips completed output dirs; shuffled runs
  write to `_shufN` dirs.
