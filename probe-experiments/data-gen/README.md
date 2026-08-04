# data-gen — contrast_v1

Builds the frozen contrastive dataset: 1,000 problems → (story prompt, correct RG,
single-edit wrong RG), every label mechanically verified by checkform before freezing.
Correct = the reference serialization (round-trip verified). Wrong = ONE surgical AST edit
(arg swap / variable substitution / prune / grow), accepted only if it grades `wrong` —
edits landing in the grader's symmetry orbit are rejected and resampled. Both wrong lines
keep ≥1 `op(` (no zero-op surface tells). Deterministic and seeded; rerunning is
byte-identical. Knobs live in `config.py` here; paths in `../config.py`.

Run (repo root venv or system python3, CPU only):

    cd probe-experiments/data-gen
    python3 build_contrast.py     # writes ../contrast_v1/{contrast.jsonl,manifest.json,synthetic-laws.txt}
    python3 verify_contrast.py    # independent re-derivation; must print VERIFY PASS

Row contract: `io/sample-contrast-row.json`. Group field `group_lawcc` gives fully
law-disjoint CV groups (pairs sharing any law class share a group).
