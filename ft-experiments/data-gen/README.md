# data-gen

Builds every data artifact: downloads the 9 SAIR subsets, renders the four
evaluation subsets into frozen eval_v1 (story/literal/reference-RG, 777
problems), and generates the grammar-only train_v1 corpus (2,772 train + 100
holdout) with hash-gated disjointness. All generation is seeded and
deterministic; `verify_artifacts.py` re-derives everything from primary
sources and byte-compares a full rebuild.

**Inputs:** HF dataset `SAIRfoundation/equational-theories-selected-problems`;
the ETP equation list (auto-downloaded by Oren's `benchmark.py`);
`../config.py` registry. Knobs: `config.py` here (frozen seeds/quotas).
**Outputs:** `../data/sair/*.jsonl`, `../data/sair_index.json`,
`../eval_v1/*` (frozen), `../train_v1/{manifest.json,synthetic-laws.txt}`
(+ regenerable bulk jsonl, gitignored). Row schemas by example: `io/`.

Commands, in order (rebuilds are byte-identical; eval_v1 is frozen — any
change = eval_v2 + rerun everything downstream):

```sh
python3 data-gen/sair_fetch.py          # 2,669 rows across 9 subsets
python3 data-gen/build_sair_index.py    # pair/law class-hash index (all rows)
python3 data-gen/build_eval.py          # eval_v1: 180/197/200/200 per tier
python3 data-gen/build_train.py         # train_v1: 2,772 + 100 holdout
python3 data-gen/verify_artifacts.py    # gate — must print ALL CHECKS PASSED
```

Disjointness (signed off 2026-07-24): pair-class gate vs ALL SAIR rows;
law-class gate vs eval_v1's laws only; hashes collide under the grader's
symmetry group (renaming / side swap / consistent dualization).
