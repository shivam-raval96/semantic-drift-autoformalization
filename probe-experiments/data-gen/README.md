# data-gen — contrast_v1

Builds the frozen contrastive dataset: 1,000 problems → (story prompt, correct RG,
perturbed-wrong RG), every label verified by checkform before freezing. Deterministic and
seeded; commits the generator + hash manifest. Knobs live in `../config.py`.

Status: not yet implemented (Step 1).
