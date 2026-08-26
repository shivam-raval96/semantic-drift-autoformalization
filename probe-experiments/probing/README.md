# probing — linear probes

Per-layer logistic probes (correct vs wrong) on the captured activations, both sites,
out-of-fold AUROC under GroupKFold grouped by problem (twins never straddle a split).
Controls: char-TF-IDF lexical floor on answer text, answer-length probe, layer-0
embeddings, shuffled-label probes (~0.5 required), and a law-disjoint robustness split
(`group_lawcc`; caveat: one 556-problem component).

    cd probe-experiments
    .venv/bin/python probing/fit_probes.py    # CPU, ~10-20 min

Output: `runs/probe-v1/probe_results.json` (committed).
