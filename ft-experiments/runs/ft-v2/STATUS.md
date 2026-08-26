# FT v2 — COMPLETE (2026-08-19)

Train on story->RG pairs in one grammar; eval the frozen 777 problems in the
trained grammar, a re-skinned grammar, and a restructured one, vs base.

| Phase | State |
|---|---|
| Design, data, instruments | DONE (v2/DESIGN.md, train_v2 frozen, grammar-B module + tests) |
| Base matrix (3 models x 6 arms) | DONE (14 runs) |
| Training (8B / 14B / 32B) | DONE |
| FT matrix (3 models x 6 arms) | DONE (18 runs) |
| Checkpoint curves | DONE (36 runs; 32B via validated one-engine harness) |
| Format-following controls | DONE (6 runs) |
| Representation probes | DONE (base + FT + FT-unseen subset) |
| Adversarial QC | DONE (19,425 verdicts re-derived; 2 defects fixed) |
| Writeup + figures | DONE (RESULTS.md, v2_transfer.png, v2_dynamics.png, RESEARCH_LOG 26-31) |

Headline: task-pair FT teaches translation, not a format. Grammar A 0.1->96.8
(8B) / 14.0->89.4 (14B) / 18.8->99.7 (32B); re-skinned grammar 94-100%;
restructured grammar 37/50/80% (scales with capacity, fails syntactically).
Internal correctness representation installed in all three (0.52->0.95,
0.61->0.98, 0.60->0.92 law-disjoint). Controls clean everywhere.
