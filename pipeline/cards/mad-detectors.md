# MAD trusted-only detectors — grumeter card
(mad.py — mentors' tree, synced on request; frozen protocol: mad-spec.md)

**Measures:** whether certified-drift outputs carry an internal activation
signature detectable WITHOUT anomaly labels — detectors fit on a certified-clean
trusted set only (Mahalanobis / PCA-residual / kNN), scored against certificates.

**Valid within:** Llama-3.1-8B, big-E items, fixed mid-depth layer per site
(no eval-informed selection), the four spec regimes. Exploratory: R1 0.94 AUROC,
recall@1%FPR 0.70; surface floor 0.51.

**Positive means:** the anomaly signature is internal (surface features carry
~nothing on unnamed items); label-free flagging is feasible in-domain.
/ **Does NOT mean:** "mechanism" in ARC's strong sense (extensional definition:
certified semantic failure); transfer across models/domains; robustness to an
adversary optimizing against the detector.

**Null means:** no trusted-support deviation at the probed layer/site — not the
absence of internal signature elsewhere.

**Confounds & controls:** input difficulty (surface-floor control, R3 shift
regime); trusted-set contamination (R2); base-rate inversion — faithful is the
MINORITY at 8B on E, so R4 (5% anomalies) is DEGENERATE in v0 (n_pos=2; its
1.000 rows are artifacts). Risk-coverage (honest-silence axis) + cluster
bootstrap = v0.1, not yet run.
