# Judge-vs-certificate labels — grumeter card
(judge_labels.py, judgeswap.py, judgeswap-report.md — mentors' tree, synced on request)

**Measures:** what a monitor LEARNS when trained on LLM-judge labels vs
certificate labels — identical activations, identical split, only labels
differ; evaluated against certified truth, stratified by judge-correct /
judge-error items.

**Valid within:** one target model's activations (Llama-3.1-8B), two judges
(sonnet-4.6 acc 0.53, gpt-4o-mini acc 0.77 on this pool), one frozen judge
template, 995 judged pairs, test = 408 candidates / 34 items (cluster CIs).

**Positive means:** the judge-trained monitor is near-perfectly
ANTI-correlated with truth exactly where its judge erred (0.046 on the error
stratum) while the certificate monitor is indifferent to judge strata
(0.91-0.98): the training signal defines the evaluator the monitor becomes —
the ELK human-simulator signature, reproduced where truth is decidable. /
**Does NOT mean:** LLM judges are useless (4o-mini's monitor reached 0.71), or
that this magnitude transfers to richer domains.

**The twist means:** "stronger model = better judge" is
distribution-dependent — sonnet was the WORSE judge here, consistent with its
measured self-report collapse on explicit structure.

**Confounds:** judge accuracy conflated with judge-error STRUCTURE (a
structured 0.53 judge teaches structured error); one template.

**Licenses:** "training-signal-internal labels define the judge, not the
truth" as a structural claim in this lab (pending H4 CI). **Never licenses:**
blanket anti-LLM-judge claims, or extrapolation of exact AUROCs beyond this
pool.
