# Lambda GPU runbook — 8B-class probe replication (~$5-15 of the $200)

Goal: replicate probes.py on Llama-3.1-8B-Instruct — the exact model of Goodfire's
"geometric calculator" post (S3) — so the which-Goodfire-world question is answered
on the model where clean formal geometry is KNOWN to exist. CPU-scale results here
(Qwen 0.5B) establish the pipeline; the 8B run is the citable one.

## One-time setup (any 1×A10/A100 instance, ~10 min)
```bash
git clone <this repo or scp the semdiff_pilot folder>
cd semdiff_pilot
pip install torch transformers scikit-learn numpy
# Llama-3.1-8B-Instruct is gated: either `huggingface-cli login` with an approved
# account, or swap in an ungated equivalent (Qwen/Qwen2.5-7B-Instruct) — probes.py
# and hf_backend.py are model-agnostic.
```

## Run (GPU makes dtype/device automatic via accelerate if installed)
```bash
python etp_items.py 150 0                      # regenerates data-etp.jsonl
python hf_backend.py meta-llama/Llama-3.1-8B-Instruct   # ~10-20 min on GPU
python probes.py                                # probe pair + geometry report
```
Note: hf_backend.py loads float32 CPU by default — on GPU edit the from_pretrained
call to `dtype=torch.bfloat16, device_map='auto'` and move `enc` to the model device
(two-line change, marked by the dtype argument).

## What to look at
- probe-report-hf-*.md: monitor AUROC and intent accuracy, linear vs nonlinear per
  layer. Divergence >0.1 = curvature flag (Rule 1); intent decodable while output
  drifts = ELK-lite positive.
- Compare the silhouette-by-layer curve against the 0.5B run in this folder: does
  law geometry get cleaner with scale (S3-world) or fragment (S4-world)?

## Optional second $: telephone at 8B
`python telephone.py` needs only an OpenRouter key, no GPU — but a LOCAL telephone
run (informalize+formalize with the same open model) closes the loop between the
behavioral survival curves and the activations captured at each hop. That extension
is not yet coded; the chain runner in telephone.py takes any `transport` callable,
so wiring HFBackend into it is ~10 lines.
