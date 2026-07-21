# 06 -- GPT-5.5 frontier saturation check

Run 2026-07-21.

## Question

Does a current frontier model still make measurable errors on the direct
Story -> R.G. task, or is the existing benchmark configuration saturated?

## Setup

- **Model**: `openai/gpt-5.5` through OpenRouter.
- **Form**: direct Story -> R.G. (`--form story`).
- **Reasoning**: on, using the same reasoning wrapper as Experiment 02.
- **Prompt**: unmodified `formalize_prompt.md`, sha256
  `ad33f6de859156b81be0d889abd3c56e4d9275bd855eb6d804d4e8ebcfe4983c`.
- **Sampling**: seed 0. The 20-pair pilot is the prefix of Experiment 02's
  uniform 30-pair sample. The confirmatory run uses the exact Experiment 02
  stratified sample: five pairs in each total-operation bin 1--8 (40 total).
- **Limits**: the pilot used 16,384 max tokens; the stratified run used 4,096.
  Every request ended normally, and the largest completion was 1,387 tokens.

## Reproduce

```sh
python3 benchmark.py --seed 0 --n 20 --form story --reasoning on \
    --models openai/gpt-5.5 \
    --out-dir experiments/06-frontier-saturation/runs/run-s0-n20-story-gpt55-think-on

python3 benchmark.py --seed 0 --stratify-ops 5 --form story --reasoning on \
    --models openai/gpt-5.5 --max-tokens 4096 \
    --out-dir experiments/06-frontier-saturation/runs/run-strat5-s0-story-gpt55-think-on

python3 charts.py experiments/06-frontier-saturation/runs/* \
    --out experiments/06-frontier-saturation/report/benchmark-report.html --pdf
```

## Results

| Sample | Exact | Correct-swapped | Wrong | Unparseable | Correct | Median reasoning | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform pilot (20) | 20 | 0 | 0 | 0 | 100% | 594 | $0.64 |
| Stratified 5 x 8 (40) | 39 | 1 | 0 | 0 | 100% | 516 | $1.05 |

The OpenRouter-reported cost was $0.640155 for the 20-pair pilot and
$1.051775 for the 40-pair stratified run, or **$1.691930 total ($1.69)**.

`correct-swapped` exchanges the sides of one equality and is semantically
equivalent under the benchmark's accepted symmetries.

## Conclusions

GPT-5.5 saturated the current direct Story -> R.G. task: it was faithful on
all 60 evaluated rows, including all 40 rows spanning operation-count bins
1--8. Because the direct Story arm is already at ceiling, Literal and
two-stage arms have no accuracy headroom for this model under this setup.
Further pipeline extensions should pause until the benchmark is made harder
or the claim is explicitly limited to weaker or low-reasoning models.

This result diagnoses the current task configuration, not ETP as a whole.
