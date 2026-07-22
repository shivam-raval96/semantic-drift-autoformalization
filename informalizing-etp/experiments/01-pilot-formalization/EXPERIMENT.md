# 01 — Pilot: can models formalize the stories at all?

Run 2026-07-14.

## Question

Does the end-to-end eval loop work — story rendering, OpenRouter querying,
syntactic grading — and can off-the-shelf models formalize question-stories
back into the implication with meaningful (non-floor, non-ceiling) accuracy?

## Setup

- **Models** (4, via OpenRouter):
  - `deepseek/deepseek-chat-v3.1`
  - `qwen/qwen3-32b`
  - `openai/gpt-4o-mini`
  - `google/gemini-2.5-flash`
- **Thinking mode**: uncontrolled. This experiment predates the
  `--reasoning` regime flag; each model/provider used its own default
  (hybrid-reasoning models like DeepSeek and Qwen3 were free to think,
  gpt-4o-mini cannot). This confound is what Experiment 02 was designed
  to remove.
- **Sampling**: seed 0, `--n 30` uniform pairs from the ETP equation list
  (`equations.txt` sha256 `e30e1a67…c8010`). Uniform sampling lands almost
  entirely on maximal-complexity 4+4-operation pairs.
- **Other**: `max_tokens` 4096, temperature 0 (harness default).
- A 2-sample smoke run (`runs/smoke`, gpt-4o-mini only) preceded the real
  run to shake out the harness; kept for provenance only.

## Prompts

- Template: `formalize_prompt.md` at sha256
  `ad33f6de859156b81be0d889abd3c56e4d9275bd855eb6d804d4e8ebcfe4983c`
  (the version currently in the repo reproduces every sent prompt
  byte-exactly). It teaches the `op(first, second)` / `ASSUME:` / `ASK:`
  notation self-containedly and embeds the story at `{story}`.
- No wrapper text was added — the template output was sent verbatim,
  identically to every model.
- The exact prompt for each sample is the `prompt` field in
  `runs/run-s0-n30/samples.jsonl`.

## Reproduce

```sh
python3 benchmark.py --seed 0 --n 30 \
    --models deepseek/deepseek-chat-v3.1,qwen/qwen3-32b,openai/gpt-4o-mini,google/gemini-2.5-flash \
    --out-dir experiments/01-pilot-formalization/runs/run-s0-n30
```

(The original invocation predates the `--reasoning` flag; today's harness
reproduces it by omitting `--reasoning`.)

## Results

Artifacts: `runs/run-s0-n30/` (`summary.md` has the full table). No chart
report was generated; `charts.py` was written for Experiment 02.

| model | correct% (of 30) | exact | swapped | wrong | unparseable |
|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 90.0 | 21 | 6 | 2 | 1 |
| qwen/qwen3-32b | 90.0 | 17 | 10 | 2 | 1 |
| google/gemini-2.5-flash | 60.0 | 13 | 5 | 10 | 2 |
| openai/gpt-4o-mini | 6.7 | 2 | 0 | 25 | 3 |

## Conclusions

- The loop works: responses parse, grading is deterministic, and accuracy
  spans nearly the full range (6.7%–90%), so the task discriminates.
- The models that scored high are exactly the ones that think by default
  (DeepSeek, Qwen3), suggesting the spread measures reasoning budget as
  much as formalization ability — motivating Experiment 02's standardized
  thinking regimes.
- Side-swapped answers are common and correctly accepted; dualization was
  never observed in this sample.
