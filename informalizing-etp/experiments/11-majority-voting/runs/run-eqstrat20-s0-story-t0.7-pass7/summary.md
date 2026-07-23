# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=0.7

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 59 | 12 | 0 | 72 | 57 | 0 | 200 | 35.5 | 0 | 0 |
| qwen/qwen3-32b | 21 | 15 | 2 | 119 | 43 | 0 | 200 | 19.0 | 185 | 1 |
| meta-llama/llama-3.3-70b-instruct | 20 | 30 | 2 | 70 | 78 | 0 | 200 | 26.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 15 | 24 | 1 | 81 | 79 | 0 | 200 | 20.0 | 0 | 0 |
