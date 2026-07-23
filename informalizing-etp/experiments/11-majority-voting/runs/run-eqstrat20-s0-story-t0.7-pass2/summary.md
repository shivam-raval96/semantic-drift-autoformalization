# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=0.7

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 53 | 18 | 0 | 58 | 71 | 0 | 200 | 35.5 | 0 | 0 |
| qwen/qwen3-32b | 21 | 18 | 2 | 113 | 46 | 0 | 200 | 20.5 | 177 | 1 |
| meta-llama/llama-3.3-70b-instruct | 14 | 34 | 1 | 71 | 80 | 0 | 200 | 24.5 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 16 | 23 | 1 | 74 | 86 | 0 | 200 | 20.0 | 0 | 0 |
