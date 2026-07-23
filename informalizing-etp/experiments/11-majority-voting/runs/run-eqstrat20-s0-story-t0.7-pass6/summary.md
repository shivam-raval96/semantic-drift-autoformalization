# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=0.7

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 49 | 18 | 0 | 69 | 64 | 0 | 200 | 33.5 | 0 | 0 |
| qwen/qwen3-32b | 30 | 13 | 0 | 123 | 34 | 0 | 200 | 21.5 | 178 | 1 |
| meta-llama/llama-3.3-70b-instruct | 19 | 33 | 1 | 77 | 70 | 0 | 200 | 26.5 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 22 | 22 | 0 | 74 | 82 | 0 | 200 | 22.0 | 0 | 0 |
