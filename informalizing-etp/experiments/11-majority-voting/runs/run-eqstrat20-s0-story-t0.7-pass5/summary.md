# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=0.7

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 51 | 18 | 0 | 69 | 62 | 0 | 200 | 34.5 | 0 | 0 |
| qwen/qwen3-32b | 19 | 17 | 3 | 111 | 50 | 0 | 200 | 19.5 | 170 | 1 |
| meta-llama/llama-3.3-70b-instruct | 18 | 34 | 2 | 69 | 77 | 0 | 200 | 27.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 18 | 20 | 2 | 79 | 81 | 0 | 200 | 20.0 | 0 | 0 |
