# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=0.7

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 55 | 16 | 0 | 65 | 64 | 0 | 200 | 35.5 | 0 | 0 |
| qwen/qwen3-32b | 30 | 9 | 2 | 121 | 38 | 0 | 200 | 20.5 | 192 | 1 |
| meta-llama/llama-3.3-70b-instruct | 17 | 34 | 2 | 74 | 73 | 0 | 200 | 26.5 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 15 | 26 | 0 | 82 | 77 | 0 | 200 | 20.5 | 0 | 0 |
