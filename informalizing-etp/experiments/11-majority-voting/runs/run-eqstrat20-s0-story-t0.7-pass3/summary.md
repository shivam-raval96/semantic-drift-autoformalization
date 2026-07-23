# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=0.7

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 51 | 18 | 0 | 59 | 72 | 0 | 200 | 34.5 | 0 | 0 |
| qwen/qwen3-32b | 25 | 17 | 2 | 102 | 54 | 0 | 200 | 22.0 | 164 | 1 |
| meta-llama/llama-3.3-70b-instruct | 18 | 34 | 1 | 65 | 82 | 0 | 200 | 26.5 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 21 | 20 | 2 | 74 | 83 | 0 | 200 | 21.5 | 0 | 0 |
