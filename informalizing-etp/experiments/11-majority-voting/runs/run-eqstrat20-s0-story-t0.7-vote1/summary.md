# Majority vote: k=1, temp=0.7, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 49 | 19 | 0 | 71 | 61 | 0 | 200 | 34.0 | 0 | 0 |
| qwen/qwen3-32b | 28 | 10 | 3 | 125 | 34 | 0 | 200 | 20.5 | 176 | 1 |
| meta-llama/llama-3.3-70b-instruct | 15 | 35 | 1 | 79 | 70 | 0 | 200 | 25.5 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 17 | 26 | 1 | 71 | 85 | 0 | 200 | 22.0 | 0 | 0 |
