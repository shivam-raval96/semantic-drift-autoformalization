# Majority vote: k=3, temp=0.7, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 54 | 20 | 0 | 88 | 38 | 0 | 200 | 37.0 | 0 | 0 |
| qwen/qwen3-32b | 30 | 12 | 3 | 141 | 14 | 0 | 200 | 22.5 | 180 | 1 |
| meta-llama/llama-3.3-70b-instruct | 15 | 36 | 1 | 105 | 43 | 0 | 200 | 26.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 19 | 27 | 1 | 97 | 56 | 0 | 200 | 23.5 | 0 | 0 |
